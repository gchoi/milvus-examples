import os
import pathlib

import pandas as pd

from milvus.client.utils import create_collection, insert, search
from milvus.model import Model
from milvus.utils import get_configurations
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)

from milvus.utils import run_command


MAX_TRIALS = 10
ROOT = pathlib.Path(__file__).resolve().parent


def dense_search(col, query_dense_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]


def sparse_search(col, query_sparse_embedding, limit=10):
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]


def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]


def doc_text_formatting(ef, query, docs):
    tokenizer = ef.model.tokenizer
    query_tokens_ids = tokenizer.encode(query, return_offsets_mapping=True)
    query_tokens = tokenizer.convert_ids_to_tokens(query_tokens_ids)
    formatted_texts = []

    for doc in docs:
        ldx = 0
        landmarks = []
        encoding = tokenizer.encode_plus(doc, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])[1:-1]
        offsets = encoding["offset_mapping"][1:-1]
        for token, (start, end) in zip(tokens, offsets):
            if token in query_tokens:
                if len(landmarks) != 0 and start == landmarks[-1]:
                    landmarks[-1] = end
                else:
                    landmarks.append(start)
                    landmarks.append(end)
        close = False
        formatted_text = ""
        for i, c in enumerate(doc):
            if ldx == len(landmarks):
                pass
            elif i == landmarks[ldx]:
                if close:
                    formatted_text += "</span>"
                else:
                    formatted_text += "<span style='color:red'>"
                close = not close
                ldx = ldx + 1
            formatted_text += c
        if close is True:
            formatted_text += "</span>"
        formatted_texts.append(formatted_text)
    return formatted_texts


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_hybrid_search.yaml")
    configs = get_configurations(config_yaml_path=config_path)

    # -- Milvus configurations
    uri = f"{configs.get('milvus').get('host')}:{configs.get('milvus').get('port')}"
    if not uri.startswith("http://"):
        uri = f"http://{uri}"
    collection_name = configs.get("milvus").get("collection_name")

    # -- Model configurations
    model = Model(
        platform=configs.get("model").get("platform"),
        embedding_model=configs.get("model").get("embedding_model"),
        chat_model=configs.get("model").get("chat_model"),
    )


    ########################################################################
    # Download Dataset
    ########################################################################

    file_path = "quora_duplicate_questions.tsv"
    if not os.path.exists(path=file_path):
        run_command(cmd="wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv", cwd=ROOT)


    ########################################################################
    # Load and Prepare Data
    ########################################################################

    df = pd.read_csv(file_path, sep="\t")
    questions = set()
    for _, row in df.iterrows():
        obj = row.to_dict()
        questions.add(obj["question1"][:512])
        questions.add(obj["question2"][:512])
        if len(questions) > 500:  # Skip this if you want to use the full dataset
            break

    docs = list(questions)

    print(f"Example question: {docs[0]}")


    ########################################################################
    # Use BGE-M3 Model for Embeddings
    ########################################################################

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim["dense"]

    docs_embeddings = ef(docs)

    collection = create_collection(
        uri=uri,
        collection_name=collection_name,
        dense_dim=dense_dim,
        consistency_level="Bounded",
        overwrite=True,
        collection_type="hybrid_search",
        dense_search_metric_type="IP",
        sparse_search_metric_type="IP"
    )
    collection.load()

    ########################################################################
    # Insert Data into Milvus Collection
    ########################################################################
    for i in range(0, len(docs), 50):
        batched_entities = [
            docs[i : i + 50],
            docs_embeddings["sparse"][i : i + 50],
            docs_embeddings["dense"][i : i + 50],
        ]
        collection.insert(data=batched_entities)
    print("Number of entities inserted:", collection.num_entities)


    ########################################################################
    # Enter Your Search Query
    ########################################################################

    query = "How to start learning programming?"
    query_embeddings = ef([query])

    print(f"query: {query}")
    print(f"query_embeddings: {query_embeddings}")

    ########################################################################
    # Run the Search
    ########################################################################

    dense_results = dense_search(col=collection, query_dense_embedding=query_embeddings["dense"][0])
    sparse_results = sparse_search(col=collection, query_sparse_embedding=query_embeddings["sparse"][[0]])
    hybrid_results = hybrid_search(
        col=collection,
        query_dense_embedding=query_embeddings["dense"][0],
        query_sparse_embedding=query_embeddings["sparse"][[0]],
        sparse_weight=0.7,
        dense_weight=1.0,
    )

    ########################################################################
    # Display Search Results
    ########################################################################

    # Dense search results
    print("**Dense Search Results:**")
    formatted_results = doc_text_formatting(ef=ef, query=query, docs=dense_results)
    for result in dense_results:
        print(result)

    # Sparse search results
    print("\n**Sparse Search Results:**")
    formatted_results = doc_text_formatting(ef=ef, query=query, docs=sparse_results)
    for result in formatted_results:
        print(result)

    # Hybrid search results
    print("\n**Hybrid Search Results:**")
    formatted_results = doc_text_formatting(ef=ef, query=query, docs=hybrid_results)
    for result in formatted_results:
        print(result)
    return


if __name__ == "__main__":
    main()
