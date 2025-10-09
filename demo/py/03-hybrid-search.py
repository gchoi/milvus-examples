import os
import pathlib

import pandas as pd

from milvus.client.utils import create_collection, search_from_collection
from milvus.utils import get_configurations
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from milvus.utils import run_command
from milvus.conf import Logger


# -- logger settings
logger = Logger(env="dev")

MAX_TRIALS = 10
ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = os.path.join("..", "..", "data")


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
        if close:
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
    uri = configs.get('milvus').get("uri")
    collection_name = configs.get("milvus").get("collection_name")


    ########################################################################
    # Download Dataset
    ########################################################################

    FILENAME = "quora_duplicate_questions.tsv"
    DATA_DEST = os.path.join(DATA_DIR, FILENAME)
    if not os.path.exists(path=DATA_DEST):
        logger.info(f"Downloading {FILENAME}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        cmd = f"wget http://qim.fs.quoracdn.net/{FILENAME}"
        run_command(cmd=cmd, cwd=ROOT)
        cmd = f"mv {FILENAME} {DATA_DIR}/{FILENAME}"
        run_command(cmd=cmd, cwd=ROOT)


    ########################################################################
    # Load and Prepare Data
    ########################################################################

    df = pd.read_csv(DATA_DEST, sep="\t")
    questions = set()
    for _, row in df.iterrows():
        obj = row.to_dict()
        questions.add(obj["question1"][:512])
        questions.add(obj["question2"][:512])
        if len(questions) > 500:  # Skip this if you want to use the full dataset
            break

    docs = list(questions)
    logger.info(f"Example question: {docs[0]}")


    ########################################################################
    # Use BGE-M3 Model for Embeddings
    ########################################################################

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim.get("dense")

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
    batched_entities = []
    for i in range(0, len(docs), 50):
        batched_entities.append(
            {
                "text": docs[i],
                "sparse_vector": docs_embeddings.get("sparse")[i],
                "dense_vector": docs_embeddings.get("dense")[i]
            }
        )
        batched_entities = [
            docs[i : i + 50],
            docs_embeddings["sparse"][i : i + 50],
            docs_embeddings["dense"][i : i + 50],
        ]
        logger.debug(collection.insert(data=batched_entities))
    logger.info(f"Number of entities inserted: {collection.num_entities}")


    ########################################################################
    # Enter Your Search Query
    ########################################################################

    query = "How to start learning programming?"
    query_embeddings = ef([query])

    logger.debug(f"query: {query}")
    logger.debug(f"query_embeddings: {query_embeddings}")

    ########################################################################
    # Run the Search
    ########################################################################

    dense_results = search_from_collection(
        col=collection,
        search_type="dense_search",
        query_dense_embedding=query_embeddings["dense"][0],
        dense_metric_type="IP"
    )
    sparse_results = search_from_collection(
        col=collection,
        search_type="sparse_search",
        query_sparse_embedding=query_embeddings["sparse"][[0]],
        sparse_metric_type="IP"
    )
    hybrid_results = search_from_collection(
        col=collection,
        search_type="hybrid_search",
        query_dense_embedding=query_embeddings["dense"][0],
        query_sparse_embedding=query_embeddings["sparse"][[0]],
        dense_metric_type="IP",
        sparse_metric_type="IP",
        dense_weight=1.0,
        sparse_weight=0.7
    )

    ########################################################################
    # Display Search Results
    ########################################################################

    # Dense search results
    print()
    print("*" * 100)
    print("Dense Search Results:")
    print("*" * 100)
    formatted_results = doc_text_formatting(ef=ef, query=query, docs=dense_results)
    for result in formatted_results:
        print(result)

    # Sparse search results
    print()
    print("*" * 100)
    print("Sparse Search Results:")
    print("*" * 100)
    formatted_results = doc_text_formatting(ef=ef, query=query, docs=sparse_results)
    for result in formatted_results:
        print(result)

    # Hybrid search results
    print()
    print("*" * 100)
    print("Hybrid Search Results:")
    print("*" * 100)
    formatted_results = doc_text_formatting(ef=ef, query=query, docs=hybrid_results)
    for result in formatted_results:
        print(result)
    return


if __name__ == "__main__":
    main()
