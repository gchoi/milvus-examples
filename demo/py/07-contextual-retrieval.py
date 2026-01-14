import os
from typing import List, Dict, Any, Callable
import json

from dotenv import load_dotenv
from tqdm import tqdm
from pymilvus.model.dense import VoyageEmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import CohereRerankFunction
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)
import anthropic

from milvus.conf import Logger
from milvus.utils import get_configurations


# -- logger settings
logger = Logger(env="dev")

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['VOYAGE_API_KEY'] = os.getenv('VOYAGE_API_KEY')
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')


class MilvusContextualRetriever:
    def __init__(
        self,
        uri: str = "milvus.db",
        collection_name: str = "contexual_retrieval",
        dense_embedding_function=None,
        use_sparse=False,
        sparse_embedding_function=None,
        use_contextualize_embedding=False,
        anthropic_client=None,
        use_reranker=False,
        rerank_function=None,
    ):
        self.collection_name = collection_name

        self.client = MilvusClient(uri)

        self.embedding_function = dense_embedding_function

        self.use_sparse = use_sparse
        self.sparse_embedding_function = None

        self.use_contextualize_embedding = use_contextualize_embedding
        self.anthropic_client = anthropic_client

        self.use_reranker = use_reranker
        self.rerank_function = rerank_function

        if use_sparse is True and sparse_embedding_function:
            self.sparse_embedding_function = sparse_embedding_function
        elif sparse_embedding_function is False:
            raise ValueError(
                "Sparse embedding function cannot be None if use_sparse is False"
            )
        else:
            pass

    def build_collection(self):
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_function.dim,
        )
        if self.use_sparse:
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )
        if self.use_sparse:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

    def insert_data(self, chunk, metadata):
        dense_vec = self.embedding_function([chunk])[0]
        if self.use_sparse:
            sparse_result = self.sparse_embedding_function.encode_documents([chunk])
            if type(sparse_result) == dict:
                sparse_vec = sparse_result["sparse"][[0]]
            else:
                sparse_vec = sparse_result[[0]]
            self.client.insert(
                collection_name=self.collection_name,
                data={
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    **metadata,
                },
            )
        else:
            self.client.insert(
                collection_name=self.collection_name,
                data={"dense_vector": dense_vec, **metadata},
            )

    def insert_contextualized_data(self, doc, chunk, metadata):
        contextualized_text, usage = self.situate_context(doc, chunk)
        metadata["context"] = contextualized_text
        text_to_embed = f"{chunk}\n\n{contextualized_text}"
        dense_vec = self.embedding_function([text_to_embed])[0]
        if self.use_sparse:
            sparse_vec = self.sparse_embedding_function.encode_documents(
                [text_to_embed]
            )["sparse"][[0]]
            self.client.insert(
                collection_name=self.collection_name,
                data={
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    **metadata,
                },
            )
        else:
            self.client.insert(
                collection_name=self.collection_name,
                data={"dense_vector": dense_vec, **metadata},
            )

    def situate_context(self, doc: str, chunk: str):
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        response = self.anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            "cache_control": {
                                "type": "ephemeral"
                            },  # we will make use of prompt caching for the full documents
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        },
                    ],
                },
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        return response.content[0].text, response.usage

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        dense_vec = self.embedding_function([query])[0]
        if self.use_sparse:
            sparse_vec = self.sparse_embedding_function.encode_queries([query])[
                "sparse"
            ][[0]]

        req_list = []
        if self.use_reranker:
            k = k * 10
        if self.use_sparse:
            req_list = []
            dense_search_param = {
                "data": [dense_vec],
                "anns_field": "dense_vector",
                "param": {"metric_type": "IP"},
                "limit": k * 2,
            }
            dense_req = AnnSearchRequest(**dense_search_param)
            req_list.append(dense_req)

            sparse_search_param = {
                "data": [sparse_vec],
                "anns_field": "sparse_vector",
                "param": {"metric_type": "IP"},
                "limit": k * 2,
            }
            sparse_req = AnnSearchRequest(**sparse_search_param)

            req_list.append(sparse_req)

            docs = self.client.hybrid_search(
                self.collection_name,
                req_list,
                RRFRanker(),
                k,
                output_fields=[
                    "content",
                    "original_uuid",
                    "doc_id",
                    "chunk_id",
                    "original_index",
                    "context",
                ],
            )
        else:
            docs = self.client.search(
                self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                limit=k,
                output_fields=[
                    "content",
                    "original_uuid",
                    "doc_id",
                    "chunk_id",
                    "original_index",
                    "context",
                ],
            )
        if self.use_reranker and self.use_contextualize_embedding:
            reranked_texts = []
            reranked_docs = []
            for i in range(k):
                if self.use_contextualize_embedding:
                    reranked_texts.append(
                        f"{docs[0][i]['entity']['content']}\n\n{docs[0][i]['entity']['context']}"
                    )
                else:
                    reranked_texts.append(f"{docs[0][i]['entity']['content']}")
            results = self.rerank_function(query, reranked_texts)
            for result in results:
                reranked_docs.append(docs[0][result.index])
            docs[0] = reranked_docs
        return docs


def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    retrieval_function: Callable,
    db, k: int = 20
) -> Dict[str, float]:
    total_score = 0
    total_queries = len(queries)
    for query_item in tqdm(queries, desc="Evaluating retrieval"):
        query = query_item["query"]
        golden_chunk_uuids = query_item["golden_chunk_uuids"]

        # Find all golden chunk contents
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next(
                (
                    doc
                    for doc in query_item["golden_documents"]
                    if doc["uuid"] == doc_uuid
                ),
                None,
            )
            if not golden_doc:
                print(f"Warning: Golden document not found for UUID {doc_uuid}")
                continue

            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc["chunks"]
                    if chunk["index"] == chunk_index
                ),
                None,
            )
            if not golden_chunk:
                print(
                    f"Warning: Golden chunk not found for index {chunk_index} in document {doc_uuid}"
                )
                continue

            golden_contents.append(golden_chunk["content"].strip())

        if not golden_contents:
            print(f"Warning: No golden contents found for query: {query}")
            continue

        retrieved_docs = retrieval_function(query, db, k=k)

        # Count how many golden chunks are in the top k retrieved documents
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[0][:k]:
                retrieved_content = doc["entity"]["content"].strip()
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break

        query_score = chunks_found / len(golden_contents)
        total_score += query_score

    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {
        "pass_at_n": pass_at_n,
        "average_score": average_score,
        "total_queries": total_queries,
    }


def retrieve_base(query: str, db, k: int = 20) -> List[Dict[str, Any]]:
    return db.search(query, k=k)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def evaluate_db(db, original_jsonl_path: str, k):
    # Load the original JSONL data for queries and ground truth
    original_data = load_jsonl(original_jsonl_path)

    # Evaluate retrieval
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"Total Score: {results['average_score']}")
    print(f"Total queries: {results['total_queries']}")


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_graph_rag.yaml")
    configs = get_configurations(config_yaml_path=config_path)

    # -- Milvus configurations
    uri = f"{configs.get('milvus').get('host')}:{configs.get('milvus').get('port')}"
    if not uri.startswith("http://"):
        uri = f"http://{uri}"

    dense_ef = VoyageEmbeddingFunction(api_key=os.getenv('VOYAGE_API_KEY'), model_name="voyage-2")
    sparse_ef = BGEM3EmbeddingFunction()
    cohere_rf = CohereRerankFunction(api_key=os.getenv('COHERE_API_KEY'))

    json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "codebase_chunks.json")
    with open(json_path, "r") as f:
        dataset = json.load(f)


    ########################################################################
    # Experiment I: Standard Retrieval
    ########################################################################

    standard_retriever = MilvusContextualRetriever(
        uri=uri,
        collection_name=configs.get("milvus").get("collection_name"),
        dense_embedding_function=dense_ef
    )

    standard_retriever.build_collection()
    for doc in dataset:
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            standard_retriever.insert_data(chunk_content, metadata)
    evaluate_db(standard_retriever, "evaluation_set.jsonl", 5)


    ########################################################################
    # Experiment II: Hybrid Retrieval
    ########################################################################

    hybrid_retriever = MilvusContextualRetriever(
        uri=uri,
        collection_name="hybrid",
        dense_embedding_function=dense_ef,
        use_sparse=True,
        sparse_embedding_function=sparse_ef,
    )

    hybrid_retriever.build_collection()
    for doc in dataset:
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            hybrid_retriever.insert_data(chunk_content, metadata)

    return


if __name__ == "__main__":
    main()
