from typing import List, Dict, Union, Optional

import numpy as np
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    connections,
    FieldSchema,
    CollectionSchema,
    Collection,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker,
)

from ..conf import Logger

# -- logger settings
logger = Logger(env="dev")


def drop_collection(uri: str, collection_name: str) -> None:
    """
        Drop Collection

        Args:
            uri (str):
                Milvus URI.
            collection_name (str):
                milvus에 지정할 collection_name 객체.
    """
    client = MilvusClient(uri=uri, token="root:Milvus")
    if client.has_collection(collection_name):
        client.drop_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' dropped.")
    else:
        logger.info(f"Collection '{collection_name}' not found, nothing to drop.")
    return


def create_collection(
    uri: str,
    collection_name: str,
    embedding_dim: int = 0,
    dense_dim: int = 0,
    consistency_level: str = "Bounded",
    overwrite: bool = False,
    collection_type: str = "semantic_search",
    dense_search_metric_type: str = "IP",
    sparse_search_metric_type: str = "BM25",
    auto_id: bool = True,
    enable_dynamic_field: bool = True
) -> Optional[MilvusClient | Collection]:
    """
        Create Collection

        Args:
            uri (str):
                Milvus URI.
            collection_name (str):
                milvus에 지정할 collection_name 객체.
            embedding_dim (int):
                Embedding dimension.
            consistency_level (str):
                Consistency level. (default: ``Bounded``). Supported levels: "Strong", "Session", "Bounded", "Eventually".
            overwrite (bool):
                collection_name에 overwrite 여부 구분자 (default: ``False``)
            collection_type (str):
                Collection type: (default: ``vector_search``) "semantic_search" | "full_text_search".
                - semantic_search: only for semantic search (vector search).
                - full_text_search: full text search or hybrid search.
            dense_search_metric_type (str):
                Dense metric type. (default: ``IP``). Supported types: "IP", "L2", "HAMMING", "JACCARD", "TANIMOTO".
            sparse_search_metric_type (str):
                Sparse metric type for text search. (default: ``BM25``)
            dense_dim (int):
                Dense dimension for hybrid search.
            vector_field_name (str):
                Name of the vector field. (default: ``vector``)
            auto_id (bool):
                Whether to use auto-generated IDs. (default: ``True``)
            enable_dynamic_field (bool):
                Whether to enable dynamic fields. (default: ``True``)

        Returns:
            Optional[MilvusClient | Collection]:
                MilvusClient or Collection object.
    """
    client = MilvusClient(uri=uri, token="root:Milvus")

    if overwrite:
        if client.has_collection(collection_name=collection_name):
            logger.info(f"Collection '{collection_name}' removed.")
            client.drop_collection(collection_name=collection_name)

    match collection_type:
        case "semantic_search":
            client.create_collection(
                collection_name=collection_name,
                dimension=embedding_dim,
                metric_type=dense_search_metric_type,
                consistency_level=consistency_level,
            )
            logger.info(f"Collection '{collection_name}' created.")
            return client

        case "full_text_search":
            # -- schema
            schema = client.create_schema()
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100,
            )
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,
                analyzer_params={"tokenizer": "standard", "filter": ["lowercase"]},
                enable_match=True,      # Enable text matching
                enable_analyzer=True,   # Enable text analysis
            )
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=embedding_dim
            )
            schema.add_field(
                field_name="metadata",
                datatype=DataType.JSON
            )
            bm25_function = Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names="sparse_vector",
            )
            schema.add_function(bm25_function)

            # -- index params
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type=sparse_search_metric_type,
            )
            index_params.add_index(
                field_name="dense_vector",
                index_type="FLAT",
                metric_type=dense_search_metric_type
            )

            # Create the collection
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"Collection '{collection_name}' created.")
            return client

        case "hybrid_search":
            # Specify the data schema for the new Collection
            fields = [
                # Use an auto-generated id as a primary key
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
                # Store the original text to retrieve based on semantical distance
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
                # Milvus now supports both sparse and dense vectors,
                # we can store each in a separate field to conduct hybrid search on both vectors
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
            ]
            schema = CollectionSchema(fields)

            # Create a collection (drop the old one if exists)
            connections.connect(uri=uri)
            collection = Collection(name=collection_name, schema=schema, consistency_level="Bounded")

            # To make vector search efficient, we need to create indices for the vector fields
            sparse_index = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": sparse_search_metric_type
            }
            collection.create_index(field_name="sparse_vector", index_params=sparse_index)
            dense_index = {
                "index_type": "AUTOINDEX",
                "metric_type": dense_search_metric_type
            }
            collection.create_index(field_name="dense_vector", index_params=dense_index)
            logger.info(f"Collection '{collection_name}' created.")
            return collection

        case "multimodal_search":
            client.create_collection(
                collection_name=collection_name,
                auto_id=auto_id,
                dimension=embedding_dim,
                enable_dynamic_field=enable_dynamic_field,
            )
            logger.info(f"Collection '{collection_name}' created.")
            return client

        case _:
            raise ValueError(f"Unsupported collection type: {collection_type}")


def insert(uri: str, collection_name: str, data: List[Dict]) -> None:
    """
        Create Collection

        Args:
            uri (str):
                Milvus URI.
            collection_name (str):
                milvus에 지정할 collection_name 객체.
            data (List[Dict]):
                Data to be inserted.
    """

    client = MilvusClient(uri=uri, token="root:Milvus")
    client.insert(collection_name=collection_name, data=data)
    logger.info(f"Data inserted into '{collection_name}'.")
    return


def search(
    uri: str,
    collection_name: str,
    queries: Union[List[List[float]], List[str]],
    query_embeddings: Union[List[List[float]], List[str]],
    limit: int = 3,
    search_type: str = "semantic_search",
    dense_search_metric_type: str = "IP",
    sparse_search_metric_type: str = "BM25",
    output_fields: List[str] = None,
    anns_field: Optional[str] = None
) -> List[List[Dict]]:
    """
        Search with embeddings.

        Args:
            uri (str):
                Milvus URI.
            collection_name (str):
                milvus에 지정할 collection_name 객체.
            queries (Union[List[List[float]], List[str]]):
                List of query text.
            query_embeddings (List[List[float]]):
                List of query embeddings or texts.
            limit (int):
                Number of results to return. (default: ``3``)
            search_type (str):
                Collection type: (default: ``vector_search``) "semantic_search" | "full_text_search" | "combined_search".
            dense_search_metric_type (str):
                Dense metric type for vector search. (default: ``IP``)
            sparse_search_metric_type (str):
                Sparse metric type for text search. (default: ``BM25``)
            output_fields (List[str]):
                Output fields (default: ``text``).
            anns_field (str):
                Anns field (default: ``None``).

        Returns:
            List[List[Dict]]: List of the Pymilvus SearchResult.
    """
    if output_fields is None:
        output_fields = ["text"]
    client = MilvusClient(uri=uri, token="root:Milvus")

    search_res = None
    match search_type:
        case "semantic_search":
            search_res = client.search(
                collection_name=collection_name,
                data=query_embeddings,
                limit=limit,
                search_params={"metric_type": dense_search_metric_type, "params": {}},  # Inner product distance
                output_fields=output_fields,  # Return the text field,
                anns_field=anns_field
            )

        case "full_text_search":
            search_res = client.search(
                collection_name=collection_name,
                data=queries,
                anns_field="sparse_vector",
                limit=limit,
                output_fields=["content", "metadata"],
            )

        case "combined_search":
            sparse_request = AnnSearchRequest(
                data=queries,
                anns_field="sparse_vector",
                param={"metric_type": sparse_search_metric_type},
                limit=limit
            )
            dense_request = AnnSearchRequest(
                data=query_embeddings,
                anns_field="dense_vector",
                param={"metric_type": dense_search_metric_type},
                limit=limit
            )
            search_res = client.hybrid_search(
                collection_name=collection_name,
                reqs=[sparse_request, dense_request],
                ranker=RRFRanker(),  # Reciprocal Rank Fusion for combining results
                limit=limit,
                output_fields=["content", "metadata"],
            )

        case "image_search":
            search_res = client.search(
                collection_name=collection_name,
                data=query_embeddings,
                limit=limit,
                search_params={"metric_type": dense_search_metric_type, "params": {}},  # Inner product distance
                output_fields=output_fields
            )

        case "multimodal_search":
            search_res = client.search(
                collection_name=collection_name,
                data=query_embeddings,
                output_fields=["image_path"],
                limit=limit,  # Max number of search results to return
                search_params={"metric_type": dense_search_metric_type, "params": {}},  # Search parameters
            )

        case _:
            raise ValueError(f"Unsupported collection type: {search_type}")

    logger.info(f"Search results: {search_res}")
    return search_res


def search_from_collection(
    col: Collection,
    search_type: str,
    query_dense_embedding: np.ndarray = np.zeros(1),
    query_sparse_embedding: np.ndarray = np.zeros(1),
    dense_metric_type: str = "IP",
    sparse_metric_type: str = "IP",
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    limit: int = 10,
    params: Optional[Dict] = None
):
    if params is None:
        params = {}

    res = None
    match search_type:
        case "dense_search":
            search_params = {
                "metric_type": dense_metric_type,
                "params": params
            }
            res = col.search(
                data=[query_dense_embedding],
                anns_field="dense_vector",
                param=search_params,
                limit=limit,
                output_fields=["text"],
            )[0]
            res = [hit.get("text") for hit in res]

        case "sparse_search":
            search_params = {
                "metric_type": sparse_metric_type,
                "params": params
            }
            res = col.search(
                data=[query_sparse_embedding],
                anns_field="sparse_vector",
                param=search_params,
                limit=limit,
                output_fields=["text"],
            )[0]
            res = [hit.get("text") for hit in res]

        case "hybrid_search":
            dense_search_params = {
                "metric_type": dense_metric_type,
                "params": params
            }
            sparse_search_params = {
                "metric_type": sparse_metric_type,
                "params": params
            }

            dense_req = AnnSearchRequest(
                data=[query_dense_embedding],
                anns_field="dense_vector",
                param=dense_search_params,
                limit=limit
            )
            sparse_req = AnnSearchRequest(
                data=[query_sparse_embedding],
                anns_field="sparse_vector",
                param=sparse_search_params,
                limit=limit
            )

            rerank = WeightedRanker(sparse_weight, dense_weight)
            res = col.hybrid_search(
                reqs=[sparse_req, dense_req],
                rerank=rerank,
                limit=limit,
                output_fields=["text"]
            )[0]
            res = [hit.get("text") for hit in res]

        case _:
            raise ValueError(f"Unsupported collection type: {search_type}")
    return res
