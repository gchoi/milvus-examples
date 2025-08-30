from typing import List, Dict

from pymilvus import MilvusClient

from ..conf import Logger

# -- logger settings
logger = Logger(env="dev")


def drop_collection(uri: str, collection_name: str):
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
    embedding_dim: int,
    metric_type: str = "IP",
    consistency_level: str = "Bounded",
    overwrite: bool = False
):
    """
        Create Collection

        Args:
            uri (str):
                Milvus URI.
            collection_name (str):
                milvus에 지정할 collection_name 객체.
            embedding_dim (int):
                Embedding dimension.
            metric_type (str):
                Metric type. (default: ``IP``). Supported types: "IP", "L2", "HAMMING", "JACCARD", "TANIMOTO".
            consistency_level (str):
                Consistency level. (default: ``Bounded``). Supported levels: "Strong", "Session", "Bounded", "Eventually".
            overwrite (bool):
                collection_name에 overwrite 여부 구분자 (default: ``False``)
    """
    if overwrite:
        drop_collection(collection_name=collection_name, uri=uri)

    client = MilvusClient(uri=uri, token="root:Milvus")
    client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type=metric_type,
        consistency_level=consistency_level
    )
    logger.info(f"Collection '{collection_name}' created.")
    return


def insert(uri: str, collection_name: str, data: List[Dict]):
    """
        Create Collection

        Args:
            uri (str):
                Milvus URI.
            collection_name (str):
                milvus에 지정할 collection_name 객체.
            data (List[Dict])):
                Data to be inserted.
    """

    client = MilvusClient(uri=uri, token="root:Milvus")
    client.insert(collection_name=collection_name, data=data)
    logger.info(f"Data inserted into '{collection_name}'.")
    return


def search(
    uri: str,
    collection_name: str,
    query_embeddings: List[List[float]],
    limit: int = 3,
    metric_type: str = "IP"
):
    """
        Search with embeddings.

        Args:
            uri (str):
                Milvus URI.
            collection_name (str):
                milvus에 지정할 collection_name 객체.
            query_embeddings (List[List[float]]):
                Query embeddings.
            limit (int):
                Number of results to return. (default: ``3``)
            metric_type (str):
                Metric type. (default: ``IP``)
    """
    client = MilvusClient(uri=uri, token="root:Milvus")
    search_res = client.search(
        collection_name=collection_name,
        data=query_embeddings,
        limit=limit,
        search_params={"metric_type": metric_type, "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    logger.info(f"Search results: {search_res}")
    return search_res
