from typing import List
import urllib.parse
import urllib

from pymilvus import utility, connections
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings


def create_collection(
        docs: List[Document],
        collection_name: str, 
        embedding: Embeddings, 
        uri: str,
        overwrite: bool = False
):
    """Create Collection

    Parameters
    ----------
        docs: List[Document]
            문서 리스트
        collection_name: str
            milvus에 지정할 collection_name 객체
        embedding: Embeddings
            embedding 객체
        uri: str
            Milvus URI
        overwrite: bool
            collection_name에 overwrite 여부 구분자, by default False
    """
    if overwrite:
        uri_ = urllib.parse.urlsplit(uri)
        connections.connect("default", host=uri_.hostname, port=uri_.port)
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
    Milvus.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=collection_name,
        connection_args={"uri":uri}
    )
