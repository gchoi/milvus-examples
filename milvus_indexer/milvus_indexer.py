import logging
from typing import Any, Dict, List, Optional

from langchain.indexes import SQLRecordManager
from langchain.indexes import index as langchain_index_func
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility
from pymilvus import db as milvus_db

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exceptions for MilvusIndexer
class MilvusIndexerError(Exception):
    """Base exception for MilvusIndexer errors."""
    pass

class CollectionNotFoundError(MilvusIndexerError):
    """Raised when a collection does not exist."""
    pass

class MilvusConnectionError(MilvusIndexerError):
    """Raised when there is an error connecting to Milvus."""
    pass

class QueryError(MilvusIndexerError):
    """Raised when there is an error executing a query."""
    pass

class IndexingError(MilvusIndexerError):
    """Raised when there is an error indexing documents."""
    pass

class DeletionError(MilvusIndexerError):
    """Raised when there is an error deleting documents."""
    pass

class InvalidDocumentError(MilvusIndexerError):
    """Raised when a document is invalid. (ex, missing source_id_key)"""
    pass


class MilvusIndexer:
    """
    A class to handle Milvus collections, Langchain vector stores,
    and SQLRecordManager for document indexing and retrieval.

    This class simplifies common RAG operations such as connecting to Milvus,
    defining a collection schema, indexing documents, performing similarity
    searches, and managing the lifecycle of indexed data.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_function: Embeddings,
        milvus_uri: str = "http://localhost:19530",
        milvus_db_name: str = "default",
        enable_dynamic_field: bool = False,
        index_params: Optional[Dict[str, Any]] = None,
        consistency_level: str = "Strong",
        record_manager_db_url: str = "sqlite:///record_manager_cache.sql",
        drop_old_collection: bool = False,
        milvus_connection_alias: str = "default_milvus_conn",
        connection_timeout: Optional[float] = 10.0,
        source_id_key: str = "source",
    ):
        """Initializes the MilvusIndexer.

        Args:
            collection_name (str): Name for the Milvus collection.
            embedding_function (Embeddings): Langchain Embeddings model.
            milvus_uri (str, optional): URI of the Milvus server.
                Defaults to "http://localhost:19530".
            milvus_db_name (str, optional): Milvus database name.
                Defaults to "default".
            enable_dynamic_field (bool, optional): Enable dynamic field in Milvus collection.
                If True, fields can be added dynamically. If False, all fields must be
                defined in the schema. Defaults to False.
            index_params (Optional[Dict[str, Any]], optional): Specific parameters for Milvus index.
                Defaults to None, which uses a default IVF_FLAT index:
                `{\"metric_type\": \"L2\", \"index_type\": \"IVF_FLAT\", \"params\": {\"nlist\": 128}}`.
            consistency_level (str, optional): Milvus consistency level.
                Defaults to "Strong".
            record_manager_db_url (str, optional): DB URL for SQLRecordManager.
                Defaults to "sqlite:///record_manager_cache.sql".
            drop_old_collection (bool, optional): Drop and recreate collection if True.
                Defaults to False.
            milvus_connection_alias (str, optional): Alias for this Milvus connection.
                Defaults to "default_milvus_conn".
            connection_timeout (Optional[float], optional): Timeout in seconds for
                establishing Milvus connection. Defaults to 10.0.
            source_id_key (str, optional): The key in document metadata to use for
                source tracking and cleanup. Defaults to "source".
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.milvus_uri = milvus_uri
        self.db_name = milvus_db_name
        self.enable_dynamic_field = enable_dynamic_field

        _default_index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128} # Number of clusters for IVF index
        }
        self.user_index_params = index_params or _default_index_params
        self.consistency_level = consistency_level
        self.source_id_key = source_id_key

        self.record_manager_db_url = record_manager_db_url
        self.record_manager_namespace = f"{self.db_name}/{self.collection_name}"

        self.drop_old_collection = drop_old_collection
        self.milvus_connection_alias = milvus_connection_alias
        self.connection_timeout = connection_timeout

        self._connect_milvus()
        self._setup_database()

        self.vectorstore = self._init_vectorstore()
        self.record_manager = self._init_record_manager()

        logger.info(f"Initialized for collection '{self.collection_name}' in DB '{self.db_name}'.")


    def _setup_database(self) -> None:
        """Setup the target database, creating it if needed."""
        try:
            db_list = milvus_db.list_database(using=self.milvus_connection_alias)
            if self.db_name not in db_list:
                milvus_db.create_database(self.db_name, using=self.milvus_connection_alias)
                logger.debug(f"Created database '{self.db_name}'.")
            else:
                logger.debug(f"Using existing database '{self.db_name}'.")

            milvus_db.using_database(self.db_name, using=self.milvus_connection_alias)
            logger.debug(f"Connected to database '{self.db_name}'.")
        except Exception as e:
            logger.error(f"Error in database setup: {e}")
            raise MilvusConnectionError(f"Failed to setup database '{self.db_name}': {e}") from e

    def _connect_milvus(self) -> None:
        """Establishes connection to the Milvus server and database."""
        try:
            if not connections.has_connection(self.milvus_connection_alias):
                connections.connect(
                    alias=self.milvus_connection_alias,
                    uri=self.milvus_uri,
                    timeout=self.connection_timeout
                )
                logger.debug(f"Connected to Milvus at '{self.milvus_uri}' with alias '{self.milvus_connection_alias}'.")

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
            raise MilvusConnectionError(f"Failed to connect to Milvus at '{self.milvus_uri}': {e}") from e

    def _init_vectorstore(self) -> Milvus:
        """Initializes the Langchain Milvus vector store."""
        try:
            vectorstore = Milvus(
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
                connection_args={"db_name": self.db_name},
                consistency_level=self.consistency_level,
                enable_dynamic_field=self.enable_dynamic_field,
                index_params=self.user_index_params,
                drop_old=self.drop_old_collection,
            )
            logger.debug("Milvus vectorstore initialized.")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to initialize Milvus vectorstore: {e}", exc_info=True)
            raise IndexingError(f"Failed to initialize Milvus vectorstore: {e}") from e

    def _init_record_manager(self) -> SQLRecordManager:
        """Initializes the SQLRecordManager."""
        try:
            rm = SQLRecordManager(
                namespace=self.record_manager_namespace,
                db_url=self.record_manager_db_url
            )
            rm.create_schema()
            logger.debug(f"SQLRecordManager initialized with namespace '{self.record_manager_namespace}'.")
            return rm
        except Exception as e:
            logger.error(f"Failed to initialize SQLRecordManager: {e}", exc_info=True)
            raise IndexingError("Failed to initialize SQLRecordManager with namespace "
                                f"'{self.record_manager_namespace}': {e}") from e

    def index_documents(
        self,
        documents: List[Document],
        cleanup: Optional[str] = "incremental", # "full", "incremental", "scoped_full", None
    ) -> Dict[str, Any]:
        """
        Indexes documents into Milvus using Langchain's indexing mechanism.

        This method handles deduplication and updates based on the SQLRecordManager.

        Args:
            documents (List[Document]): A list of Langchain Document objects to index.
            cleanup (Optional[str]): Cleanup mode for the record manager.
                "incremental": Adds new, updates changed, deletes missing from current batch if source_id_key matches.
                "full": Deletes all previous records for this namespace then adds current batch.
                "scoped_full": Deletes records matching source_id_key values in the current batch, then adds.
                None: No cleanup, just adds.

        Returns:
            Dict[str, Any]: A dictionary containing indexing statistics (num_added, num_updated, etc.).
        """
        if not documents:
            logger.info("No documents provided for indexing.")
            return {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

        try:
            logger.info(f"Indexing {len(documents)} documents with cleanup mode '{cleanup}'...")

            if cleanup in ["incremental", "scoped_full"]:
                for doc in documents:
                    if self.source_id_key not in doc.metadata:
                        raise InvalidDocumentError(f"Document metadata must contain '{self.source_id_key}'.")

            logger.debug(f"Starting indexing of {len(documents)} documents with cleanup mode '{cleanup}'.")
            indexing_result = langchain_index_func(
                docs_source=documents,
                record_manager=self.record_manager,
                vector_store=self.vectorstore,
                cleanup=cleanup,
                source_id_key=self.source_id_key,
            )

            logger.info(f"Indexed: +{indexing_result.get('num_added',0)}, ~{indexing_result.get('num_updated',0)}, "
                                f"={indexing_result.get('num_skipped',0)}, -{indexing_result.get('num_deleted',0)}")
            return indexing_result
        except Exception as e:
            logger.error(f"Error indexing documents: {e}", exc_info=True)
            raise IndexingError(f"Failed to index documents: {e}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_expr: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Performs a similarity search in the vector store.

        Args:
            query (str): The query string.
            k (int, optional): The number of results to return. Defaults to 4.
            filter_expr (Optional[str], optional): Optional Milvus boolean expression for metadata filtering.
                Example: "book_name == 'The Great Gatsby' and chapter > 5"
            search_params (Optional[Dict[str, Any]], optional): Optional parameters for search algorithm
                (e.g., {'ef': 128} for HNSW).
            **kwargs: Additional arguments for the underlying vector store's search method.

        Returns:
            List[Document]: A list of relevant Document objects.
        """
        logger.debug(f"Search query: '{query[:30]}...' (k={k}).")
        search_kwargs = {"k": k}
        if filter_expr:
            search_kwargs["expr"] = filter_expr
        if search_params:
            search_kwargs["params"] = search_params
        search_kwargs.update(kwargs)

        # Langchain Milvus uses 'expr' for filtering in its search methods
        results = self.vectorstore.similarity_search(query, **search_kwargs)
        logger.debug(f"Found {len(results)} results for search.")
        return results

    def delete_documents_by_sources(
        self,
        sources: List[str]
    ) -> int:
        """
        Deletes documents from Milvus and updates the record manager based on a list of source metadata.
        Iteratively fetches and deletes documents in batches if many documents match the given sources.

        Args:
            sources (List[str]): A list of source strings to delete.
                Documents whose 'source' metadata field matches any of these strings will be deleted.

        Returns:
            int: The total number of documents deleted from Milvus.
        """
        if not sources:
            logger.warning("No sources provided for deletion.")
            return 0

        logger.debug(f"Deleting {len(sources)} docs by sources: {sources[:3]}...")
        try:
            # Step 1: Find all documents with the given sources and get their primary keys
            expr = " or ".join([f"source == '{source}'" for source in sources])
            pks_to_delete = self.__get_pks_by_sources_via_query(sources, expr)
            if not pks_to_delete:
                logger.warning("No documents found to delete for sources: {sources}")
                return 0

            # Step 2: Delete from Milvus vector store for the current batch
            self.vectorstore.delete(ids=pks_to_delete)

            # Step 3: Update the SQLRecordManager to maintain consistency
            pks_in_batch_str = [str(pk) for pk in pks_to_delete]
            self.record_manager.delete_keys(pks_in_batch_str)

            logger.info(f"Deleted {len(pks_to_delete)} documents.")
            return len(pks_to_delete)
        except Exception as e:
            logger.error(f"Error deleting documents by sources: {e}", exc_info=True)
            raise e

    def __get_pks_by_sources_via_query(self, sources: List[str], expr: str) -> List[str]:
        results = self.vectorstore.col.query(
            expr=expr,
            output_fields=[self.vectorstore._primary_field],
            offset=0)
        return [pk[self.vectorstore._primary_field] for pk in results]

    def __get_pks_by_sources_via_search(self, sources: List[str], expr: str) -> List[str]:
        pks = []
        batch_size = 10000
        while True:
            docs = self.vectorstore.similarity_search(query="", expr=expr, k=batch_size)
            if not docs:
                break

            pks_in_batch = [doc.metadata["pk"] for doc in docs if "pk" in doc.metadata]
            if not pks_in_batch:
                break

            pks.extend(pks_in_batch)

            if len(docs) < batch_size:
                break
        return pks

    def clear_all_indexed_data(self) -> Dict[str, Any]:
        """
        Clears all data indexed by the SQLRecordManager for this collection's namespace
        from the vector store. This effectively empties the collection as tracked by
        the record manager.

        Returns:
            Dict[str, Any]: Indexing statistics from the cleanup operation.
        """
        try:
            logger.info(f"Clearing all data for namespace '{self.record_manager_namespace}'.")
            # This uses the "full" cleanup mode with an empty document list
            # to remove all documents associated with this record_manager's namespace.
            result = langchain_index_func(
                [], # Empty list of documents
                self.record_manager,
                self.vectorstore,
                cleanup="full",
                source_id_key=self.source_id_key, # Needs a source_id_key even if docs are empty
            )
            logger.debug("All indexed data cleared.")
            return result
        except Exception as e:
            logger.error(f"Error clearing indexed data: {e}", exc_info=True)
            raise DeletionError(f"Failed to clear indexed data: {e}") from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retrieves statistics for the current Milvus collection.

        Returns:
            Dict[str, Any]: A dictionary containing statistics about the collection.
        """
        try:
            if not utility.has_collection(self.collection_name, using=self.milvus_connection_alias):
                logger.warning(f"Collection '{self.collection_name}' does not exist.")
                raise CollectionNotFoundError(f"Collection '{self.collection_name}' does not exist")

            _collection = self.vectorstore.col
            _collection.flush() # Ensure all data is written to disk for accurate count
            stats = {
                "name": _collection.name,
                "description": _collection.description,
                "is_empty": _collection.is_empty,
                "num_entities": _collection.num_entities,
                "primary_field": _collection.primary_field.name,
                "aliases": _collection.aliases,
                "schema_fields": [f.name for f in _collection.schema.fields],
                "indexes": [idx.field_name for idx in _collection.indexes],
            }
            logger.debug(f"Collection '{self.collection_name}': {stats['num_entities']} entities.")
            return stats
        except CollectionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving collection stats: {e}", exc_info=True)
            raise MilvusIndexerError(f"Failed to retrieve collection stats: {e}") from e

    def drop_collection(self) -> None:
        """
        Drops the Milvus collection. This action is irreversible.
        Before dropping, all indexed data (vectors & record manager metadata) are cleared for this namespace.
        The SQLRecordManager namespace itself is NOT automatically dropped.
        """
        try:
            logger.warning(f"Preparing to clear all indexed data for '{self.collection_name}' before drop.")
            try:
                self.clear_all_indexed_data()
            except Exception as cleanup_exc:
                logger.error(f"Failed to clear indexed data before dropping collection: {cleanup_exc}")
                # Proceeding to drop collection anyway, as requested
            logger.warning(f"Attempting to drop collection '{self.collection_name}' completely. This is irreversible.")
            if utility.has_collection(self.collection_name, using=self.milvus_connection_alias):
                utility.drop_collection(self.collection_name, using=self.milvus_connection_alias)
                logger.info(f"Collection '{self.collection_name}' dropped.")
            else:
                logger.warning(f"Collection '{self.collection_name}' not found, nothing to drop.")
                raise CollectionNotFoundError(f"Collection '{self.collection_name}' does not exist")
        except CollectionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error dropping collection '{self.collection_name}': {e}", exc_info=True)
            raise MilvusIndexerError(f"Failed to drop collection '{self.collection_name}': {e}") from e

    def disconnect(self) -> None:
        """Disconnects from the Milvus server for the current alias."""
        try:
            connections.disconnect(self.vectorstore.client._using)
            connections.disconnect(self.vectorstore.aclient._using)
            connections.disconnect(self.milvus_connection_alias)
            logger.debug(f"Disconnected from Milvus (alias: '{self.milvus_connection_alias}').")
        except Exception as e:
            logger.warning(f"Error during disconnection for alias '{self.milvus_connection_alias}': {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

def _drop_all_dbs():
    try:
        connections.connect(uri="http://localhost:19530", timeout=3.0)
        for db_name in milvus_db.list_database():
            milvus_db.using_database(db_name)
            for collection_name in utility.list_collections():
                utility.drop_collection(collection_name)
            if db_name != "default":
                milvus_db.drop_database(db_name)
    except Exception as e:
        logger.error(f"Error dropping all databases: {e}")
