import os
import time

from milvus.client.utils import (
    create_collection,
    insert,
    search
)
from milvus.utils import get_configurations
from milvus.model import Model


MAX_TRIALS = 10


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_full_text_search.yaml")
    configs = get_configurations(config_yaml_path=config_path)

    # -- Milvus configurations
    uri = f"{configs.get("milvus").get("host")}:{configs.get("milvus").get("port")}"
    if not uri.startswith("http://"):
        uri = f"http://{uri}"
    collection_name = configs.get("milvus").get("collection_name")

    # -- Model configurations
    model = Model(
        platform=configs.get("model").get("platform"),
        embedding_model=configs.get("model").get("embedding_model"),
        chat_model=configs.get("model").get("chat_model")
    )


    ########################################################################
    # Build RAG
    ########################################################################

    # -- Define documents
    documents = [
        {
            "content": "Milvus is a vector database built for embedding similarity search and AI applications.",
            "metadata": {"source": "documentation", "topic": "introduction"},
        },
        {
            "content": "Full-text search in Milvus allows you to search using keywords and phrases.",
            "metadata": {"source": "tutorial", "topic": "full-text search"},
        },
        {
            "content": "Hybrid search combines the power of sparse BM25 retrieval with dense vector search.",
            "metadata": {"source": "blog", "topic": "hybrid search"},
        },
    ]

    entities = []
    texts = [doc["content"] for doc in documents]
    embeddings = [model.get_text_embedding(text=text) for text in texts]

    # -- Create a collection
    create_collection(
        uri=uri,
        collection_name=collection_name,
        embedding_dim=len(embeddings[0]),
        consistency_level="Bounded",
        overwrite=True,
        collection_type="full_text_search",
        dense_search_metric_type="IP",
        sparse_search_metric_type="BM25"
    )

    for i, doc in enumerate(documents):
        entities.append(
            {
                "content": doc["content"],
                "dense_vector": embeddings[i],
                "metadata": doc.get("metadata", {}),
            }
        )

    insert(uri=uri, collection_name=collection_name, data=entities)
    time.sleep(3)


    ########################################################################
    # Perform Retrieval
    ########################################################################

    # -- Full-Text Search
    # Example query for keyword search
    query = "full-text search keywords"

    # BM25 sparse vectors
    sparse_results = []
    trial = 0
    while True:
        trial = trial + 1
        sparse_results = search(
            uri=uri,
            collection_name=collection_name,
            queries=[query],
            query_embeddings=[],
            limit=5,
            search_type="full_text_search"
        )
        if len(sparse_results) > 0 or trial > MAX_TRIALS:
            break
        else:
            time.sleep(1)

    sparse_results = sparse_results[0]

    # Print results
    print("\nSparse Search (Full-text search):")
    for i, result in enumerate(sparse_results):
        print(f"{i + 1}. Score: {result.get('distance'):.4f}, Content: {result.get('entity').get('content')}")
    print()

    # -- Semantic Search
    query = "How does Milvus help with similarity search?"
    query_embedding = model.get_text_embedding(text=query)

    dense_results = []
    trial = 0
    while True:
        trial = trial + 1
        dense_results = search(
            uri=uri,
            collection_name=collection_name,
            queries=[query],
            query_embeddings=[query_embedding],
            limit=5,
            search_type="semantic_search",
            dense_search_metric_type="IP",
            output_fields=["content", "metadata"],
            anns_field="dense_vector",
        )
        if len(sparse_results) > 0 or trial > MAX_TRIALS:
            break

    dense_results = dense_results[0]

    print("\nDense Search (Semantic):")
    for i, result in enumerate(dense_results):
        print(f"{i + 1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}")
    print()

    # -- Hybrid Search
    query = "what is hybrid search"
    query_embedding = model.get_text_embedding(text=query)

    hybrid_results = []
    trial = 0
    while True:
        trial = trial + 1
        hybrid_results = search(
            uri=uri,
            collection_name=collection_name,
            queries=[query],
            query_embeddings=[query_embedding],
            limit=5,
            search_type="combined_search",
            dense_search_metric_type="IP",
            sparse_search_metric_type="BM25",
            output_fields=["content", "metadata"]
        )
        if len(hybrid_results) > 0 or trial > MAX_TRIALS:
            break

    hybrid_results = hybrid_results[0]

    print("\nHybrid Search (Combined):")
    for i, result in enumerate(hybrid_results):
        print(f"{i + 1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}")
    print()

    ########################################################################
    # Answer Generation
    ########################################################################

    query = "what is hybrid search"
    context = "\n\n".join([doc["entity"]["content"] for doc in hybrid_results])

    SYSTEM_PROMPT= "You are a helpful assistant that answers questions based on the provided context."
    USER_PROMPT = f"""
        Answer the following question based on the provided context. If the context doesn't contain relevant 
        information, just say "I don't have enough information to answer this question."
    
        Context:
        {context}
    
        Question: {query}
    
        Answer:
    """

    response = model.process_query(system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT)
    print("\nAnswer Generation:")
    print(f"* Question: {query}")
    print(f"* Response: {response}")
    return


if __name__ == "__main__":
    main()
