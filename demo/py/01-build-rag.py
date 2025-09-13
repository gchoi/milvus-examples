import os
from glob import glob
import json

from tqdm import tqdm

from milvus.client.utils import (
    drop_collection,
    create_collection,
    insert,
    search
)
from milvus.utils import get_configurations
from milvus.model import Model


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
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

    # -- Drop collection if exists
    drop_collection(
        uri=uri,
        collection_name=collection_name
    )

    # -- Get embeddings
    text_lines = []
    for file_path in glob(pathname="../../milvus_docs/en/faq/*.md", recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()
        text_lines += file_text.split("# ")

    embedding_dim = model.get_embedding_dim()

    # -- Create a collection
    create_collection(
        uri=uri,
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        metric_type=configs.get("milvus").get("search").get("metric_type"),
        consistency_level="Bounded",
        overwrite=True,
    )

    # -- Insert data
    data = []
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append(
            {
                "id": i,
                "vector": model.get_text_embedding(text=line),
                "text": line
            }
        )
    insert(uri=uri, collection_name=collection_name, data=data)

    # -- Retrieve data for a query
    question = "How is data stored in milvus?"

    search_res = search(
        uri=uri,
        collection_name=collection_name,
        query_embeddings=[model.get_text_embedding(text=question)],
        limit=configs.get("milvus").get("search").get("limit"),
        metric_type=configs.get("milvus").get("search").get("metric_type"),
    )

    retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
    print("Search results:")
    print(json.dumps(retrieved_lines_with_distances, indent=4))

    ########################################################################
    # Use LLM to get a RAG response
    ########################################################################

    context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
    SYSTEM_PROMPT = """
        Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage 
        snippets provided.
    """
    USER_PROMPT = f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question 
        enclosed in <question> tags.
        <context>
            {context}
        </context>
        <question>
            {question}
        </question>
    """

    response = model.process_query(system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT)
    print(f"Question: {question}")
    print(f"Response: {response}")
    return


if __name__ == "__main__":
    main()
