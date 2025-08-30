import os
from glob import glob
import json
from dotenv import load_dotenv

from openai import OpenAI
from tqdm import tqdm

from milvus.client.utils import (
    drop_collection,
    create_collection,
    insert,
search
)


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -- Milvus configurations
URI = "http://localhost:19530"
COLLECTION_NAME = "my_rag_collection"

# -- Model configurations
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"


def emb_text(client, text):
    return (
        client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        .data[0]
        .embedding
    )


def main():
    # -- Drop collection if exists
    drop_collection(uri=URI, collection_name=COLLECTION_NAME)

    # -- Get embeddings
    text_lines = []
    for file_path in glob(pathname="../../milvus_docs/en/faq/*.md", recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()
        text_lines += file_text.split("# ")

    openai_client = OpenAI()
    test_embedding = emb_text(client=openai_client, text="This is a test")
    embedding_dim = len(test_embedding)
    print(f"Embedding dimension: {embedding_dim}")

    # -- Create a collection
    create_collection(
        uri=URI,
        collection_name=COLLECTION_NAME,
        embedding_dim=embedding_dim,
        metric_type="IP",
        consistency_level="Bounded",
        overwrite=True,
    )

    # -- Insert data
    data = []
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append(
            {
                "id": i,
                "vector": emb_text(client=openai_client, text=line),
                "text": line
            }
        )
    insert(uri=URI, collection_name=COLLECTION_NAME, data=data)

    ########################################################################
    # Build RAG
    ########################################################################

    # -- Retrieve data for a query
    question = "How is data stored in milvus?"

    search_res = search(
        uri=URI,
        collection_name=COLLECTION_NAME,
        query_embeddings=[emb_text(client=openai_client, text=question)],
        limit=3,
        metric_type="IP"
    )

    retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
    print(json.dumps(retrieved_lines_with_distances, indent=4))

    # -- Use LLM to get a RAG response
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

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print(response.choices[0].message.content)
    return


if __name__ == "__main__":
    main()
