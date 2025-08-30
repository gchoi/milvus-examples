from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")

if client.has_collection(collection_name="demo_collection"):
    print("Collection already exists.")
    client.drop_collection(collection_name="demo_collection")

client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo have 768 dimensions
)
