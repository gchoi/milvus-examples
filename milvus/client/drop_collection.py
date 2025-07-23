import urllib

from pymilvus import utility, connections

def drop_collection(
    collection_name: str, 
    uri: str
):
    uri_ = urllib.parse.urlsplit(uri)
    connections.connect("default", host=uri_.hostname, port=uri_.port)
    print(utility.drop_collection(collection_name))


if __name__=="__main__":
    drop_collection("shilla_test_20250526_openai_embedding_3_large")
    drop_collection("shilla_test_20250526_ollama_bge_m3")