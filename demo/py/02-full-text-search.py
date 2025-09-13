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

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)


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
    drop_collection(uri=uri, collection_name=collection_name)

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
        metric_type="IP",
        consistency_level="Bounded",
        overwrite=True,
    )

    # -- Text Analysis Configuration
    analyzer_params = {
        "tokenizer": "standard",
        "filter": ["lowercase"]
    }


    return


if __name__ == "__main__":
    main()
