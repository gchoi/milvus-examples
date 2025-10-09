import os
import pathlib

import pandas as pd

from milvus.client.utils import create_collection, insert, search
from milvus.model import Model
from milvus.utils import get_configurations
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)

from milvus.utils import run_command
from milvus.image import FeatureExtractor


ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = os.path.join("..", "..", "data")


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_hybrid_search.yaml")
    configs = get_configurations(config_yaml_path=config_path)

    # -- Milvus configurations
    uri = f"{configs.get('milvus').get('host')}:{configs.get('milvus').get('port')}"
    if not uri.startswith("http://"):
        uri = f"http://{uri}"
    collection_name = configs.get("milvus").get("collection_name")

    # -- Model configurations
    model = Model(
        platform=configs.get("model").get("platform"),
        embedding_model=configs.get("model").get("embedding_model"),
        chat_model=configs.get("model").get("chat_model"),
    )

    ########################################################################
    # Download Dataset
    ########################################################################

    DATA_DEST = os.path.join(DATA_DIR, "reverse_image_search")
    os.makedirs(DATA_DEST, exist_ok=True)
    if not os.path.exists(path=ROOT / "reverse_image_search.zip"):
        cmd = "wget https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/reverse_image_search.zip"
        run_command(cmd=cmd, cwd=ROOT)

    cmd = f"unzip -q -o reverse_image_search.zip -d {DATA_DEST}"
    run_command(cmd=cmd, cwd=ROOT)

    cmd = f"rm -r reverse_image_search.zip"
    run_command(cmd=cmd, cwd=ROOT)



    return


if __name__ == "__main__":
    main()
