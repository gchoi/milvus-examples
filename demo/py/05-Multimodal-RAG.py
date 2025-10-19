import os
from pathlib import Path
import torch
from visual_bge.modeling import Visualized_BGE

from milvus.utils import get_configurations, run_command


ROOT = Path(__file__).resolve().parent
DATA_DIR = os.path.join("..", "..", "data")
MODEL_DIR = os.path.join("..", "..", "models")
MAX_TRIALS = 10


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_multimodal.yaml")
    configs = get_configurations(config_yaml_path=config_path)

    # -- Milvus configurations
    uri = f"{configs.get('milvus').get('host')}:{configs.get('milvus').get('port')}"
    if not uri.startswith("http://"):
        uri = f"http://{uri}"
    collection_name = configs.get("milvus").get("collection_name")

    ########################################################################
    # Download Dataset
    ########################################################################

    FILENAME = "amazon_reviews_2023_subset.tar.gz"
    DATA_DEST = os.path.join(DATA_DIR, FILENAME.split(".tar.gz")[0])

    if not os.path.exists(path=DATA_DEST):
        os.makedirs(DATA_DEST, exist_ok=True)
        if not os.path.exists(path=ROOT / FILENAME):
            cmd = f"wget https://github.com/milvus-io/bootcamp/releases/download/data/{FILENAME}"
            run_command(cmd=cmd, cwd=ROOT)

        cmd = f"tar -xzf {FILENAME}"
        run_command(cmd=cmd, cwd=ROOT)

        cmd = f"mv images_folder {DATA_DEST}/images"
        run_command(cmd=cmd, cwd=ROOT)

        cmd = f"rm -r {FILENAME}"
        run_command(cmd=cmd, cwd=ROOT)


    ########################################################################
    # Download Model
    ########################################################################

    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_NAME = "Visualized_base_en_v1.5.pth"
    MODEL_DEST = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(path=MODEL_DEST):
        cmd = f"wget https://huggingface.co/BAAI/bge-visualized/resolve/main/{MODEL_NAME}"
        run_command(cmd=cmd, cwd=ROOT)

        cmd = f"mv ./{MODEL_NAME} {MODEL_DIR}"
        run_command(cmd=cmd, cwd=ROOT)

    return

if __name__ == "__main__":
    main()
