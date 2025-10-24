import os
import time
from glob import glob
from pathlib import Path

import torch
from transformers import AutoModel
from tqdm import tqdm

from milvus.utils import get_configurations, run_command
from milvus.client.utils import create_collection, insert, search


ROOT = Path(__file__).resolve().parent
DATA_DIR_ROOT = os.path.join("..", "..", "data")
MODEL_DIR = os.path.join("..", "..", "models")
MAX_TRIALS = 10


class Encoder:
    def __init__(self, model_name: str):
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True
        )  # You must set trust_remote_code=True
        self.model.set_processor(model_name)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(images=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(images=[image_path])
        return query_emb[0].tolist()


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
    DATA_DEST = os.path.join(DATA_DIR_ROOT, FILENAME.split(".tar.gz")[0])

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

    ########################################################################
    # Load Embedding Model
    ########################################################################

    model_name = "BAAI/BGE-VL-base"
    encoder = Encoder(model_name=model_name)

    ########################################################################
    # Generate embeddings
    ########################################################################

    # Generate embeddings for the image dataset
    data_dir = os.path.join(DATA_DIR_ROOT, "amazon_reviews_2023_subset", "images")
    image_list = glob(os.path.join(data_dir, "images","*.jpg"))  # We will only use images ending with ".jpg"
    image_dict = {}
    for image_path in tqdm(image_list, desc="Generating image embeddings: "):
        try:
            image_dict[image_path] = encoder.encode_image(image_path)
        except Exception as e:
            print(f"Failed to generate embedding for {image_path}. Skipped.")
            continue
    print("Number of encoded images:", len(image_dict))


    ########################################################################
    # Create a Milvus Collection
    ########################################################################

    embedding_dim = len(image_dict.get([*image_dict.keys()][0]))

    # Connect to Milvus client given URI
    create_collection(
        uri=uri,
        collection_name=collection_name,
        collection_type="multimodal_search",
        auto_id=True,
        embedding_dim=embedding_dim,
        enable_dynamic_field=True,
        overwrite=True
    )


    ########################################################################
    # Insert into Milvus
    ########################################################################

    data = [{"image_path": k, "vector": v} for k, v in image_dict.items()]
    insert(
        uri=uri,
        collection_name=collection_name,
        data=data,
    )
    time.sleep(3)


    ########################################################################
    # Run search
    ########################################################################

    query_image = os.path.join(data_dir, "leopard.jpg")  # Change to your own query image path
    query_text = "phone case with this image theme"

    # Generate the query embedding given image and text instructions
    query_vec = encoder.encode_query(image_path=query_image, text=query_text)

    trial = 0
    while True:
        trial = trial + 1
        search_results = search(
            uri=uri,
            collection_name=collection_name,
            queries=[],
            query_embeddings=[query_vec],
            limit=configs.get("milvus").get("search").get("limit"),
            search_type="multimodal_search",
            dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
        )[0]

        if len(search_results) > 0 or trial > MAX_TRIALS:
            break
        else:
            time.sleep(1)

    retrieved_images = [hit.get("entity").get("image_path") for hit in search_results]
    print(retrieved_images)
    return

if __name__ == "__main__":
    main()
