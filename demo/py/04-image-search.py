import os
from pathlib import Path
import time

from PIL import Image

from milvus.client.utils import create_collection, insert, search
from milvus.utils import get_configurations, run_command
from milvus.image import FeatureExtractor


ROOT = Path(__file__).resolve().parent
DATA_DIR = os.path.join("..", "..", "data")
MAX_TRIALS = 10


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_image_search.yaml")
    configs = get_configurations(config_yaml_path=config_path)

    # -- Milvus configurations
    uri = f"{configs.get('milvus').get('host')}:{configs.get('milvus').get('port')}"
    if not uri.startswith("http://"):
        uri = f"http://{uri}"
    collection_name = configs.get("milvus").get("collection_name")


    ########################################################################
    # Download Dataset
    ########################################################################

    FILENAME = "reverse_image_search.zip"
    DATA_DEST = os.path.join(DATA_DIR, FILENAME.split(".zip")[0])
    if not os.path.exists(path=DATA_DEST):
        os.makedirs(DATA_DEST, exist_ok=True)
        if not os.path.exists(path=ROOT / FILENAME):
            cmd = f"wget https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/{FILENAME}"
            run_command(cmd=cmd, cwd=ROOT)

        cmd = f"unzip -q -o {FILENAME} -d {DATA_DEST}"
        run_command(cmd=cmd, cwd=ROOT)

        cmd = f"rm -r {FILENAME}"
        run_command(cmd=cmd, cwd=ROOT)

    ########################################################################
    # Create Dummy Embeddings to Get Dimension
    ########################################################################

    extractor = FeatureExtractor(model_name="resnet34")
    embedding_dim = extractor.get_embedding_dim()


    ########################################################################
    # Create a Milvus Collection
    ########################################################################

    create_collection(
        uri=uri,
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        overwrite=True,
        collection_type="image_search",
        dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
        vector_field_name="vector",
        auto_id=True,
        enable_dynamic_field=True
    )


    ########################################################################
    # Insert the Embeddings
    ########################################################################

    data = []
    for dirpath, foldername, filenames in os.walk(os.path.join(DATA_DEST, "train")):
        for filename in filenames:
            if filename.endswith(".JPEG"):
                filepath = dirpath + "/" + filename
                image_embedding = extractor(filepath)
                data.append(
                    {
                        "vector": image_embedding,
                        "filename": filepath
                    }
                )
    insert(
        uri=uri,
        collection_name=collection_name,
        data=data
    )
    time.sleep(3)

    ########################################################################
    # Search from Query Image
    ########################################################################

    query_image = os.path.join(DATA_DEST, "test", "basketball", "n02802426_12693.JPEG")
    image_embedding = extractor(query_image)

    trial = 0
    while True:
        trial = trial + 1
        results = search(
            uri=uri,
            collection_name=collection_name,
            queries=[],
            query_embeddings=[image_embedding],
            limit=10,
            search_type="image_search",
            output_fields=["filename"],
            dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
        )
        if len(results[0]) > 0 or trial > MAX_TRIALS:
            break
        else:
            time.sleep(1)

    images = []
    for result in results:
        for hit in result[:10]:
            filename = hit["entity"]["filename"]
            print(f"Filename: {filename}")
            img = Image.open(filename)
            img = img.resize((150, 150))
            images.append(img)

    width = 150 * 5
    height = 150 * 2
    concatenated_image = Image.new(mode="RGB", size=(width, height))

    for idx, img in enumerate(images):
        x = idx % 5
        y = idx // 5
        concatenated_image.paste(img, (x * 150, y * 150))

    Image.open(query_image).resize((150, 150)).show()
    concatenated_image.show()
    return


if __name__ == "__main__":
    main()
