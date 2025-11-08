import os
import time
from glob import glob
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import cv2

from milvus.utils import get_configurations, run_command
from milvus.client.utils import create_collection, insert, search
from milvus.image import create_panoramic_view, ImageEncoder
from milvus.model import Model


ROOT = Path(__file__).resolve().parent
DATA_DIR_ROOT = os.path.join("..", "..", "data")
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
    # Load Image Encoder Model
    ########################################################################

    encoder = ImageEncoder(model_name=configs.get("model").get("image_encoder"))


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


    ########################################################################
    # Rerank with GPT-4o
    ########################################################################

    # -- Create a panoramic view
    panoramic_image = create_panoramic_view(query_image, retrieved_images)
    panoramic_image = Image.fromarray(cv2.cvtColor(panoramic_image, cv2.COLOR_BGR2RGB)).resize((300, 300))
    panoramic_image.show()

    # -- Rerank and explain
    # Model configurations
    model = Model(
        platform=configs.get("model").get("platform"),
        embedding_model=configs.get("model").get("embedding_model"),
        chat_model=configs.get("model").get("chat_model"),
    )

    USER_PROMPT = f"""
        You are responsible for ranking results for a Composed Image Retrieval.
        The user retrieves an image with an 'instruction' indicating their retrieval intent.
        For example, if the user queries a red car with the instruction 'change this car to blue,' a similar type of car in blue would be ranked higher in the results.
        Now you would receive instruction and query image with blue border. Every item has its red index number in its top left. Do not misunderstand it. 
        User instruction: {query_text} \n\n
        Provide a new ranked list of indices from most suitable to least suitable, followed by an explanation for the top 1 most suitable item only.
        "The format of the response has to be 'Ranked list: []' with the indices in brackets as integers, followed by 'Reasons:' plus the explanation why this most fit user's query intent.
    """

    response = model.process_query(user_prompt=USER_PROMPT, image_pil=panoramic_image, max_tokens=300)

    # Parse the ranked indices from the response
    start_idx = response.find("[")
    end_idx = response.find("]")
    ranked_indices_str = response[start_idx + 1 : end_idx].split(",")
    ranked_indices = [int(index.strip()) for index in ranked_indices_str]

    # extract explanation
    explanation = response[end_idx + 1 :].strip()
    print(explanation)

    best_index = ranked_indices[0]
    best_img = Image.open(retrieved_images[best_index])
    best_img = best_img.resize((150, 150))
    best_img.show()
    return

if __name__ == "__main__":
    main()
