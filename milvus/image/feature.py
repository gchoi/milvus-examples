import PIL
import timm
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoModel


class FeatureExtractor:
    def __init__(self, model_name: str):
        # Load the pre-trained model
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg.get("input_size")

        config = resolve_data_config({}, model=model_name)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def get_embeddings(self, input_image: PIL.Image.Image):
        # Preprocess the input image
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()
        embeddings = normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
        return embeddings

    def get_embedding_dim(self):
        img = Image.new("RGB", (224, 224), color="white")
        embeddings = self.get_embeddings(input_image=img)
        return len(embeddings)

    def __call__(self, image_path: str):
        input_image = Image.open(image_path).convert("RGB")  # Convert to RGB if needed
        embeddings = self.get_embeddings(input_image=input_image)
        return embeddings


class ImageEncoder:
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
