import os
from typing import Any, Union, Optional
import base64

from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import ollama
from ollama import chat
import numpy as np
from PIL import Image

from ..conf import Logger


# -- logger settings
logger = Logger(env="dev")

load_dotenv()


class Model:
    def __init__(self, platform: str, embedding_model: str, chat_model: str):
        self.platform = platform
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        return

    def set_model(self) -> None:
        match self.platform.lower():
            case "openai":
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            case "google":
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                pass
            case "ollama":
                pass
            case _:
                raise ValueError(f"Unsupported platform: {self.platform}")
        return

    def get_text_embedding(self, text: Union[str, Any]):
        match self.platform.lower():
            case "openai":
                result = OpenAI().embeddings.create(input=text, model=self.embedding_model).data[0]
                return result.embedding
            case "google":
                result = genai.embed_content(model=self.embedding_model, content=text)
                return result["embedding"]
            case "ollama":
                response = ollama.embed(model=self.embedding_model, input=text)
                result = response["embeddings"][0]
                return result
            case _:
                raise ValueError(f"Unsupported platform: {self.platform}")

    def get_embedding_dim(self) -> int:
        test_embedding = self.get_text_embedding(text="test")

        dim = np.array(test_embedding).shape
        if len(dim) == 1:
            embedding_dim = len(test_embedding)
        elif len(dim) == 2:
            embedding_dim = len(test_embedding[0])
        else:
            embedding_dim = 0
        logger.info(f"Embedding dimension: {embedding_dim}")
        return embedding_dim

    def process_query(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        image_pil: Optional[Image] = None,
        max_tokens: int = 300
    ) -> str:
        match self.platform.lower():
            case "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                user_content = []
                if user_prompt:
                    user_content.append({
                        "type": "text",
                        "text": user_prompt
                    })
                if image_pil:
                    base64_image = base64.b64encode(image_pil).decode("utf-8")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })
                if len(user_content) > 0:
                    messages.append(user_content)

                response = OpenAI().chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            case "google":
                parts = []
                if system_prompt is not None and user_prompt is not None:
                    parts = [system_prompt + user_prompt]
                if system_prompt is not None and user_prompt is None:
                    parts = [system_prompt]
                if system_prompt is None and user_prompt is not None:
                    parts = [user_prompt]

                response = (
                    genai.GenerativeModel(self.chat_model).generate_content(
                        contents=[
                        {
                            "role": "user",
                            "parts": parts
                        }
                    ]
                ))
                return response.text

            case "ollama":
                response = chat(
                    model=self.chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                    ],
                )
                return response['message']['content']
            case _:
                raise ValueError(f"Unsupported platform: {self.platform}")
