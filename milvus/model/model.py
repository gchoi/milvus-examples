import os
from typing import Any, Union

from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import ollama
from ollama import chat, ChatResponse
import numpy as np

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

    def set_model(self):
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

    def get_embedding_dim(self):
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

    def process_query(self, system_prompt: str, user_prompt: str):
        match self.platform.lower():
            case "openai":
                response = OpenAI().chat.completions.create(
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
                return response.choices[0].message.content
            case "google":
                response = (
                    genai.GenerativeModel(self.chat_model).generate_content(
                        contents=[
                        {
                            "role": "user",
                            "parts": [system_prompt + user_prompt]
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
