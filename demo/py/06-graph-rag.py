import os
from collections import defaultdict

from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from pymilvus import MilvusClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from milvus.utils import get_configurations
from milvus.model import Model
from milvus.conf import Logger

from dataset import nano_dataset


# -- logger settings
logger = Logger(env="dev")

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_graph_rag.yaml")
    configs = get_configurations(config_yaml_path=config_path)

    # -- Milvus configurations
    uri = f"{configs.get('milvus').get('host')}:{configs.get('milvus').get('port')}"
    if not uri.startswith("http://"):
        uri = f"http://{uri}"
    collection_name = configs.get("milvus").get("collection_name")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    return


if __name__ == "__main__":
    main()
