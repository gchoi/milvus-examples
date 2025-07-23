from pathlib import Path
import pandas as pd
from langchain.embeddings import init_embeddings
from langchain_milvus import Milvus
from vmilvus.vectordb import RetrievalEvaluator
from vmilvus.utils import preprocess_documents

BASE_DIR = Path(__file__).parent
# URI = "http://10.128.0.20:19530"
URI = str(BASE_DIR / "milvus_test.db")
COLLECTION_NAME = "example_collection"
EMBEDDINGS = init_embeddings(model="bge-m3:latest", provider="ollama", base_url="http://10.128.0.21:11435")
DESC_PATH = BASE_DIR / "text2sql_test_description.xlsx"
EXAMPLE_PATH = BASE_DIR / "example2.xlsx"

df = pd.read_excel(EXAMPLE_PATH)

# RetrievalEvaluator 생성
evaluator = RetrievalEvaluator(df)

# Vectorstore 생성()
documents = preprocess_documents(DESC_PATH)
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=EMBEDDINGS,
    connection_args={"uri": URI},
    collection_name=COLLECTION_NAME,
    drop_old=True
)

# 평가
evaluator.evaluate(vectorstore, k=5)

