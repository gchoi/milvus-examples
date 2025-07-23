from langchain_core.vectorstores import VectorStore
from vmilvus.utils import extract_column_names
from typing import List, Tuple
import pandas as pd


class RetrievalEvaluator:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataset = self._get_dataset(dataframe)

    def _get_dataset(self, dataframe: pd.DataFrame) -> List[Tuple[str, str, List[str]]]:
        """dataset을 생성합니다.

        dataset은 question, sql, column_names의 튜플 리스트입니다.

        Args:
            dataframe (pd.DataFrame): question과 sql이 포함된 데이터프레임

        Returns:
            List[Tuple[str, str, List[str]]]: question, sql, column_names의 튜플 리스트
        """
        qid = dataframe['question_id'].tolist()
        questions = dataframe['question'].tolist()
        sqls = dataframe['sql'].tolist()
        column_names = [extract_column_names(sql) for sql in sqls]

        return list(zip(qid, questions, sqls, column_names, strict=True))
    
    def evaluate(self, vectorstore: VectorStore, k: int = 5, expr: str = ""):
        """vectorstore에 저장된 document들과 question을 이용해 retrieval 성능을 평가합니다.

        Args:
            k (int, optional): top k documents를 이용해 평가합니다. Defaults to 5.
            expr (str, optional): milvus expr. Defaults to "".
        """
        for qid, question, sql, column_names in self.dataset:
            if len(column_names) == 0:
                continue
            docs = vectorstore.similarity_search(question, k=k, expr=expr)
            retrieved_column_names = [doc.metadata["column_name"] for doc in docs]

            missing_columns = set(column_names) - set(retrieved_column_names)
            recall = len(set(column_names) & set(retrieved_column_names)) / len(column_names)

            missing_str = "" if len(missing_columns) == 0 else str(missing_columns)
            print(f"[{qid}]. Recall: {recall:.2f}. Missing: {missing_str}")
            
