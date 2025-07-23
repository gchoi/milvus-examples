from typing import List
import pandas as pd
import sqlglot
from langchain_core.documents import Document

def extract_table_names(sql: str) -> List[str]:
    """
    SQL 쿼리에서 사용된 테이블 이름들을 추출합니다.

    Args:
        sql (str): SQL 쿼리 문자열

    Returns:
        List[str]: 테이블 이름 리스트 (중복 없이)
    """
    try:
        parsed = sqlglot.parse_one(sql)
        # 1. WITH/CTE 이름 목록 추출
        cte_names = set()
        for cte in parsed.find_all(sqlglot.exp.CTE):
            if cte.alias_or_name:
                cte_names.add(cte.alias_or_name)
        # 2. 전체 테이블 목록에서 CTE 이름 제외
        tables = set()
        for table in parsed.find_all(sqlglot.exp.Table):
            name = f"{table.db}.{table.name}" if table.db else table.name
            if name not in cte_names:
                tables.add(name)
        return list(tables)
    except Exception:
        return []


def extract_column_names(sql: str) -> List[str]:
    """
    SQL 쿼리에서 사용된 컬럼 이름들을 추출합니다. (테이블 alias나 스키마는 제외)

    Args:
        sql (str): SQL 쿼리 문자열

    Returns:
        List[str]: 컬럼 이름 리스트 (중복 없이)
    """
    try:
        parsed = sqlglot.parse_one(sql)
        column_names = set()
        for column_exp in parsed.find_all(sqlglot.exp.Column):
            column_names.add(column_exp.name) # 'name' 속성은 순수 컬럼명만 반환
        return list(column_names)
    except Exception:
        # 파싱 실패 시 빈 리스트 반환
        return []


def preprocess_documents(desc_path: str) -> List[Document]:
    """
    description 엑셀 파일의 내용을 Document로 변환합니다.
    각 시트는 테이블에 해당하며, 각 행은 해당 테이블의 컬럼 설명을 나타냅니다.

    Args:
        desc_path (str): description 엑셀 파일 경로

    Returns:
        List[Document]: 생성된 Document 객체 리스트
    """
    # 각 시트에서 필수로 존재해야 하는 컬럼 이름들
    REQUIRED_COLUMNS = ['original_column_name', 'column_description', 'data_format']

    try:
        all_sheets_data = pd.read_excel(desc_path, sheet_name=None)
    except FileNotFoundError:
        print(f"오류: 설명 파일을 찾을 수 없습니다 - 경로: {desc_path}")
        return []
    except Exception as e:
        print(f"오류: Excel 파일 읽기 중 문제 발생 ({desc_path}): {e}")
        return []

    documents = []
    for sheet_name, sheet_df in all_sheets_data.items():
        # 1. 현재 시트에 필수 컬럼이 모두 있는지 확인
        if not all(col in sheet_df.columns for col in REQUIRED_COLUMNS):
            print(
                f"경고: 시트 '{sheet_name}' (파일: '{desc_path}')에 필수 컬럼 {REQUIRED_COLUMNS} 중 일부가 없습니다. "
                f"존재 컬럼: {list(sheet_df.columns)}"
            )
            continue  # 필수 컬럼이 없으면 이 시트는 건너뜀

        # 2. 각 행을 dict로 순회
        for row in sheet_df.to_dict(orient='records'):
            original_column_name = str(row.get('original_column_name', '') or '')
            column_description = str(row.get('column_description', '') or '')
            data_format = str(row.get('data_format', '') or '')

            # 컬럼 설명이 비어있으면 문서로 만들지 않음
            if not column_description.strip():
                print(f"정보: 시트 '{sheet_name}'의 컬럼 '{original_column_name}'은 설명이 없어 건너뜀.")
                continue

            documents.append(Document(
                page_content=column_description,
                metadata={
                    "column_name": original_column_name,
                    "data_format": data_format,
                    "table_name": sheet_name
                }
            ))
    return documents