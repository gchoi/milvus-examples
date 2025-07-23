import logging

from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import utility

from milvus_indexer.milvus_indexer import MilvusIndexer

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
MILVUS_URI = "http://localhost:19530"
OLLAMA_BASE_URL = "http://10.128.0.21:11435"
EMBEDDING_MODEL = "bge-m3:latest"
QUERY = ""

# 예제용 데이터베이스 및 컬렉션 이름
DB_NAME = "milvus_indexer_example_db"
COLLECTION_NAME = "example_docs_collection_v1"

def drop_db(db_name: str):
    from pymilvus import connections
    from pymilvus import db as milvus_db

    try:
        connections.connect(uri=MILVUS_URI, timeout=3.0)
        for db_name in milvus_db.list_database():
            if db_name != DB_NAME:
                continue
            milvus_db.using_database(db_name)
            for collection_name in utility.list_collections():
                utility.drop_collection(collection_name)
            if db_name != "default":
                milvus_db.drop_database(db_name)
    except Exception as e:
        logger.error(f"Error dropping database '{db_name}': {e}")

def print_search_results(results: list[Document], prefix: str, query: str):
    logger.info(f"{prefix} '{query}' 검색 결과:")
    if not results:
        logger.info("  문서를 찾을 수 없습니다.")
        return
    for i, res in enumerate(results):
        logger.info(f"  결과 {i+1}: 내용='{res.page_content}', 메타데이터={res.metadata}")

def main():
    logger.info("0-1. 새로운 시작을 위해 기존의 데이터베이스와 컬렉션 삭제")
    drop_db(DB_NAME)

    # --- 임베딩 함수 초기화 ---
    logger.info(f"0-2. Ollama ('{OLLAMA_BASE_URL}')에서 임베딩 모델 '{EMBEDDING_MODEL}'을 초기화")
    try:
        embedding = init_embeddings(model=EMBEDDING_MODEL, provider="ollama", base_url=OLLAMA_BASE_URL)
        _ = embedding.embed_query("임베딩 테스트")
    except Exception as e:
        logger.error(f"임베딩 함수 초기화 실패: {e}")
        logger.error(f"Ollama가 실행 중이고 모델 '{EMBEDDING_MODEL}'이 다운로드되었는지 확인하십시오 "
                f"(예: 'ollama pull {EMBEDDING_MODEL}'), 그리고 기본 URL '{OLLAMA_BASE_URL}'이 올바른지 확인하십시오.")
        return

    indexer = None
    try:
        # 1. MilvusIndexer 생성
        logger.info(f"1. MilvusIndexer 생성 (컬렉션: '{COLLECTION_NAME}', DB: '{DB_NAME}')")
        indexer = MilvusIndexer(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            milvus_uri=MILVUS_URI,
            milvus_db_name=DB_NAME,
            drop_old_collection=True,
            enable_dynamic_field=False
        )

        # 문서들 생성
        files = [
            ("members_ds.txt", "민호님은 DS팀의 팀원입니다.\n소현님도 DS팀의 팀원입니다."),
            ("members_vision.txt", "진수님은 비전팀의 팀원입니다.\n지원님도 비전팀의 팀원입니다.")
        ]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        docs = []
        for file_name, content in files:
            docs.extend(text_splitter.split_documents([Document(page_content=content, metadata={"source": file_name})]))

        # 2. 문서 추가 및 검색
        logger.info(f"2. 초기 문서 {len(docs)}개 추가")
        indexer.index_documents(docs)

        stats = indexer.get_collection_stats()
        logger.info(f"2-1. 초기 문서 추가 후 컬렉션 상태: {stats}")

        results_add1 = indexer.similarity_search(QUERY, k=5)
        print_search_results(results_add1, "2-2.", QUERY)

        # 3. 일부 문서 변경 후, 상태 확인 및 문서 검색해서 확인
        logger.info("3. 일부 문서 변경")
        files_to_update = [
            ("members_vision.txt", "대경님은 비전팀의 팀원입니다.\n현진님도 비전팀의 팀원입니다.")
        ]
        docs = []
        for file_name, content in files_to_update:
            docs.extend(text_splitter.split_documents([Document(page_content=content, metadata={"source": file_name})]))
        indexer.index_documents(docs)

        stats = indexer.get_collection_stats()
        logger.info(f"3-1. 일부 문서 변경 후 컬렉션 상태: {stats}")

        results_update1 = indexer.similarity_search(QUERY, k=5)
        print_search_results(results_update1, "3-2.", QUERY)

        # 4. 일부 문서 삭제 후, 상태 확인 및 문서 검색해서 확인
        logger.info("4. 일부 문서 삭제")
        sources_to_delete = ["members_vision.txt"]
        indexer.delete_documents_by_sources(sources_to_delete)

        stats = indexer.get_collection_stats()
        logger.info(f"4-1. 일부 문서 삭제 후 컬렉션 상태: {stats}")

        results_delete1 = indexer.similarity_search(QUERY, k=5)
        print_search_results(results_delete1, "4-2.", QUERY)

        # 5. 전체 문서 삭제 후, 상태 확인 및 문서 검색해서 확인
        logger.info("5. 컬렉션의 모든 인덱싱된 문서 삭제")
        indexer.clear_all_indexed_data()

        stats = indexer.get_collection_stats()
        logger.info(f"5-1. 모든 데이터 삭제 후 컬렉션 상태: {stats}")

        results_clear = indexer.similarity_search(QUERY, k=5)
        print_search_results(results_clear, "5-2.", QUERY)

        logger.info("6. 예제 성공적으로 완료")

    except Exception as e:
        logger.error(f"MilvusIndexer 예제에서 오류 발생: {e}", exc_info=True)
    finally:
        if indexer:
            logger.info("7. MilvusIndexer 연결 해제")
            indexer.disconnect()
        logger.info("8. MilvusIndexer 예제 스크립트 종료")

if __name__ == "__main__":
    main()
