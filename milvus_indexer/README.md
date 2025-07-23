# Milvus Indexer 모듈

## 1. 개요
본 문서는 Milvus 기반 벡터 인덱싱 및 검색을 위한 핵심 모듈인 MilvusIndexer(milvus_indexer.py)에 대해 설명합니다.
해당 모듈은 LangChain, Milvus, SQLRecordManager를 활용하여 문서 임베딩, 인덱싱, 유사도 검색, 데이터 관리(RAG 워크플로우)를 쉽게 구현할 수 있도록 설계되었습니다.

## 2. MilvusIndexer
### 2.1 주요 기능
Milvus 컬렉션 관리, 벡터 인덱싱, SQLRecordManager 기반 중복 관리, 유사도 검색, 데이터 삭제 및 통계 조회 등 RAG 파이프라인의 핵심 기능을 제공합니다.

### 2.2 지원 메서드, 컨텍스트 매니저, 예외

- **주요 메서드**
    - `index_documents(docs, ...)`: 문서 인덱싱 및 중복 관리
    - `similarity_search(query, k=5, ...)`: 쿼리 기반 유사도 검색
    - `delete_documents_by_sources(sources)`: 소스 기반 문서 삭제
    - `clear_all_indexed_data()`: 전체 인덱스 데이터 삭제
    - `get_collection_stats()`: 컬렉션 통계 조회
    - `drop_collection()`: 컬렉션 삭제
    - `disconnect()`: Milvus 연결 해제
- **컨텍스트 매니저 지원**
    - `with MilvusIndexer(...) as indexer:` 블록 종료 시 자동 disconnect
- **커스텀 예외**
    - `MilvusIndexerError`: 기본 예외
    - `CollectionNotFoundError`: 컬렉션 없음
    - `MilvusConnectionError`: Milvus 연결 오류
    - `QueryError`: 쿼리 오류
    - `IndexingError`: 인덱싱 오류
    - `DeletionError`: 삭제 오류

### 2.3 주요 파라미터

| 파라미터                | 설명                                               | 기본값                   |
|-----------------------|--------------------------------------------------|------------------------|
| `collection_name`     | Milvus 컬렉션 이름                                 | (필수)                  |
| `embedding_function`  | 임베딩 함수(LangChain Embeddings)                  | (필수)                  |
| `milvus_uri`          | Milvus 서버 주소                                   | `"http://localhost:19530"` |
| `milvus_db_name`      | Milvus DB 이름                                     | `"default"`             |
| `enable_dynamic_field`| 컬렉션 동적 필드 지원 여부                         | `False`                 |
| `index_params`        | 인덱스 상세 파라미터(예: IVF_FLAT, nlist 등)        | `{ "metric_type": "L2", ...}`|
| `consistency_level`   | Milvus 일관성 레벨                                 | `"Strong"`              |
| `record_manager_db_url`| SQLRecordManager DB 경로                           | `"sqlite:///record_manager_cache.sql"` |
| `drop_old_collection` | 기존 컬렉션 삭제 후 재생성 여부                     | `False`                 |
| `milvus_connection_alias`| Milvus 연결 별칭                                 | `"default_milvus_conn"` |
| `connection_timeout`  | Milvus 연결 타임아웃(초)                            | `10.0`                  |
| `source_id_key`       | 문서 메타데이터 중 소스 필드명                      | `"source"`              |

**참고**: `source_id_key`는 문서의 출처를 식별하고 `delete_documents_by_sources` 기능을 정확하게 사용하기 위해 매우 중요합니다. 이 키를 일관되게 설정하고 모든 문서의 메타데이터에 해당 키와 값을 포함해야 합니다. `InvalidDocumentError`는 주로 이 키가 누락되었을 때 발생할 수 있습니다.

### 2.4 내부 동작 및 확장성

- DB/컬렉션 자동 생성 및 연결, SQLRecordManager 기반 중복 관리
- 내부 메서드: `_setup_database`, `_connect_milvus`, `_init_vectorstore`, `_init_record_manager` 등
- 여러 Milvus 인스턴스/DB/컬렉션을 동시에 관리 가능
- with문 컨텍스트 매니저 지원(자원 자동 해제)

### 2.5 사용 팁 및 주의사항

- Milvus 및 임베딩 서버가 정상 동작 중인지 사전 확인
- 대량 문서 인덱싱 시 batch 처리 권장
- 메타데이터 스키마 및 타입 일관성 유지
- 예외 발생 시 로그/메시지 확인 및 커스텀 예외 처리 활용
- 컬렉션/DB/레코드매니저 네임스페이스 충돌 주의
- `milvus_indexer.py` 파일 내에는 테스트 및 개발 편의를 위한 `_drop_all_dbs()`와 같이 모든 데이터베이스와 컬렉션을 삭제할 수 있는 매우 위험한 유틸리티 함수가 포함되어 있을 수 있습니다. 프로덕션 환경이나 중요한 데이터가 있는 환경에서는 이 함수를 절대로 실행하지 않도록 각별히 주의하십시오.


### 3. examples/milvus_indexer_example.py
#### 3.1 예제 목적
- MilvusIndexer의 전체 워크플로우(초기화 → 인덱싱 → 검색 → 업데이트/삭제 → 통계/정리)를 단계별로 보여주는 스크립트입니다.
- 실제 Milvus 서버/임베딩 모델 환경에서 동작하도록 설계되었습니다.

#### 3.2 주요 흐름
1. 환경/로깅/설정값 정의
2. DB 및 컬렉션 초기화/정리
3. 임베딩 함수 준비 및 테스트
4. MilvusIndexer 인스턴스 생성
5. 문서 생성 및 인덱싱
6. 유사도 검색 및 결과 출력
7. 문서 업데이트/삭제 후 상태 확인
8. 전체 인덱스 삭제 및 종료 처리

#### 3.3 예제 실행
```bash
python examples/milvus_indexer_example.py
```
실행 시 로그를 통해 각 단계별 상태, 오류, 결과(검색 결과, 통계 등)를 확인할 수 있습니다.

### 4. 환경 변수 및 의존성
- Milvus 서버, Ollama 임베딩 서버 등 외부 환경 필요
- 의존 패키지: pymilvus, langchain, langchain_milvus, langchain_core, langchain_text_splitters 등
- 환경 변수 및 설정은 코드 내 상단에 정의
