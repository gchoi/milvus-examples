# Example DB Desc Gen

이 예제에서는 상현님이 만드신 db_desc_gen(가칭)에서 생성한 table description를 Milvus DB에 넣는 과정을 실습합니다. 이 예제에서는 .env파일에 OPENAI_API_KEY가 필요합니다.

[data](./data/)에 db_desc_gen으로 생성한 json 파일이 존재합니다.

json 파일 안에는 2개의 테이블에 대한 설명이 생성되어 있습니다.

> - test_table
> - monthly_production_metrics


## 데모 실행 방법

본 데모에서는 monthly_production_metrics 테이블에 대해 실행해봅니다.

[example.ipynb](./example.ipynb) 를 실행하여 데모 실행해 봅니다. 간단한 데모입니다. 문서를 생성한다 -> milvus에 넣는다.

해당 기능에 대한 함수 정의도 참고 부탁드립니다.