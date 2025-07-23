# Example Human Gen

소현님이 만드신 5개의 테이블에 대한 table description excel 파일을 Milvus DB에 넣는 예제입니다. 이 예제에서는 .env파일에 OPENAI_API_KEY가 필요합니다.

[data](./data/)에 각 테이블에 대한 설명이 담긴 엑셀 파일이 존재합니다.

5개 테이블에 대한 정보는 아래와 같습니다.
> - daily_finished_goods_agg
> - erp_dm_subc_edited
> - erp_pm_ncr_mod
> - item_master
> - monthly_production_metrics


## 데모 실행 방법

[shilla_example.ipynb](./shilla_example.ipynb) 를 실행하여 데모 실행해 봅니다. 노트북 내부에 자세히 설명 기술해 두었습니다.