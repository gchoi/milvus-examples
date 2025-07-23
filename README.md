# Milvus

본 문서에서는 MilvusDB에 대한 가이드를 담고 있습니다. 

> Milvus Repo 사용방법은 다음과 같습니다.
> - 데모를 실행해보고 싶으시면 이쪽 가이드를 참고 [demo guide](demo_guide.md)
>
> - Milvus DB에 올리는 여러 Exmperiments를 시도해 보고 싶으시면 이쪽 가이드를 참고 [experiment guide](experiment_guide.md)
>
> - Milvus DB 서버를 실행해보고 싶으시면 아래를 참고

## 1. Milvus 서버 실행

---

```bash
docker compose up -d
```

Milvus WebUI는 해당 주소에서 실행할 수 있습니다.

```bash
http://<HOST>:9091/webui # HOST 변경 필요
```
