# Milvus Demo Guide

본 문서에서는 Milvus를 활용한 데모 실행방법이 정리되어 있습니다. demo는 모두 `demo/` 폴더 하위에 있습니다.

### 1. 준비사항

---

[just](https://github.com/casey/just?tab=readme-ov-file#packages)를 설치합니다.

```bash
# -- for only for macOS: brew install is just fine :)
$ brew install just

# -- for all OS (Linux/Windows/macOS)
mkdir -p ~/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
# add `~/bin` to the paths that your shell searches for executables
# this line should be added to your shells initialization file,
# e.g. `~/.bashrc` or `~/.zshrc`
export PATH="$PATH:$HOME/bin"
```

`just`를 입력하면 [Justfile](./Justfile)의 명령어 집합을 확인할 수 있습니다.

```bash
$ just
Available recipes:
    default      # list all tasks
    gen-env-file # Generate .env file
    install-uv   # Install uv
    venv         # Set up Python virtual environment
```

### 2. 가상환경 셋팅
```bash
just venv
```

### 3. 예제 실행

#### 3.1. Quickstart Demo
> [demo/quickstart.ipynb](./demo/quickstart.ipynb) 실행하여 데모 실행해 봅니다.
> 
> 데모는 로컬 milvus db 생성하여 실습합니다.

### 3.2. tutorials

> [Build-RAG.ipynb](./demo/tutorials/Build-RAG.ipynb) 실행하여 tutorial을 실행해봅니다.

### 3.3. Shilla example

> Shilla 관련 예제는 examples/shilla/ 폴더 하위 각 example_XXX/ 폴더 내 README 파일을 참고해주세요.
>
> 아래는 예제 가이드입니다.
>
> 1. `example_human_gen` 예제 가이드 -> [example_human_gen guide](./demo/examples/shilla/example_human_gen/README.md)
> 2. `example_db_desc_gen` 예제 가이드 -> [example_db_desc_gen guide](./demo/examples/shilla/example_db_desc_gen/README.md)

### 3.4. RetrievalEvaluator

> Retriever를 평가할 수 있는 예시입니다. 
> 
> [demo/examples/retrieval_evaluator/](./demo/examples/retrieval_evaluator/) 경로를 참고해주세요~