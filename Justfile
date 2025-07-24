# Justfile

VENV := ".venv"
PYTHON_VER := "3.12"


# list all tasks
default:
  @just --list

# Install uv
install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up Python virtual environment
venv: install-uv
    #!/usr/bin/env sh
    if [ "$(uname)" = "Darwin" ] || [ "$(uname)" = "Linux" ]; then
        echo "Installing virtual env on Darwin or Linux..."
        uv venv {{ VENV }} --python {{ PYTHON_VER }}
        . {{ VENV }}/bin/activate && uv sync
    else
        echo "Installing virtual env on Windows..."
        uv venv {{ VENV }} --python {{ PYTHON_VER }}
        . {{ VENV }}/Scripts/activate && uv sync
    fi

# Generate .env file
gen-env-file:
	cp ./.env.example ./.env

# Prepare the data Build-RAG.ipynb for tutorials
prepare-for-build-rag:
    ollama pull mxbai-embed-large
    ollama pull gemma3:latest
    brew install wget
    wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
    unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
    . {{ VENV }}/bin/activate && \
    uv sync --group tutorials