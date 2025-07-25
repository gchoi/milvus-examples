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

# Image search dataseet
image-search-dataset:
    wget https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/reverse_image_search.zip && \
    mkdir -p image-search-dataset && \
    cp ./reverse_image_search.zip ./image-search-dataset/reverse_image_search.zip && \
    cd image-search-dataset && \
    unzip -q -o reverse_image_search.zip && \
    rm -r reverse_image_search.zip && \
    cd .. && \
    rm -r reverse_image_search.zip

# Multi modal dataset
multi-modal-dataset:
    uv pip install -e FlagEmbedding && \
    wget https://github.com/milvus-io/bootcamp/releases/download/data/amazon_reviews_2023_subset.tar.gz && \
    mkdir -p amazon_reviews_2023_subset && \
    mv amazon_reviews_2023_subset.tar.gz ./amazon_reviews_2023_subset/amazon_reviews_2023_subset.tar.gz && \
    cd amazon_reviews_2023_subset && \
    tar -xzf amazon_reviews_2023_subset.tar.gz && \
    rm -r amazon_reviews_2023_subset.tar.gz && \
    wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth
