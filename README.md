# LLM Serve - LLM Inference and Latency Analysis

This repository provides a FastAPI-based LLM inference service and tools for measuring end-to-end latency.

## Quick Start

1. Clone the repository and install dependencies:
    ```bash
    git clone <your-repo>
    cd llm-serve
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Run the service locally:
    ```bash
    python src/main.py
    ```

3. Test the health endpoint:
    ```bash
    curl http://localhost:8000/health
    ```

## Docker

To build and push a multi-arch image: