#!/bin/bash

echo "Starting LLM service..."
echo "HATHORA_HOSTNAME: $HATHORA_HOSTNAME"
echo "HATHORA_DEFAULT_PORT: $HATHORA_DEFAULT_PORT"
echo "HATHORA_REGION: $HATHORA_REGION"

exec uvicorn main:app --host 0.0.0.0 --port 8000
