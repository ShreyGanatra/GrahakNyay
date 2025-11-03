#!/bin/bash
docker stop tgi-server
docker rm tgi-server

# set -a
# source .env
# set +a

model=meta-llama/Llama-3.1-8B-Instruct
# model=microsoft/phi-4
# model=openai/gpt-oss-20b
# model=google/gemma-3-12b-it
volume=<path-to-huggingface-cache> # share a volume with the Docker container to avoid downloading weights every run

docker run -d --name tgi-server --gpus '"device=2"' -p 8080:80 \
    -e HF_TOKEN=<HF_TOKEN_ID> \
    -v $volume:/data  ghcr.io/huggingface/text-generation-inference:3.2.1 \
    --model-id $model \
    --cuda-memory-fraction 0.5
