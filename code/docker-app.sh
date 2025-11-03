docker stop legal_llm_demo
docker rm legal_llm_demo
# docker build -t legal_llm:v1 .
docker run -it --env-file .env --gpus="device=2" --name legal_llm_demo -p 50002:50002 -v <path-to-GrahakNyay-repo>/code:/workspace -v <absolute-path-to-huggingface-cache>:/workspace/cache legal_llm:v1 bash


# Build the Docker image with the tag 'your_image_name:your_image_tag'
# docker build -t <your_image_name>:<your_image_tag> .

# Run the Docker container with the specified environment variables, GPU settings, port mappings, and volume mounts
# docker run -it --env-file .env --gpus="device=<your_gpu_device>" --name <your_container_name> -p <host_port>:<container_port> -v <path_to_local_code>:/workspace -v <path_to_local_cache>:/workspace/cache <your_image_name>:<your_image_tag>