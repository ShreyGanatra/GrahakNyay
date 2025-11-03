# Grahak Nyay

Grahak Nyay is a consumer grievance assistance tool designed to help users with their consumer-related queries and issues. This project leverages advanced natural language processing models to provide accurate and helpful responses to user queries.


## Setup

### Prerequisites

- Docker
- Python 3.8+
- pip

### Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

#### Environment Variables

Create a `.env` file in the code directory and add the following environment variables:
```
HF_TOKEN=<your_huggingface_token>
BASE_URL=/consumer_chatbot
```

#### Method 1: Install Libraries and Run Python App

2. Install the required Python packages:
    ```sh
    pip install -r code/requirements.txt
    ```

3. Run the Flask application:
    ```sh
    cd code
    python app.py
    ```

#### Method 2: Build and Run Docker Container

2. Build and run the Docker container:
    ```sh
    cd code

    docker build -t <your_image_name>:<your_image_tag> .

    docker run -it --env-file .env --gpus="device=<your_gpu_device>" --name <your_container_name> -p <host_port>:<container_port> -v <path_to_local_code>:/workspace -v <path_to_local_cache>:/workspace/cache <your_image_name>:<your_image_tag>
    ```

### Backend Setup

For both methods, you need to run the Docker LLM server:

1. Run the Docker LLM server:
    ```sh
    cd code
    ./tgi_docker.sh
    ```

### Usage

1. Open your web browser and navigate to `http://localhost:50002/consumer_chatbot`.

2. Interact with the chatbot by typing your queries in the chat interface.



## Project Components

### 

app.py



- Main Flask application file.
- Defines routes for the chatbot and session management.
- Handles chat interactions and session history.

### 

utils.py



- Contains utility functions for loading data, creating vector stores, and initializing language models.

### 

index.html



- HTML template for the chatbot interface.
- Includes JavaScript for handling user interactions and displaying chat messages.

###

static/css/styles.css

- CSS file for styling the chatbot interface.

### 

docker_restart.sh



- Shell script to build and run the Docker container for the project.

### 

tgi_docker.sh



- Shell script to run the text generation inference server using Docker.

### 

rag_qa.csv



- CSV file containing question-answer pairs for the chatbot.


## License

This project is licensed under the MIT License. See the LICENSE file for details.
