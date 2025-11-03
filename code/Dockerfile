FROM nvcr.io/nvidia/pytorch:24.06-py3
# RUN apt update && apt install -y sqlite3 libsqlite3-dev

# Create working directory\
WORKDIR /workspace

# Copy contents
RUN python3 -m pip install --upgrade pip wheel


COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Copy the rest of the application code
COPY . .
EXPOSE 8080
EXPOSE 50002
ENV HF_HOME=/workspace/cache
ENV HUGGING_FACE_HUB_TOKEN=<hf-token-id>
# USER root
CMD ["python3", "app.py"]