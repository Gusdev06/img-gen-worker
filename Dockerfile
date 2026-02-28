FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Dependencias Python
RUN pip install --no-cache-dir \
    diffusers>=0.31.0 \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    safetensors>=0.4.0 \
    sentencepiece \
    protobuf \
    runpod>=1.6.0 \
    boto3>=1.34.0 \
    Pillow>=10.0.0 \
    requests>=2.31.0

# Download dos modelos durante o build
# Isso faz com que os modelos fiquem no snapshot do RunPod
# e nao precisem ser baixados a cada cold start
COPY download_models.py .
RUN python download_models.py

# Copia o handler e utils
COPY handler.py .
COPY utils.py .

CMD ["python", "-u", "handler.py"]