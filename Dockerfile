FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies only (no model download)
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git \
    transformers \
    accelerate \
    peft \
    "Pillow>=12.0" \
    runpod \
    sentencepiece \
    protobuf

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
