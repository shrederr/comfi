FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git \
    transformers \
    accelerate \
    peft \
    "Pillow>=12.0" \
    runpod \
    sentencepiece \
    protobuf

# Download models at build time (baked into image = fast cold start)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python -c "\
from huggingface_hub import login; \
import os; \
login(token=os.environ['HF_TOKEN']); \
from diffusers import FluxKontextPipeline; \
pipe = FluxKontextPipeline.from_pretrained('black-forest-labs/FLUX.1-Kontext-dev', torch_dtype='auto'); \
from huggingface_hub import hf_hub_download; \
hf_hub_download('prithivMLmods/Kontext-Watermark-Remover', filename='Kontext-Watermark-Remover.safetensors'); \
print('Models downloaded!') \
"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
