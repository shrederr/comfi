# Watermark Remover — RunPod Serverless

Flux Kontext + Watermark Remover LoRA on RunPod Serverless.

## Deploy

1. Push this repo to GitHub
2. RunPod → Serverless → New Endpoint → Import GitHub Repository
3. Select GPU: **24 GB VRAM** (RTX A5000 / RTX 4090 / A100)
4. Set env var: `HF_TOKEN=hf_your_token`
5. Deploy

## API

POST `https://api.runpod.ai/v2/{endpoint_id}/run`

```json
{
  "input": {
    "image": "<base64 encoded image>",
    "prompt": "Remove the watermark from this photo, preserve all details",
    "steps": 28,
    "guidance_scale": 2.5,
    "seed": 42,
    "max_dim": 1024
  }
}
```

## Test

```bash
export RUNPOD_API_KEY=your_key
export RUNPOD_ENDPOINT_ID=your_endpoint_id
python test_local.py photo.jpg
```
