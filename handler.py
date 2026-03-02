"""
RunPod Serverless Handler — Flux Kontext Watermark Remover
"""

import runpod
import torch
import base64
import io
from PIL import Image
from diffusers import FluxKontextPipeline

# --- Load model once at startup ---
print("Loading Flux Kontext pipeline...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

print("Loading Watermark Remover LoRA...")
pipe.load_lora_weights(
    "prithivMLmods/Kontext-Watermark-Remover",
    weight_name="Kontext-Watermark-Remover.safetensors",
    adapter_name="watermark_remover",
)
pipe.set_adapters(["watermark_remover"], adapter_weights=[1.0])
print("Model ready!")


def handler(job):
    """Process a single image: remove watermark."""
    job_input = job["input"]

    # Get image from base64
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No 'image' field in input (base64 encoded)"}

    image_bytes = base64.b64decode(image_b64)
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = input_image.size

    # Settings (with defaults matching RunningHub)
    prompt = job_input.get(
        "prompt",
        "Remove the watermark from this photo, preserve all details and textures",
    )
    steps = job_input.get("steps", 28)
    guidance = job_input.get("guidance_scale", 2.5)
    seed = job_input.get("seed", 42)
    max_dim = job_input.get("max_dim", 1024)

    # Resize if needed (keep aspect, multiple of 16)
    scale = min(max_dim / max(w, h), 1.0)
    w_new = int(w * scale) // 16 * 16
    h_new = int(h * scale) // 16 * 16
    img_resized = input_image.resize((w_new, h_new), Image.LANCZOS)

    # Generate
    result = pipe(
        image=img_resized,
        prompt=prompt,
        guidance_scale=guidance,
        width=w_new,
        height=h_new,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]

    # Resize back to original dimensions
    result = result.resize((w, h), Image.LANCZOS)

    # Encode to base64 PNG
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image": result_b64,
        "width": w,
        "height": h,
        "steps": steps,
        "guidance_scale": guidance,
    }


runpod.serverless.start({"handler": handler})
