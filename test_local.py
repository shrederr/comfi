"""
Test script to call the RunPod serverless endpoint.
Usage: python test_local.py <image_path> [endpoint_id] [api_key]
"""

import sys
import base64
import requests
import time
import os

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")


def main():
    image_path = sys.argv[1]
    endpoint_id = sys.argv[2] if len(sys.argv) > 2 else ENDPOINT_ID
    api_key = sys.argv[3] if len(sys.argv) > 3 else RUNPOD_API_KEY

    if not endpoint_id or not api_key:
        print("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY env vars, or pass as args")
        sys.exit(1)

    # Read and encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"Sending {image_path} ({len(image_b64) // 1024} KB base64)...")

    # Submit job
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "input": {
            "image": image_b64,
            "steps": 28,
            "guidance_scale": 2.5,
        }
    }

    resp = requests.post(url, json=payload, headers=headers)
    job = resp.json()
    job_id = job["id"]
    print(f"Job submitted: {job_id}")

    # Poll for result
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    while True:
        resp = requests.get(status_url, headers=headers)
        data = resp.json()
        status = data["status"]
        print(f"  Status: {status}")

        if status == "COMPLETED":
            result_b64 = data["output"]["image"]
            out_path = image_path.rsplit(".", 1)[0] + "_clean.png"
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(result_b64))
            print(f"Saved: {out_path}")
            break
        elif status == "FAILED":
            print(f"FAILED: {data}")
            break

        time.sleep(5)


if __name__ == "__main__":
    main()
