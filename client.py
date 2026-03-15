import os
import requests
MODEL = os.getenv("MODEL")
BASE_URL = "http://localhost:12346/v1/chat/completions"
image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image? One sentence."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    "max_tokens": 64,
}
 
r = requests.post(BASE_URL, json=payload, timeout=3000)
r.raise_for_status()
print(r.json()["choices"][0]["message"]["content"])
