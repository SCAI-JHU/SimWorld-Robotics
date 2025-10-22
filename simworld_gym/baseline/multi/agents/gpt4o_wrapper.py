import base64
import cv2
import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from PIL import Image


class GPT4oVision:
    """Wrapper for GPT-4 Vision API with support for OpenAI and Gemini backends."""
    
    def __init__(self, backend="openai", model="gpt-4o"):
        self.model = model
        if backend == "openai":
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif backend == "gemini":
            self.client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            self.client = OpenAI(api_key="EMPTY", base_url=backend)

    @staticmethod
    def _img_to_b64(img: np.ndarray) -> str:
        """Convert numpy array image to base64 encoded data URL."""
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[np.ndarray] = None,
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> str:
        """Send a chat request with optional image to the LLM."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image is not None:
            content.insert(0, {
                "type": "image_url",
                "image_url": {"url": self._img_to_b64(image)}
            })

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content
