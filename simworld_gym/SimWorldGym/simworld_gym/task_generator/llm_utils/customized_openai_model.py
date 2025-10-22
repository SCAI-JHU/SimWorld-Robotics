import time
import os
import asyncio
from typing import Optional, Union, List

from openai import OpenAI, AzureOpenAI, AsyncOpenAI


class CustomizedOpenAIModel:
    def __init__(
        self,
        model: str,
        backend: str = "openai",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = "2025-04-01-preview",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.model = model
        self.backend = backend
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = azure_api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = self._init_client()
        self._async_client: Optional[AsyncOpenAI] = None

    def _init_client(self):
        if self.backend == "openai":
            return OpenAI(api_key=self.api_key)
        if self.backend == "azure":
            return AzureOpenAI(api_key=self.api_key, azure_endpoint=self.azure_endpoint, api_version=self.azure_api_version)
        if self.backend == "gemini":
            return OpenAI(api_key=os.environ.get("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        return OpenAI(api_key=self.api_key or "", base_url=self.backend)

    @staticmethod
    def _to_messages(p: Union[str, list]):
        if isinstance(p, str):
            return [{"role": "user", "content": p}]
        return p

    def generate(
        self,
        prompt: Union[str, list],
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[float] = None,
        retry: int = 8,
        **kwargs,
    ) -> str:
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        for attempt in range(1, retry + 1):
            try:
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._to_messages(prompt),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=1,
                    stop=stop,
                    **kwargs,
                )
                return (r.choices[0].message.content or "")
            except Exception as e:
                wait = min(self.retry_backoff * (2 ** (attempt - 1)), 60)
                print(f"OpenAI generate error (attempt {attempt}/{retry}): {e}; retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"CustomizedOpenAIModel.generate failed after {retry} attempts")

    async def generate_async(
        self,
        prompts: List[Union[str, list]],
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[float] = None,
        retry: int = 8,
        **kwargs,
    ) -> List[str]:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature

        async def call_with_retries(p):
            for attempt in range(1, retry + 1):
                try:
                    if rate_limit_per_min is not None:
                        await asyncio.sleep(60 / rate_limit_per_min)
                    r = await self._async_client.chat.completions.create(
                        model=self.model,
                        messages=self._to_messages(p),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=1,
                        stop=stop,
                        **kwargs,
                    )
                    return (r.choices[0].message.content or "")
                except Exception as e:
                    wait = min(self.retry_backoff * (2 ** (attempt - 1)), 60)
                    print(f"OpenAI async generate error (attempt {attempt}/{retry}): {e}; retrying in {wait}s")
                    await asyncio.sleep(wait)
            raise RuntimeError("generate_async failed after retries")

        tasks = [call_with_retries(p) for p in prompts]
        return await asyncio.gather(*tasks)