from typing import List, Optional, Dict, Any
from openai import AzureOpenAI, OpenAI
import os
import time


class BaseModel:
    """Base model class for LLM generation with reasoning and non-reasoning modes."""
    
    def __init__(
        self,
        backend: str = "openai",
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "2025-04-01-preview",
        model: str = "gpt-4o",
        temperature: float = 0.2,
        reasoning_effort: str = "medium",
        use_reasoning: bool = False,
        max_output_tokens: int = 2048,
        top_p: float = 1.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.backend = backend
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.use_reasoning = use_reasoning
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = self._client_init()

    def _client_init(self):
        """Initialize the appropriate client based on backend."""
        if self.backend == "openai":
            return OpenAI(api_key=self.api_key if self.api_key else os.environ.get("OPENAI_API_KEY"))
        elif self.backend == "azure":
            return AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
            )
        elif self.backend == "gemini":
            return OpenAI(
                api_key=self.api_key if self.api_key else os.environ.get("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            return OpenAI(api_key="EMPTY", base_url=self.backend)

    def generation(
        self,
        messages: List[Dict[str, Any]],
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate response from the model.
        
        Args:
            messages: List of message dictionaries with role and content
            max_output_tokens: Optional override for max tokens
            
        Returns:
            Dictionary with:
                - output: The generated text
                - reason: Reasoning summary (if use_reasoning=True)
                - usage: Dict with input and output token counts
        """
        if self.use_reasoning:
            return self._generation_reasoning(messages, max_output_tokens)
        else:
            return self._generation_standard(messages, max_output_tokens)

    def _generation_standard(
        self,
        messages: List[Dict[str, Any]],
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Standard generation with temperature control."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=self.temperature,
                    max_output_tokens=max_output_tokens or self.max_output_tokens,
                    top_p=self.top_p,
                )
                
                output_text = response.output_text
                usage = {
                    "input": response.usage.input_tokens if response.usage else 0,
                    "output": response.usage.output_tokens if response.usage else 0,
                }
                
                return {
                    "output": output_text,
                    "reason": "",  # No reasoning summary in standard mode
                    "usage": usage,
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_backoff * (2 ** attempt))

    def _generation_reasoning(
        self,
        messages: List[Dict[str, Any]],
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reasoning generation with effort parameter (for o1/o3/gpt-5 models)."""
        for msg in messages:
            if msg["role"] == "system":
                msg["role"] = "user"
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    reasoning={"effort": self.reasoning_effort, "summary": "auto"},
                    input=messages,
                    max_output_tokens=max_output_tokens or self.max_output_tokens,
                    top_p=self.top_p,
                )
                
                output_text = response.output_text
                reason_summary = ""
                if response.output and len(response.output) > 0 and response.output[0].summary:
                    reason_summary = response.output[0].summary[0].text
                
                usage = {
                    "input": response.usage.input_tokens if response.usage else 0,
                    "output": response.usage.output_tokens if response.usage else 0,
                }
                
                return {
                    "output": output_text,
                    "reason": reason_summary,
                    "usage": usage,
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_backoff * (2 ** attempt))
