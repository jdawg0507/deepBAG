"""
Unified LLM Client interface for deepBAG.
Supports:
1. Nebius Token Factory (OpenAI-compatible client, https://api.tokenfactory.nebius.com/v1/)
2. Google Gemini (via google.generativeai, REST transport)

Provides unified:
- generate(system_prompt, user_prompt, temperature, model) -> str
- generate_json(system_prompt, user_prompt, schema, temperature) -> dict
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List

from core.telemetry import TelemetryManager

logger = logging.getLogger("deepbag.llm_client")

class UnifiedLLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.nebius_key = os.environ.get("NEBIUS_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        # Decide provider
        if provider:
            self.provider = provider.lower()
        elif self.nebius_key:
            self.provider = "nebius"
        else:
            self.provider = "gemini"
            
        self.model = model
        self._init_client()

    def _init_client(self):
        if self.provider == "nebius":
            if not self.nebius_key:
                logger.warning("NEBIUS_API_KEY not found in environment; falling back to Gemini.")
                self.provider = "gemini"
                return self._init_client()
                
            from openai import OpenAI
            base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
            self.model = self.model or os.environ.get("NEBIUS_LLM_MODEL", "nvidia/Nemotron-3-Super-120B-A12B")
            self.client = OpenAI(base_url=base_url, api_key=self.nebius_key)
            logger.info(f"Initialized Nebius LLM client with model {self.model}")
            
        elif self.provider == "gemini":
            import google.generativeai as genai
            if self.gemini_key:
                genai.configure(api_key=self.gemini_key, transport="rest")
            self.model = self.model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
            self.client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Gemini LLM client with model {self.model}")
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        model_name: Optional[str] = None,
    ) -> str:
        target_model = model_name or self.model
        with TelemetryManager.start_span("deepbag.llm_generate", {
            "provider": self.provider,
            "model": target_model,
            "temperature": temperature
        }) as span:
            if self.provider == "nebius":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                )
                text = response.choices[0].message.content or ""
                return text.strip()
                
            elif self.provider == "gemini":
                import google.generativeai as genai
                gen_config = genai.types.GenerationConfig(
                    temperature=temperature,
                )
                active_model = self.client
                if model_name or system_prompt:
                    active_model = genai.GenerativeModel(
                        model_name=target_model,
                        system_instruction=system_prompt if system_prompt else None,
                        generation_config=gen_config
                    )
                
                response = active_model.generate_content(user_prompt)
                return (response.text or "").strip()

    def generate_json(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response formatted strictly as a JSON object"""
        json_instruction = "\nReturn ONLY a valid, parseable JSON object without markdown formatting, codeblocks, or extra text."
        sys_p = (system_prompt or "") + json_instruction
        raw_text = self.generate(
            user_prompt=user_prompt,
            system_prompt=sys_p,
            temperature=temperature,
            model_name=model_name
        )
        
        # Clean markdown fences if any
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}\nRaw output:\n{raw_text}")
            raise
