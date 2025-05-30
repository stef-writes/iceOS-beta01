from app.llm_providers.openai_handler import OpenAIHandler
from app.llm_providers.anthropic_handler import AnthropicHandler
from app.llm_providers.google_gemini_handler import GoogleGeminiHandler
from app.llm_providers.deepseek_handler import DeepSeekHandler
from app.models.config import LLMConfig, ModelProvider
from typing import Dict, Any, Optional, Tuple

class LLMService:
    """
    Service abstraction for LLM calls. Routes to the correct handler based on provider.
    """
    def __init__(self):
        self.handlers = {
            ModelProvider.OPENAI: OpenAIHandler(),
            ModelProvider.ANTHROPIC: AnthropicHandler(),
            ModelProvider.GOOGLE: GoogleGeminiHandler(),
            ModelProvider.DEEPSEEK: DeepSeekHandler(),
        }

    async def generate(
        self,
        llm_config: LLMConfig,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[list] = None
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[str]]:
        """
        Generate text using the specified LLM provider.
        """
        provider = llm_config.provider
        handler = self.handlers.get(provider)
        if not handler:
            return "", None, f"No handler for provider: {provider}"
        return await handler.generate_text(
            llm_config=llm_config,
            prompt=prompt,
            context=context or {},
            tools=tools
        )
