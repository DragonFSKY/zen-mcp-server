"""DeepSeek model provider implementation."""

from typing import Optional, TYPE_CHECKING
import logging

from .openai_compatible import OpenAICompatibleProvider
from .base import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    create_temperature_constraint,
)

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

logger = logging.getLogger(__name__)


class DeepSeekModelProvider(OpenAICompatibleProvider):
    """Provider for DeepSeek's OpenAI-compatible API."""

    FRIENDLY_NAME = "DeepSeek"

    SUPPORTED_MODELS = {
        "deepseek-chat": ModelCapabilities(
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-chat",
            friendly_name="DeepSeek (Chat)",
            context_window=128_000,
            max_output_tokens=16_000,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            description="DeepSeek V3.1 chat model (non-reasoning mode)",
            aliases=["deepseek", "chat"],
        ),
        "deepseek-reasoner": ModelCapabilities(
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-reasoner",
            friendly_name="DeepSeek (Reasoner)",
            context_window=128_000,
            max_output_tokens=16_000,
            supports_extended_thinking=True,
            max_thinking_tokens=16_000,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            description="DeepSeek V3.1 reasoning model with thinking tokens",
            aliases=["reasoner"],
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize DeepSeek provider with API key and optional settings."""
        kwargs.setdefault("base_url", "https://api.deepseek.com")
        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.DEEPSEEK

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific DeepSeek model."""
        if model_name in self.SUPPORTED_MODELS:
            capabilities = self.SUPPORTED_MODELS[model_name]
        else:
            resolved_name = self._resolve_model_name(model_name)
            capabilities = self.SUPPORTED_MODELS.get(resolved_name)
            model_name = resolved_name

        if not capabilities:
            raise ValueError(f"Unsupported DeepSeek model: {model_name}")

        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(ProviderType.DEEPSEEK, capabilities.model_name, model_name):
            raise ValueError(f"DeepSeek model '{model_name}' is not allowed by restriction policy.")

        return capabilities

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported and allowed."""
        resolved_name = self._resolve_model_name(model_name)
        if resolved_name not in self.SUPPORTED_MODELS:
            return False
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(ProviderType.DEEPSEEK, resolved_name, model_name):
            logger.debug(
                f"DeepSeek model '{model_name}' -> '{resolved_name}' blocked by restrictions"
            )
            return False
        return True

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using DeepSeek API with alias resolution."""
        resolved_model_name = self._resolve_model_name(model_name)
        return super().generate_content(
            prompt=prompt,
            model_name=resolved_model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        resolved = self._resolve_model_name(model_name)
        capabilities = self.SUPPORTED_MODELS.get(resolved)
        return bool(capabilities and capabilities.supports_extended_thinking)

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Simple preference for DeepSeek models based on category."""
        if not allowed_models:
            return None
        # Currently no special preferences beyond first allowed model
        return allowed_models[0]
