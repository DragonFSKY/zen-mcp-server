"""Registry loader for custom OpenAI-compatible endpoints."""

from __future__ import annotations

from utils.env import get_env

from ..shared import ModelCapabilities, ProviderType
from .base import CAPABILITY_FIELD_NAMES, CapabilityModelRegistry


class CustomEndpointModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/custom_models.json``."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__(
            env_var_name="CUSTOM_MODELS_CONFIG_PATH",
            default_filename="custom_models.json",
            provider=ProviderType.CUSTOM,
            friendly_prefix="Custom ({model})",
            config_path=config_path,
        )

        custom_model = (get_env("CUSTOM_MODEL_NAME", "") or "").strip()
        if custom_model and not self.resolve(custom_model):
            self.model_map[custom_model] = ModelCapabilities(
                provider=ProviderType.CUSTOM,
                model_name=custom_model,
                friendly_name=f"Custom ({custom_model})",
                description="Custom model configured by CUSTOM_MODEL_NAME",
                context_window=32_768,
                max_output_tokens=16_000,
                intelligence_score=6,
            )
            self.alias_map[custom_model.lower()] = custom_model

    def _finalise_entry(self, entry: dict) -> tuple[ModelCapabilities, dict]:
        filtered = {k: v for k, v in entry.items() if k in CAPABILITY_FIELD_NAMES}
        filtered.setdefault("provider", ProviderType.CUSTOM)
        capability = ModelCapabilities(**filtered)
        return capability, {}
