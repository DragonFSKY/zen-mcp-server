"""OpenRouter provider implementation."""

import json
import logging
import os
from typing import Optional

from .base import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    RangeTemperatureConstraint,
)
from .openai_compatible import OpenAICompatibleProvider
from .openrouter_registry import OpenRouterModelRegistry


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter unified API provider.

    OpenRouter provides access to multiple AI models through a single API endpoint.
    See https://openrouter.ai for available models and pricing.
    """

    FRIENDLY_NAME = "OpenRouter"

    # Custom headers required by OpenRouter
    DEFAULT_HEADERS = {
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/BeehiveInnovations/zen-mcp-server"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Zen MCP Server"),
    }

    # Model registry for managing configurations and aliases
    _registry: Optional[OpenRouterModelRegistry] = None

    def __init__(self, api_key: str, **kwargs):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            **kwargs: Additional configuration
        """
        base_url = "https://openrouter.ai/api/v1"
        super().__init__(api_key, base_url=base_url, **kwargs)

        # Initialize model registry
        if OpenRouterProvider._registry is None:
            OpenRouterProvider._registry = OpenRouterModelRegistry()
            # Log loaded models and aliases only on first load
            models = self._registry.list_models()
            aliases = self._registry.list_aliases()
            logging.info(f"OpenRouter loaded {len(models)} models with {len(aliases)} aliases")

        # Parse reasoning effort configuration
        self._parse_reasoning_config()

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model aliases to OpenRouter model names.

        Args:
            model_name: Input model name or alias

        Returns:
            Resolved OpenRouter model name
        """
        # Try to resolve through registry
        config = self._registry.resolve(model_name)

        if config:
            if config.model_name != model_name:
                logging.info(f"Resolved model alias '{model_name}' to '{config.model_name}'")
            return config.model_name
        else:
            # If not found in registry, return as-is
            # This allows using models not in our config file
            logging.debug(f"Model '{model_name}' not found in registry, using as-is")
            return model_name

    def _parse_reasoning_config(self):
        """Parse reasoning effort configuration from environment variables."""
        # Parse global default reasoning effort
        self.default_reasoning_effort = os.environ.get("OPENROUTER_DEFAULT_REASONING_EFFORT", "").lower().strip()

        # Auto-map max to high (OpenRouter doesn't support max)
        if self.default_reasoning_effort == "max":
            logging.info(
                "Mapping OPENROUTER_DEFAULT_REASONING_EFFORT='max' to 'high' "
                "(OpenRouter doesn't support 'max' level)"
            )
            self.default_reasoning_effort = "high"
        elif self.default_reasoning_effort and self.default_reasoning_effort not in [
            "minimal",
            "low",
            "medium",
            "high",
        ]:
            logging.warning(
                f"Invalid OPENROUTER_DEFAULT_REASONING_EFFORT value: '{self.default_reasoning_effort}'. "
                "Valid values are: minimal, low, medium, high, max (auto-mapped to high)"
            )
            self.default_reasoning_effort = ""

        # Parse model-specific reasoning effort map (JSON format)
        self.reasoning_effort_map = {}
        map_str = os.environ.get("OPENROUTER_REASONING_EFFORT_MAP", "").strip()
        if map_str:
            try:
                raw_map = json.loads(map_str)
                if not isinstance(raw_map, dict):
                    logging.error(f"OPENROUTER_REASONING_EFFORT_MAP must be a JSON object, got: {type(raw_map)}")
                    raw_map = {}

                for model, effort in raw_map.items():
                    effort = effort.strip().lower() if isinstance(effort, str) else str(effort).lower()

                    # Auto-map max to high
                    if effort == "max":
                        effort = "high"
                        logging.debug(f"Mapped 'max' to 'high' for model '{model}' (OpenRouter limitation)")

                    # Validate effort value
                    if effort not in ["minimal", "low", "medium", "high"]:
                        logging.warning(
                            f"Invalid reasoning effort '{effort}' for model '{model}'. "
                            "Valid values are: minimal, low, medium, high, max (auto-mapped to high)"
                        )
                        continue

                    # Try to resolve model alias to canonical name
                    normalized_model = self._normalize_model_key(model)
                    self.reasoning_effort_map[normalized_model] = effort
                    if normalized_model != model:
                        logging.debug(f"Normalized model key '{model}' to '{normalized_model}'")

                if self.reasoning_effort_map:
                    logging.info(f"Loaded reasoning effort configuration for {len(self.reasoning_effort_map)} models")

            except json.JSONDecodeError as e:
                logging.error(
                    f"Failed to parse OPENROUTER_REASONING_EFFORT_MAP as JSON: {e}\n"
                    f'Expected format: {{"model1": "effort1", "model2": "effort2"}}'
                )

    def _normalize_model_key(self, key: str) -> str:
        """Normalize a model key, resolving aliases to canonical names.

        Args:
            key: Model name or alias

        Returns:
            Canonical model name if resolved, original key otherwise
        """
        # Try to resolve through registry
        config = self._registry.resolve(key)
        if config:
            return config.model_name
        return key

    def _supports_reasoning(self, model_name: str) -> bool:
        """Check if an OpenAI model supports reasoning effort parameter.

        Currently only OpenAI models are supported via OpenRouter:
        - O3 series: uses reasoning_effort parameter
        - O4 series: uses reasoning_effort parameter
        - GPT-5 series: uses reasoning parameter (nested object)

        Args:
            model_name: Model name to check

        Returns:
            True if model supports reasoning effort, False otherwise
        """
        lower_name = model_name.lower()

        # Only support OpenAI models for now
        if not any(prefix in lower_name for prefix in ["openai/", "gpt-", "gpt5", "o3", "o4"]):
            return False

        # GPT-5 models support reasoning
        if "gpt-5" in lower_name or "gpt5" in lower_name:
            return True

        # O3 models support reasoning_effort
        if "o3" in lower_name:
            return True

        # O4 models might support reasoning (future-proofing)
        if "o4" in lower_name:
            return True

        return False

    def _get_reasoning_effort(self, model_name: str) -> Optional[str]:
        """Get the reasoning effort configuration for a specific model.

        Configuration priority (highest to lowest):
        1. OPENROUTER_REASONING_EFFORT_MAP - model-specific entry
        2. OPENROUTER_REASONING_EFFORT_MAP - wildcard pattern matching
        3. OPENROUTER_DEFAULT_REASONING_EFFORT - global default
        4. DEFAULT_THINKING_MODE_THINKDEEP - system fallback

        Args:
            model_name: Model name to get configuration for

        Returns:
            Reasoning effort value (minimal/low/medium/high) or None
        """
        # Check if model supports reasoning
        if not self._supports_reasoning(model_name):
            return None

        # Normalize the model name
        normalized_name = self._normalize_model_key(model_name)

        # Check model-specific mapping first (highest priority)
        if normalized_name in self.reasoning_effort_map:
            effort = self.reasoning_effort_map[normalized_name]
            logging.debug(f"Using model-specific reasoning effort '{effort}' for {normalized_name}")
            return effort

        # Check for wildcard patterns (e.g., "*" for all models)
        for pattern, effort in self.reasoning_effort_map.items():
            if "*" in pattern:
                if pattern == "*":
                    # Match all models
                    logging.debug(f"Using wildcard reasoning effort '{effort}' for {normalized_name} (pattern: *)")
                    return effort
                # Support prefix patterns like "openai/gpt-5*"
                pattern_base = pattern.replace("*", "")
                if normalized_name.startswith(pattern_base) or model_name.startswith(pattern_base):
                    logging.debug(
                        f"Using wildcard reasoning effort '{effort}' for {normalized_name} (pattern: {pattern})"
                    )
                    return effort
                # Support suffix patterns like "*-mini"
                if pattern.startswith("*"):
                    pattern_suffix = pattern[1:]  # Remove the *
                    if normalized_name.endswith(pattern_suffix) or model_name.endswith(pattern_suffix):
                        logging.debug(
                            f"Using wildcard reasoning effort '{effort}' for {normalized_name} (pattern: {pattern})"
                        )
                        return effort

        # Use global default if configured
        if self.default_reasoning_effort:
            logging.debug(f"Using default reasoning effort '{self.default_reasoning_effort}' for {normalized_name}")
            return self.default_reasoning_effort

        # Optional: Fall back to DEFAULT_THINKING_MODE_THINKDEEP if configured
        thinking_mode = os.environ.get("DEFAULT_THINKING_MODE_THINKDEEP", "").lower().strip()
        if thinking_mode:
            # Map thinking modes to reasoning efforts
            mode_mapping = {
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "max": "high",  # Map max to high since OpenRouter doesn't support max
            }
            if thinking_mode in mode_mapping:
                effort = mode_mapping[thinking_mode]
                logging.debug(
                    f"Using reasoning effort '{effort}' for {normalized_name} "
                    f"(fallback from DEFAULT_THINKING_MODE_THINKDEEP='{thinking_mode}')"
                )
                return effort

        return None

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a model.

        Args:
            model_name: Name of the model (or alias)

        Returns:
            ModelCapabilities from registry or generic defaults
        """
        # Try to get from registry first
        capabilities = self._registry.get_capabilities(model_name)

        if capabilities:
            return capabilities
        else:
            # Resolve any potential aliases and create generic capabilities
            resolved_name = self._resolve_model_name(model_name)

            logging.debug(
                f"Using generic capabilities for '{resolved_name}' via OpenRouter. "
                "Consider adding to custom_models.json for specific capabilities."
            )

            # Create generic capabilities with conservative defaults
            capabilities = ModelCapabilities(
                provider=ProviderType.OPENROUTER,
                model_name=resolved_name,
                friendly_name=self.FRIENDLY_NAME,
                context_window=32_768,  # Conservative default context window
                max_output_tokens=32_768,
                supports_extended_thinking=False,
                supports_system_prompts=True,
                supports_streaming=True,
                supports_function_calling=False,
                temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            )

            # Mark as generic for validation purposes
            capabilities._is_generic = True

            return capabilities

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.OPENROUTER

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is allowed.

        As the catch-all provider, OpenRouter accepts any model name that wasn't
        handled by higher-priority providers. OpenRouter will validate based on
        the API key's permissions and local restrictions.

        Args:
            model_name: Model name to validate

        Returns:
            True if model is allowed, False if restricted
        """
        # Check model restrictions if configured
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if restriction_service:
            # Check if model name itself is allowed
            if restriction_service.is_allowed(self.get_provider_type(), model_name):
                return True

            # Also check aliases - model_name might be an alias
            model_config = self._registry.resolve(model_name)
            if model_config and model_config.aliases:
                for alias in model_config.aliases:
                    if restriction_service.is_allowed(self.get_provider_type(), alias):
                        return True

            # If restrictions are configured and model/alias not in allowed list, reject
            return False

        # No restrictions configured - accept any model name as the fallback provider
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
        """Generate content using the OpenRouter API.

        Args:
            prompt: User prompt to send to the model
            model_name: Name of the model (or alias) to use
            system_prompt: Optional system prompt for model behavior
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse with generated content and metadata
        """
        # Process thinking parameters with environment config override
        # The base class method already handles all the logic we need
        kwargs = self.process_thinking_parameters(model_name, **kwargs)

        # Resolve model alias to actual OpenRouter model name
        resolved_model = self._resolve_model_name(model_name)

        # Always disable streaming for OpenRouter
        # MCP doesn't use streaming, and this avoids issues with O3 model access
        if "stream" not in kwargs:
            kwargs["stream"] = False

        # Call parent method with resolved model name
        return super().generate_content(
            prompt=prompt,
            model_name=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode.

        Currently, no models via OpenRouter support extended thinking.
        This may change as new models become available.

        Args:
            model_name: Model to check

        Returns:
            False (no OpenRouter models currently support thinking mode)
        """
        return False

    def list_models(self, respect_restrictions: bool = True) -> list[str]:
        """Return a list of model names supported by this provider.

        Args:
            respect_restrictions: Whether to apply provider-specific restriction logic.

        Returns:
            List of model names available from this provider
        """
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service() if respect_restrictions else None
        models = []

        if self._registry:
            for model_name in self._registry.list_models():
                # =====================================================================================
                # CRITICAL ALIAS-AWARE RESTRICTION CHECKING (Fixed Issue #98)
                # =====================================================================================
                # Previously, restrictions only checked full model names (e.g., "google/gemini-2.5-pro")
                # but users specify aliases in OPENROUTER_ALLOWED_MODELS (e.g., "pro").
                # This caused "no models available" error even with valid restrictions.
                #
                # Fix: Check both model name AND all aliases against restrictions
                # TEST COVERAGE: tests/test_provider_routing_bugs.py::TestOpenRouterAliasRestrictions
                # =====================================================================================
                if restriction_service:
                    # Get model config to check aliases as well
                    model_config = self._registry.resolve(model_name)
                    allowed = False

                    # Check if model name itself is allowed
                    if restriction_service.is_allowed(self.get_provider_type(), model_name):
                        allowed = True

                    # CRITICAL: Also check aliases - this fixes the alias restriction bug
                    if not allowed and model_config and model_config.aliases:
                        for alias in model_config.aliases:
                            if restriction_service.is_allowed(self.get_provider_type(), alias):
                                allowed = True
                                break

                    if not allowed:
                        continue

                models.append(model_name)

        return models

    def list_all_known_models(self) -> list[str]:
        """Return all model names known by this provider, including alias targets.

        Returns:
            List of all model names and alias targets known by this provider
        """
        all_models = set()

        if self._registry:
            # Get all models and aliases from the registry
            all_models.update(model.lower() for model in self._registry.list_models())
            all_models.update(alias.lower() for alias in self._registry.list_aliases())

            # For each alias, also add its target
            for alias in self._registry.list_aliases():
                config = self._registry.resolve(alias)
                if config:
                    all_models.add(config.model_name.lower())

        return list(all_models)

    def get_model_configurations(self) -> dict[str, ModelCapabilities]:
        """Get model configurations from the registry.

        For OpenRouter, we convert registry configurations to ModelCapabilities objects.

        Returns:
            Dictionary mapping model names to their ModelCapabilities objects
        """
        configs = {}

        if self._registry:
            # Get all models from registry
            for model_name in self._registry.list_models():
                # Only include models that this provider validates
                if self.validate_model_name(model_name):
                    config = self._registry.resolve(model_name)
                    if config and not config.is_custom:  # Only OpenRouter models, not custom ones
                        # Use ModelCapabilities directly from registry
                        configs[model_name] = config

        return configs

    def get_all_model_aliases(self) -> dict[str, list[str]]:
        """Get all model aliases from the registry.

        Returns:
            Dictionary mapping model names to their list of aliases
        """
        # Since aliases are now included in the configurations,
        # we can use the base class implementation
        return super().get_all_model_aliases()
