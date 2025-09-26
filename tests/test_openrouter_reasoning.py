"""Tests for OpenRouter reasoning effort configuration."""

import os
from unittest.mock import MagicMock, patch

from providers.openrouter import OpenRouterProvider


class TestOpenRouterReasoningConfig:
    """Test reasoning effort configuration for OpenRouter provider."""

    def setup_method(self):
        """Setup for each test."""
        # Clear any existing environment variables
        for key in [
            "OPENROUTER_DEFAULT_REASONING_EFFORT",
            "OPENROUTER_REASONING_EFFORT_MAP",
            "DEFAULT_THINKING_MODE_THINKDEEP",
        ]:
            os.environ.pop(key, None)

    def teardown_method(self):
        """Cleanup after each test."""
        # Clear environment variables
        for key in [
            "OPENROUTER_DEFAULT_REASONING_EFFORT",
            "OPENROUTER_REASONING_EFFORT_MAP",
            "DEFAULT_THINKING_MODE_THINKDEEP",
        ]:
            os.environ.pop(key, None)

    def test_parse_reasoning_config_default(self):
        """Test parsing default reasoning effort configuration."""
        os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "medium"

        with patch("providers.openrouter.OpenRouterModelRegistry"):
            provider = OpenRouterProvider("test-api-key")
            assert provider.default_reasoning_effort == "medium"
            assert provider.reasoning_effort_map == {}

    def test_parse_reasoning_config_invalid_default(self):
        """Test handling invalid default reasoning effort."""
        os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "invalid"

        with patch("providers.openrouter.OpenRouterModelRegistry"):
            provider = OpenRouterProvider("test-api-key")
            assert provider.default_reasoning_effort == ""  # Should be cleared due to invalid value

    def test_parse_reasoning_config_map(self):
        """Test parsing model-specific reasoning effort map."""
        os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = '{"openai/gpt-5": "high", "openai/gpt-5-mini": "medium"}'

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            # Mock the registry's resolve method to simulate alias resolution
            mock_instance = MagicMock()
            mock_registry.return_value = mock_instance
            mock_instance.resolve.side_effect = lambda x: MagicMock(model_name=x) if "openai/" in x else None

            provider = OpenRouterProvider("test-api-key")
            provider._registry = mock_instance

            # Re-parse config after setting up the mock
            provider._parse_reasoning_config()

            assert "openai/gpt-5" in provider.reasoning_effort_map
            assert provider.reasoning_effort_map["openai/gpt-5"] == "high"
            assert "openai/gpt-5-mini" in provider.reasoning_effort_map
            assert provider.reasoning_effort_map["openai/gpt-5-mini"] == "medium"

    def test_parse_reasoning_config_invalid_effort_in_map(self):
        """Test handling invalid effort values in map."""
        os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = '{"openai/gpt-5": "invalid", "openai/gpt-5-mini": "medium"}'

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            mock_instance = MagicMock()
            mock_registry.return_value = mock_instance
            mock_instance.resolve.side_effect = lambda x: MagicMock(model_name=x) if "openai/" in x else None

            provider = OpenRouterProvider("test-api-key")
            provider._registry = mock_instance
            provider._parse_reasoning_config()

            # Invalid effort should be skipped
            assert "openai/gpt-5" not in provider.reasoning_effort_map
            assert "openai/gpt-5-mini" in provider.reasoning_effort_map

    def test_parse_reasoning_config_invalid_json(self):
        """Test handling invalid JSON format."""
        os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = '{"openai/gpt-5": "high", invalid json}'

        with patch("providers.openrouter.OpenRouterModelRegistry"):
            provider = OpenRouterProvider("test-api-key")
            # Should handle JSON error gracefully and leave map empty
            assert provider.reasoning_effort_map == {}

    def test_parse_reasoning_config_json_not_dict(self):
        """Test handling JSON that's not a dictionary."""
        os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = '["high", "medium", "low"]'

        with patch("providers.openrouter.OpenRouterModelRegistry"):
            provider = OpenRouterProvider("test-api-key")
            # Should handle non-dict JSON and leave map empty
            assert provider.reasoning_effort_map == {}

    def test_supports_reasoning_openai_models(self):
        """Test that OpenAI models are detected as supporting reasoning."""
        with patch("providers.openrouter.OpenRouterModelRegistry"):
            provider = OpenRouterProvider("test-api-key")

            # GPT-5 models
            assert provider._supports_reasoning("openai/gpt-5") is True
            assert provider._supports_reasoning("openai/gpt-5-mini") is True
            assert provider._supports_reasoning("openai/gpt-5-nano") is True
            assert provider._supports_reasoning("gpt5") is True
            assert provider._supports_reasoning("GPT-5-chat") is True

            # O3 models
            assert provider._supports_reasoning("openai/o3") is True
            assert provider._supports_reasoning("openai/o3-mini") is True
            assert provider._supports_reasoning("openai/o3-pro") is True

            # O4 models (future-proofing)
            assert provider._supports_reasoning("openai/o4-mini") is True

    def test_supports_reasoning_non_openai_models(self):
        """Test that non-OpenAI models are not supported."""
        with patch("providers.openrouter.OpenRouterModelRegistry"):
            provider = OpenRouterProvider("test-api-key")

            # These models are NOT supported even if they might support reasoning
            assert provider._supports_reasoning("anthropic/claude-4") is False
            assert provider._supports_reasoning("deepseek/deepseek-r1") is False
            assert provider._supports_reasoning("meta-llama/llama-3") is False
            assert provider._supports_reasoning("anthropic/claude-3") is False

    def test_get_reasoning_effort_with_map(self):
        """Test getting reasoning effort from model-specific map."""
        os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "low"
        os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = '{"openai/gpt-5": "high"}'

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            mock_instance = MagicMock()
            mock_registry.return_value = mock_instance
            mock_instance.resolve.side_effect = lambda x: MagicMock(model_name="openai/gpt-5") if "gpt-5" in x else None

            provider = OpenRouterProvider("test-api-key")
            provider._registry = mock_instance
            provider._parse_reasoning_config()

            # Should use map value over default
            effort = provider._get_reasoning_effort("openai/gpt-5")
            assert effort == "high"

    def test_get_reasoning_effort_with_default(self):
        """Test getting reasoning effort from default when not in map."""
        os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "medium"

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            mock_instance = MagicMock()
            mock_registry.return_value = mock_instance
            mock_instance.resolve.return_value = MagicMock(model_name="openai/gpt-5-nano")

            provider = OpenRouterProvider("test-api-key")
            provider._registry = mock_instance

            effort = provider._get_reasoning_effort("openai/gpt-5-nano")
            assert effort == "medium"

    def test_get_reasoning_effort_fallback_to_thinking_mode(self):
        """Test fallback to DEFAULT_THINKING_MODE_THINKDEEP."""
        os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "high"

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            mock_instance = MagicMock()
            mock_registry.return_value = mock_instance
            mock_instance.resolve.return_value = MagicMock(model_name="openai/gpt-5")

            provider = OpenRouterProvider("test-api-key")
            provider._registry = mock_instance

            effort = provider._get_reasoning_effort("openai/gpt-5")
            assert effort == "high"

    def test_get_reasoning_effort_max_maps_to_high(self):
        """Test that 'max' thinking mode maps to 'high' reasoning effort."""
        os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "max"

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            mock_instance = MagicMock()
            mock_registry.return_value = mock_instance
            mock_instance.resolve.return_value = MagicMock(model_name="openai/gpt-5")

            provider = OpenRouterProvider("test-api-key")
            provider._registry = mock_instance

            effort = provider._get_reasoning_effort("openai/gpt-5")
            assert effort == "high"  # max should map to high

    def test_generate_content_injects_reasoning_gpt5(self):
        """Test that generate_content injects nested reasoning parameter for GPT-5 models."""
        os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "medium"

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            with patch("providers.openai_compatible.OpenAICompatibleProvider.generate_content") as mock_generate:
                mock_instance = MagicMock()
                mock_registry.return_value = mock_instance
                mock_instance.resolve.return_value = MagicMock(model_name="openai/gpt-5")

                provider = OpenRouterProvider("test-api-key")
                provider._registry = mock_instance

                # Call generate_content
                provider.generate_content("test prompt", "openai/gpt-5")

                # Check that reasoning was added to kwargs (nested format for GPT-5)
                mock_generate.assert_called_once()
                call_kwargs = mock_generate.call_args[1]
                assert "reasoning" in call_kwargs
                assert call_kwargs["reasoning"] == {"effort": "medium"}

    def test_generate_content_injects_reasoning_o3(self):
        """Test that generate_content injects flat reasoning_effort parameter for O3 models."""
        os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "high"

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            with patch("providers.openai_compatible.OpenAICompatibleProvider.generate_content") as mock_generate:
                mock_instance = MagicMock()
                mock_registry.return_value = mock_instance
                mock_instance.resolve.return_value = MagicMock(model_name="openai/o3-mini")

                provider = OpenRouterProvider("test-api-key")
                provider._registry = mock_instance

                # Call generate_content
                provider.generate_content("test prompt", "openai/o3-mini")

                # Check that reasoning_effort was added to kwargs (flat format for O3)
                mock_generate.assert_called_once()
                call_kwargs = mock_generate.call_args[1]
                assert "reasoning_effort" in call_kwargs
                assert call_kwargs["reasoning_effort"] == "high"
                assert "reasoning" not in call_kwargs  # Should NOT have nested format

    def test_generate_content_no_reasoning_for_unsupported_models(self):
        """Test that generate_content doesn't inject reasoning for unsupported models."""
        os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "medium"

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            with patch("providers.openai_compatible.OpenAICompatibleProvider.generate_content") as mock_generate:
                mock_instance = MagicMock()
                mock_registry.return_value = mock_instance
                mock_instance.resolve.return_value = MagicMock(model_name="anthropic/claude-3")
                mock_instance.get_capabilities.return_value = MagicMock(supports_extended_thinking=False)

                provider = OpenRouterProvider("test-api-key")
                provider._registry = mock_instance

                # Call generate_content
                provider.generate_content("test prompt", "anthropic/claude-3")

                # Check that reasoning was NOT added to kwargs
                mock_generate.assert_called_once()
                call_kwargs = mock_generate.call_args[1]
                assert "reasoning" not in call_kwargs

    def test_wildcard_pattern_matching(self):
        """Test wildcard pattern matching in reasoning effort map."""
        os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = '{"openai/gpt-5*": "high"}'

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            mock_instance = MagicMock()
            mock_registry.return_value = mock_instance
            mock_instance.resolve.side_effect = lambda x: MagicMock(model_name=x) if "openai/" in x else None

            provider = OpenRouterProvider("test-api-key")
            provider._registry = mock_instance
            provider._parse_reasoning_config()

            # All GPT-5 variants should get high effort from wildcard
            assert provider._get_reasoning_effort("openai/gpt-5") == "high"
            assert provider._get_reasoning_effort("openai/gpt-5-mini") == "high"
            assert provider._get_reasoning_effort("openai/gpt-5-nano") == "high"

    def test_priority_chain(self):
        """Test the priority chain: map > default > thinking_mode."""
        os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "low"
        os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "medium"
        os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = '{"openai/gpt-5": "high"}'

        with patch("providers.openrouter.OpenRouterModelRegistry") as mock_registry:
            mock_instance = MagicMock()
            mock_registry.return_value = mock_instance

            def mock_resolve(model):
                if "gpt-5" in model and "mini" not in model and "nano" not in model:
                    return MagicMock(model_name="openai/gpt-5")
                elif "gpt-5-mini" in model:
                    return MagicMock(model_name="openai/gpt-5-mini")
                elif "gpt-5-nano" in model:
                    return MagicMock(model_name="openai/gpt-5-nano")
                return None

            mock_instance.resolve.side_effect = mock_resolve

            provider = OpenRouterProvider("test-api-key")
            provider._registry = mock_instance
            provider._parse_reasoning_config()

            # Should use map value (highest priority)
            assert provider._get_reasoning_effort("openai/gpt-5") == "high"

            # Should use default (no map entry)
            assert provider._get_reasoning_effort("openai/gpt-5-mini") == "medium"

            # Clear default to test fallback
            provider.default_reasoning_effort = ""
            assert provider._get_reasoning_effort("openai/gpt-5-nano") == "low"  # Falls back to thinking mode
