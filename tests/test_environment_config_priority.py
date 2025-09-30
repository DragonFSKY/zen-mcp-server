"""
Tests for environment configuration priority functionality.

Tests that environment variables correctly override tool parameters
for thinking_mode/reasoning_effort configuration.
"""

import json
import os

import pytest

from providers.gemini import GeminiModelProvider
from providers.openai_provider import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider


class TestEnvironmentConfigPriority:
    """Test environment configuration priority over tool parameters."""

    def setup_method(self):
        """Setup test environment."""
        # Store original environment values
        self.original_env = {}
        env_vars = [
            "GOOGLE_THINKING_MODE_MAP",
            "GOOGLE_DEFAULT_THINKING_MODE",
            "OPENAI_THINKING_MODE_MAP",
            "OPENAI_DEFAULT_THINKING_MODE",
            "OPENROUTER_REASONING_EFFORT_MAP",
            "OPENROUTER_DEFAULT_REASONING_EFFORT",
            "DEFAULT_THINKING_MODE_THINKDEEP",
        ]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
            # Clear existing values
            os.environ.pop(var, None)

    def teardown_method(self):
        """Restore original environment."""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)

    def test_google_provider_model_specific_map(self):
        """Test Google provider model-specific thinking mode map."""
        # Set environment variable
        thinking_map = {"gemini-2.5-pro": "high", "gemini-2.5-flash": "medium", "gemini-*": "low"}
        os.environ["GOOGLE_THINKING_MODE_MAP"] = json.dumps(thinking_map)

        provider = GeminiModelProvider(api_key="test-key")

        # Test exact model match
        kwargs = {"thinking_mode": "minimal"}  # Tool parameter should be overridden
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "high"
        assert "thinking_mode" in result  # Original parameter should be replaced

        # Test wildcard match
        kwargs = {"thinking_mode": "minimal"}
        result = provider.process_thinking_parameters("gemini-unknown", **kwargs)
        assert result["thinking_mode"] == "low"

        # Test no environment override (should keep tool parameter)
        provider2 = GeminiModelProvider(api_key="test-key")
        os.environ.pop("GOOGLE_THINKING_MODE_MAP", None)
        kwargs = {"thinking_mode": "minimal"}
        result = provider2.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "minimal"

    def test_google_provider_default_fallback(self):
        """Test Google provider default thinking mode fallback."""
        os.environ["GOOGLE_DEFAULT_THINKING_MODE"] = "medium"

        provider = GeminiModelProvider(api_key="test-key")

        # No specific map, should use provider default
        kwargs = {"thinking_mode": "low"}  # Should be overridden
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "medium"

    def test_openai_provider_model_specific_map(self):
        """Test OpenAI provider model-specific thinking mode map."""
        thinking_map = {"o3-mini": "high", "o3-pro": "max", "gpt-*": "medium"}
        os.environ["OPENAI_THINKING_MODE_MAP"] = json.dumps(thinking_map)

        provider = OpenAIModelProvider(api_key="test-key")

        # Test exact match
        kwargs = {"thinking_mode": "low"}
        result = provider.process_thinking_parameters("o3-mini", **kwargs)
        assert result["thinking_mode"] == "high"

        # Test wildcard match
        kwargs = {"thinking_mode": "low"}
        result = provider.process_thinking_parameters("gpt-5", **kwargs)
        assert result["thinking_mode"] == "medium"

    def test_openrouter_reasoning_effort_conversion(self):
        """Test OpenRouter reasoning effort conversion with environment override."""
        effort_map = {"openai/o3-mini": "high", "openai/gpt-5": "medium", "anthropic/*": "low"}
        os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = json.dumps(effort_map)

        provider = OpenRouterProvider(api_key="test-key")

        # Test O3 model conversion
        kwargs = {"thinking_mode": "minimal"}  # Should be overridden
        result = provider.process_thinking_parameters("openai/o3-mini", **kwargs)
        assert "reasoning_effort" in result
        assert result["reasoning_effort"] == "high"
        assert "thinking_mode" not in result

        # Test GPT-5 model conversion
        kwargs = {"thinking_mode": "minimal"}
        result = provider.process_thinking_parameters("openai/gpt-5", **kwargs)
        assert "reasoning" in result
        assert result["reasoning"]["effort"] == "medium"

    def test_global_default_fallback(self):
        """Test global DEFAULT_THINKING_MODE_THINKDEEP fallback."""
        os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "high"

        # Test with Google provider (should fall back to global default)
        provider = GeminiModelProvider(api_key="test-key")
        kwargs = {"thinking_mode": "low"}
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "high"

        # Test with OpenAI provider
        provider2 = OpenAIModelProvider(api_key="test-key")
        kwargs = {"thinking_mode": "low"}
        result = provider2.process_thinking_parameters("o3-mini", **kwargs)
        assert result["thinking_mode"] == "high"

    def test_openrouter_no_global_fallback(self):
        """Test that OpenRouter doesn't use global DEFAULT_THINKING_MODE_THINKDEEP fallback."""
        os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "high"

        provider = OpenRouterProvider(api_key="test-key")
        kwargs = {"thinking_mode": "low"}
        result = provider.process_thinking_parameters("openai/o3-mini", **kwargs)

        # Should convert thinking_mode to reasoning_effort, not use global default
        assert "reasoning_effort" in result
        assert result["reasoning_effort"] == "low"  # Converted from tool parameter
        assert "thinking_mode" not in result

    def test_priority_order(self):
        """Test complete priority order: env map > env default > global default > tool param."""
        # Set all levels
        thinking_map = {"gemini-2.5-pro": "max"}
        os.environ["GOOGLE_THINKING_MODE_MAP"] = json.dumps(thinking_map)
        os.environ["GOOGLE_DEFAULT_THINKING_MODE"] = "high"
        os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "medium"

        provider = GeminiModelProvider(api_key="test-key")

        # 1. Model-specific map should win
        kwargs = {"thinking_mode": "minimal"}
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "max"

        # 2. Remove map, should use provider default
        os.environ.pop("GOOGLE_THINKING_MODE_MAP", None)
        kwargs = {"thinking_mode": "minimal"}
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "high"

        # 3. Remove provider default, should use global default
        os.environ.pop("GOOGLE_DEFAULT_THINKING_MODE", None)
        kwargs = {"thinking_mode": "minimal"}
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "medium"

        # 4. Remove global default, should use tool parameter
        os.environ.pop("DEFAULT_THINKING_MODE_THINKDEEP", None)
        kwargs = {"thinking_mode": "minimal"}
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "minimal"

    def test_wildcard_pattern_matching(self):
        """Test wildcard pattern matching in environment maps."""
        thinking_map = {"exact-model": "high", "prefix-*": "medium", "*-suffix": "low", "*": "minimal"}
        os.environ["GOOGLE_THINKING_MODE_MAP"] = json.dumps(thinking_map)

        provider = GeminiModelProvider(api_key="test-key")

        # Test exact match
        kwargs = {"thinking_mode": "default"}
        result = provider.process_thinking_parameters("exact-model", **kwargs)
        assert result["thinking_mode"] == "high"

        # Test prefix match
        kwargs = {"thinking_mode": "default"}
        result = provider.process_thinking_parameters("prefix-test", **kwargs)
        assert result["thinking_mode"] == "medium"

        # Test suffix match
        kwargs = {"thinking_mode": "default"}
        result = provider.process_thinking_parameters("test-suffix", **kwargs)
        assert result["thinking_mode"] == "low"

        # Test catch-all match
        kwargs = {"thinking_mode": "default"}
        result = provider.process_thinking_parameters("random-model", **kwargs)
        assert result["thinking_mode"] == "minimal"

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON in environment variables."""
        # Set invalid JSON
        os.environ["GOOGLE_THINKING_MODE_MAP"] = "invalid-json"

        provider = GeminiModelProvider(api_key="test-key")

        # Should skip invalid JSON and fall back to tool parameter
        kwargs = {"thinking_mode": "medium"}
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "medium"

    def test_no_environment_config(self):
        """Test behavior when no environment configuration is set."""
        provider = GeminiModelProvider(api_key="test-key")

        # Should preserve tool parameter unchanged
        kwargs = {"thinking_mode": "high"}
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert result["thinking_mode"] == "high"

        # Should work with no thinking_mode parameter
        kwargs = {}
        result = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
        assert "thinking_mode" not in result


if __name__ == "__main__":
    pytest.main([__file__])
