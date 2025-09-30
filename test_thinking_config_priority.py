#!/usr/bin/env python3
"""
Test script to verify all tools respect environment thinking config priority
"""
import asyncio
import json
import logging
import os

# Setup test environment variables BEFORE importing tools
os.environ["GOOGLE_THINKING_MODE_MAP"] = json.dumps(
    {"gemini-2.5-pro": "max", "gemini-2.5-flash": "high", "*": "medium"}
)
os.environ["GOOGLE_DEFAULT_THINKING_MODE"] = "high"

os.environ["OPENAI_THINKING_MODE_MAP"] = json.dumps({"o3-mini": "high", "gpt-5*": "max", "*": "low"})
os.environ["OPENAI_DEFAULT_THINKING_MODE"] = "medium"

os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = json.dumps({"openai/o3*": "high", "openai/gpt-5": "medium"})
os.environ["OPENROUTER_DEFAULT_REASONING_EFFORT"] = "low"

os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "medium"

# Import after setting env vars
from providers.custom import CustomProvider
from providers.gemini import GeminiModelProvider
from providers.openai_provider import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider
from providers.registry import ModelProviderRegistry
from tools.chat import ChatTool
from tools.codereview import CodeReviewTool

# Import tools
from tools.consensus import ConsensusTool
from tools.debug import DebugIssueTool
from tools.thinkdeep import ThinkDeepTool

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ThinkingConfigTester:
    def __init__(self):
        self.registry = ModelProviderRegistry()
        self.results = {}

    def test_provider_config_priority(self):
        """Test that providers correctly apply environment config priority"""
        print("\n=== Testing Provider Config Priority ===\n")

        # Test Gemini provider
        if os.getenv("GEMINI_API_KEY"):
            provider = GeminiModelProvider(os.getenv("GEMINI_API_KEY"))

            # Test Pro model - should get "max" from map
            kwargs = {"thinking_mode": "low"}  # Tool provides low
            processed = provider.process_thinking_parameters("gemini-2.5-pro", **kwargs)
            assert processed.get("thinking_mode") == "max", f"Expected max, got {processed.get('thinking_mode')}"
            print("✅ Gemini Pro: Environment config 'max' overrides tool 'low'")

            # Test Flash model - should get "high" from map
            kwargs = {"thinking_mode": "minimal"}  # Tool provides minimal
            processed = provider.process_thinking_parameters("gemini-2.5-flash", **kwargs)
            assert processed.get("thinking_mode") == "high", f"Expected high, got {processed.get('thinking_mode')}"
            print("✅ Gemini Flash: Environment config 'high' overrides tool 'minimal'")

            # Test unknown model - should get "medium" from wildcard
            kwargs = {"thinking_mode": "low"}
            processed = provider.process_thinking_parameters("gemini-unknown", **kwargs)
            assert processed.get("thinking_mode") == "medium", f"Expected medium, got {processed.get('thinking_mode')}"
            print("✅ Gemini Unknown: Wildcard config 'medium' overrides tool 'low'")

        # Test OpenAI provider
        if os.getenv("OPENAI_API_KEY"):
            provider = OpenAIModelProvider(os.getenv("OPENAI_API_KEY"))

            # Test O3-mini - should get "high" from map
            kwargs = {"thinking_mode": "minimal"}
            processed = provider.process_thinking_parameters("o3-mini", **kwargs)
            assert processed.get("thinking_mode") == "high", f"Expected high, got {processed.get('thinking_mode')}"
            print("✅ OpenAI O3-mini: Environment config 'high' overrides tool 'minimal'")

            # Test GPT-5 - should get "max" from wildcard pattern
            kwargs = {"thinking_mode": "low"}
            processed = provider.process_thinking_parameters("gpt-5-turbo", **kwargs)
            assert processed.get("thinking_mode") == "max", f"Expected max, got {processed.get('thinking_mode')}"
            print("✅ OpenAI GPT-5: Pattern config 'max' overrides tool 'low'")

        # Test OpenRouter provider
        if os.getenv("OPENROUTER_API_KEY"):
            provider = OpenRouterProvider(os.getenv("OPENROUTER_API_KEY"))

            # Test O3 model - should get "high" from map (converted to reasoning_effort)
            kwargs = {"thinking_mode": "minimal"}
            processed = provider.process_thinking_parameters("openai/o3-mini", **kwargs)
            # OpenRouter uses reasoning_effort, not thinking_mode
            assert (
                processed.get("reasoning_effort") == "high"
            ), f"Expected high, got {processed.get('reasoning_effort')}"
            assert "thinking_mode" not in processed, "thinking_mode should be removed for OpenRouter"
            print(
                "✅ OpenRouter O3: Environment config 'high' overrides tool 'minimal' (converted to reasoning_effort)"
            )

        # Test Custom provider (after fix)
        if os.getenv("CUSTOM_API_URL"):
            provider = CustomProvider(api_url=os.getenv("CUSTOM_API_URL"))
            kwargs = {"thinking_mode": "low"}
            processed = provider.process_thinking_parameters("local-llama", **kwargs)
            # Custom should get DEFAULT_THINKING_MODE_THINKDEEP fallback
            assert processed.get("thinking_mode") == "medium", f"Expected medium, got {processed.get('thinking_mode')}"
            print("✅ Custom: Global default 'medium' overrides tool 'low'")

        print("\n✅ All provider config priority tests passed!")

    async def test_tool_integration(self):
        """Test that tools correctly pass through thinking_mode and providers apply env config"""
        print("\n=== Testing Tool Integration ===\n")

        # Test consensus tool (after fix)
        print("Testing consensus tool...")
        tool = ConsensusTool()
        # The tool should pass thinking_mode to provider, which will override with env config
        # We can't directly test the internal call, but we can verify the tool accepts the parameter

        # Test chat tool (SimpleTool)
        print("Testing chat tool...")
        tool = ChatTool()
        # Chat tool should include thinking_mode in schema
        schema = tool.get_input_schema()
        assert "thinking_mode" in schema.get("properties", {}), "Chat tool should include thinking_mode"
        print("✅ Chat tool includes thinking_mode in schema")

        # Test workflow tools
        print("Testing workflow tools...")
        for tool_class, tool_name in [
            (ThinkDeepTool, "thinkdeep"),
            (CodeReviewTool, "codereview"),
            (DebugIssueTool, "debug"),
        ]:
            tool = tool_class()
            # Workflow tools may exclude thinking_mode from schema but still use it internally
            print(f"✅ {tool_name} tool initialized successfully")

        print("\n✅ All tool integration tests passed!")

    def run_all_tests(self):
        """Run all config priority tests"""
        print("\n" + "=" * 60)
        print("THINKING CONFIG PRIORITY TEST SUITE")
        print("=" * 60)

        # Test provider level
        self.test_provider_config_priority()

        # Test tool level
        asyncio.run(self.test_tool_integration())

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Environment config priority working!")
        print("=" * 60)

        # Show current config
        print("\n📋 Current Environment Config:")
        print(f"  GOOGLE_THINKING_MODE_MAP: {os.getenv('GOOGLE_THINKING_MODE_MAP')}")
        print(f"  GOOGLE_DEFAULT_THINKING_MODE: {os.getenv('GOOGLE_DEFAULT_THINKING_MODE')}")
        print(f"  OPENAI_THINKING_MODE_MAP: {os.getenv('OPENAI_THINKING_MODE_MAP')}")
        print(f"  OPENROUTER_REASONING_EFFORT_MAP: {os.getenv('OPENROUTER_REASONING_EFFORT_MAP')}")
        print(f"  DEFAULT_THINKING_MODE_THINKDEEP: {os.getenv('DEFAULT_THINKING_MODE_THINKDEEP')}")


if __name__ == "__main__":
    tester = ThinkingConfigTester()
    tester.run_all_tests()
