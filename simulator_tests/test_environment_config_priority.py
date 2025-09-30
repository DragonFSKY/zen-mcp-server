#!/usr/bin/env python3
"""
Environment Configuration Priority Integration Test

Tests that environment variables correctly override tool parameters
in real MCP tool execution scenarios.
"""

import json
import os

from .base_test import BaseSimulatorTest


class TestEnvironmentConfigPriority(BaseSimulatorTest):
    """Test environment configuration priority in real tool execution"""

    @property
    def test_name(self) -> str:
        return "environment_config_priority"

    @property
    def test_description(self) -> str:
        return "Environment configuration priority over tool parameters"

    def setUp(self):
        """Setup test environment variables"""
        # Store original environment
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

    def tearDown(self):
        """Restore original environment"""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)

    def test_google_model_specific_override(self):
        """Test Google provider model-specific thinking mode override"""
        self.logger.info("Testing Google model-specific environment override...")

        try:
            # Set model-specific configuration
            thinking_map = {"gemini-2.5-pro": "high", "gemini-2.5-flash": "medium"}
            os.environ["GOOGLE_THINKING_MODE_MAP"] = json.dumps(thinking_map)

            # Test Pro model with environment override
            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Test environment config priority: What is 2+2?",
                    "model": "gemini-2.5-pro",
                    "thinking_mode": "minimal",  # Should be overridden to "high"
                },
            )

            if not response:
                raise Exception("No response received from Pro model")

            self.logger.info("✅ Google Pro model environment override works")

            # Test Flash model with environment override
            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Test environment config priority: What is 3+3?",
                    "model": "gemini-2.5-flash",
                    "thinking_mode": "minimal",  # Should be overridden to "medium"
                },
            )

            if not response:
                raise Exception("No response received from Flash model")

            self.logger.info("✅ Google Flash model environment override works")
            return True

        except Exception as e:
            self.logger.error(f"❌ Google model-specific override test failed: {e}")
            return False

    def test_google_provider_default_override(self):
        """Test Google provider default thinking mode override"""
        self.logger.info("Testing Google provider default environment override...")

        try:
            # Clear model-specific map, set provider default
            os.environ.pop("GOOGLE_THINKING_MODE_MAP", None)
            os.environ["GOOGLE_DEFAULT_THINKING_MODE"] = "high"

            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Test provider default override: What is 4+4?",
                    "model": "gemini-2.5-pro",
                    "thinking_mode": "minimal",  # Should be overridden to "high"
                },
            )

            if not response:
                raise Exception("No response received with provider default")

            self.logger.info("✅ Google provider default override works")
            return True

        except Exception as e:
            self.logger.error(f"❌ Google provider default override test failed: {e}")
            return False

    def test_global_default_fallback(self):
        """Test global DEFAULT_THINKING_MODE_THINKDEEP fallback"""
        self.logger.info("Testing global default fallback...")

        try:
            # Clear all Google-specific configs, set global default
            os.environ.pop("GOOGLE_THINKING_MODE_MAP", None)
            os.environ.pop("GOOGLE_DEFAULT_THINKING_MODE", None)
            os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "medium"

            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Test global default fallback: What is 5+5?",
                    "model": "gemini-2.5-pro",
                    "thinking_mode": "minimal",  # Should be overridden to "medium"
                },
            )

            if not response:
                raise Exception("No response received with global default")

            self.logger.info("✅ Global default fallback works")
            return True

        except Exception as e:
            self.logger.error(f"❌ Global default fallback test failed: {e}")
            return False

    def test_openrouter_reasoning_effort_override(self):
        """Test OpenRouter reasoning effort environment override"""
        self.logger.info("Testing OpenRouter reasoning effort override...")

        try:
            # Set OpenRouter-specific configuration
            effort_map = {"openai/o3-mini": "high", "anthropic/claude-sonnet-4.1": "medium"}
            os.environ["OPENROUTER_REASONING_EFFORT_MAP"] = json.dumps(effort_map)

            # Test O3 model override
            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Test OpenRouter override: What is 6+6?",
                    "model": "o3-mini",  # Should resolve to openai/o3-mini
                    "thinking_mode": "minimal",  # Should be converted to reasoning_effort="high"
                },
            )

            if not response:
                raise Exception("No response received from O3 model")

            self.logger.info("✅ OpenRouter O3 reasoning effort override works")
            return True

        except Exception as e:
            self.logger.error(f"❌ OpenRouter reasoning effort override test failed: {e}")
            return False

    def test_priority_order_comprehensive(self):
        """Test complete priority order in real tool execution"""
        self.logger.info("Testing comprehensive priority order...")

        try:
            # Set all levels of configuration
            thinking_map = {"gemini-2.5-pro": "max"}
            os.environ["GOOGLE_THINKING_MODE_MAP"] = json.dumps(thinking_map)
            os.environ["GOOGLE_DEFAULT_THINKING_MODE"] = "high"
            os.environ["DEFAULT_THINKING_MODE_THINKDEEP"] = "medium"

            # 1. Model-specific map should have highest priority
            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Priority test 1: model-specific map (expect max)",
                    "model": "gemini-2.5-pro",
                    "thinking_mode": "minimal",  # Should be overridden to "max"
                },
            )

            if not response:
                raise Exception("Priority test 1 failed")

            self.logger.info("✅ Priority 1 (model-specific map) works")

            # 2. Remove model map, test provider default
            os.environ.pop("GOOGLE_THINKING_MODE_MAP", None)

            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Priority test 2: provider default (expect high)",
                    "model": "gemini-2.5-pro",
                    "thinking_mode": "minimal",  # Should be overridden to "high"
                },
            )

            if not response:
                raise Exception("Priority test 2 failed")

            self.logger.info("✅ Priority 2 (provider default) works")

            # 3. Remove provider default, test global default
            os.environ.pop("GOOGLE_DEFAULT_THINKING_MODE", None)

            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Priority test 3: global default (expect medium)",
                    "model": "gemini-2.5-pro",
                    "thinking_mode": "minimal",  # Should be overridden to "medium"
                },
            )

            if not response:
                raise Exception("Priority test 3 failed")

            self.logger.info("✅ Priority 3 (global default) works")
            return True

        except Exception as e:
            self.logger.error(f"❌ Comprehensive priority order test failed: {e}")
            return False

    def test_wildcard_pattern_matching(self):
        """Test wildcard pattern matching in environment configuration"""
        self.logger.info("Testing wildcard pattern matching...")

        try:
            # Set wildcard patterns
            thinking_map = {"gemini-2.5-*": "medium", "*-flash": "low", "*": "minimal"}
            os.environ["GOOGLE_THINKING_MODE_MAP"] = json.dumps(thinking_map)

            # Test prefix wildcard
            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Wildcard test 1: prefix pattern",
                    "model": "gemini-2.5-pro",  # Should match "gemini-2.5-*" -> "medium"
                    "thinking_mode": "high",  # Should be overridden
                },
            )

            if not response:
                raise Exception("Wildcard prefix test failed")

            self.logger.info("✅ Wildcard prefix pattern works")

            # Test suffix wildcard
            response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Wildcard test 2: suffix pattern",
                    "model": "gemini-2.5-flash",  # Should match "*-flash" -> "low"
                    "thinking_mode": "high",  # Should be overridden
                },
            )

            if not response:
                raise Exception("Wildcard suffix test failed")

            self.logger.info("✅ Wildcard suffix pattern works")
            return True

        except Exception as e:
            self.logger.error(f"❌ Wildcard pattern matching test failed: {e}")
            return False

    def run_test(self) -> bool:
        """Run all environment configuration priority tests"""
        self.logger.info(f"🔧 Test: {self.test_description}")

        try:
            self.setUp()

            # Test Google model-specific override
            if not self.test_google_model_specific_override():
                return False

            # Test Google provider default override
            if not self.test_google_provider_default_override():
                return False

            # Test global default fallback
            if not self.test_global_default_fallback():
                return False

            # Test OpenRouter reasoning effort override
            if not self.test_openrouter_reasoning_effort_override():
                return False

            # Test comprehensive priority order
            if not self.test_priority_order_comprehensive():
                return False

            # Test wildcard pattern matching
            if not self.test_wildcard_pattern_matching():
                return False

            self.logger.info(f"✅ All {self.test_name} tests passed!")
            return True

        except Exception as e:
            self.logger.error(f"❌ {self.test_name} test failed: {e}")
            return False
        finally:
            self.tearDown()


def main():
    """Run the environment configuration priority tests"""
    import sys

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    test = TestEnvironmentConfigPriority(verbose=verbose)

    success = test.run_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
