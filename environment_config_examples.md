# Environment Configuration Priority Examples

The environment configuration priority feature allows you to override the `thinking_mode` parameter in tools via environment variables, providing more flexible configuration management.

## Environment Variable Priority Order

1. **Model-specific mapping** (Highest priority)
2. **Provider defaults**
3. **Global defaults**
4. **Tool parameters** (Lowest priority)

## Supported Environment Variables

### Google/Gemini Provider
```bash
# Model-specific mapping (JSON format)
export GOOGLE_THINKING_MODE_MAP='{"gemini-2.5-pro": "high", "gemini-2.5-flash": "medium", "gemini-*": "low"}'

# Provider default
export GOOGLE_DEFAULT_THINKING_MODE="medium"
```

### OpenAI Provider
```bash
# Model-specific mapping
export OPENAI_THINKING_MODE_MAP='{"o3-mini": "high", "o3-pro": "max", "gpt-*": "medium"}'

# Provider default
export OPENAI_DEFAULT_THINKING_MODE="high"
```

### OpenRouter Provider
```bash
# OpenRouter uses reasoning_effort instead of thinking_mode
export OPENROUTER_REASONING_EFFORT_MAP='{"openai/o3-mini": "high", "anthropic/*": "medium"}'

# Provider default
export OPENROUTER_DEFAULT_REASONING_EFFORT="medium"
```

### Global Defaults (Backward Compatible)
```bash
# Applies to all supported providers (except OpenRouter)
export DEFAULT_THINKING_MODE_THINKDEEP="medium"
```

## Wildcard Pattern Support

Environment variable mappings support wildcard patterns:

```bash
# Prefix matching
export GOOGLE_THINKING_MODE_MAP='{"gemini-2.5-*": "medium"}'

# Suffix matching
export GOOGLE_THINKING_MODE_MAP='{"*-flash": "low"}'

# Match all
export GOOGLE_THINKING_MODE_MAP='{"*": "minimal"}'

# Mixed patterns (by matching priority)
export GOOGLE_THINKING_MODE_MAP='{
  "gemini-2.5-pro": "high",
  "gemini-2.5-*": "medium",
  "*-flash": "low",
  "*": "minimal"
}'
```

## Practical Usage Examples

### 1. Production Environment Configuration
```bash
# .env file
GOOGLE_DEFAULT_THINKING_MODE="medium"
OPENAI_DEFAULT_THINKING_MODE="high"
OPENROUTER_DEFAULT_REASONING_EFFORT="medium"

# High-performance models use high thinking mode
GOOGLE_THINKING_MODE_MAP='{"gemini-2.5-pro": "high"}'
OPENAI_THINKING_MODE_MAP='{"o3-pro": "max", "o3-mini": "high"}'
```

### 2. Development Environment Configuration
```bash
# Use lower thinking mode during development to save time
export GOOGLE_DEFAULT_THINKING_MODE="low"
export OPENAI_DEFAULT_THINKING_MODE="medium"
export DEFAULT_THINKING_MODE_THINKDEEP="low"
```

### 3. Specific Use Case Configuration
```bash
# Code review scenario - requires deep thinking
export GOOGLE_THINKING_MODE_MAP='{"*": "high"}'

# Quick response scenario - minimal thinking
export GOOGLE_THINKING_MODE_MAP='{"*": "minimal"}'
```

## Verify Configuration

Use the following commands to verify that environment configuration is active:

```bash
# View current environment variables
echo $GOOGLE_THINKING_MODE_MAP
echo $GOOGLE_DEFAULT_THINKING_MODE
echo $DEFAULT_THINKING_MODE_THINKDEEP

# Test configuration priority
python -c "
import os
import json
from providers.gemini import GeminiModelProvider

# Set test configuration
os.environ['GOOGLE_THINKING_MODE_MAP'] = json.dumps({'gemini-2.5-pro': 'high'})

provider = GeminiModelProvider(api_key='test')
result = provider.process_thinking_parameters('gemini-2.5-pro', thinking_mode='minimal')
print(f'Result: {result}')
# Should output: {'thinking_mode': 'high'}
"
```
