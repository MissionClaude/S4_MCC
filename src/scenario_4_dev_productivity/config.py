"""Configuration loader for Scenario 4 developer productivity agent.

Loads settings from environment variables with sensible defaults.
Uses python-dotenv to load .env file if present.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_project_root = Path(__file__).resolve().parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class Config:
    """Application configuration with env-var overrides."""

    # Anthropic API
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Model
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-haiku")

    # Rate limiting
    anthropic_max_rpm: int = int(os.getenv("ANTHROPIC_MAX_RPM", "50"))

    # Timeouts
    anthropic_timeout_seconds: int = int(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "120"))

    # Context
    max_conversation_turns: int = int(os.getenv("MAX_CONVERSATION_TURNS", "15"))
    scratchpad_path: str = os.getenv("SCRATCHPAD_PATH", ".scratchpad.md")

    # MCP
    mcp_config_path: str = os.getenv("MCP_CONFIG_PATH", ".mcp.json")

    # Pipeline
    pipeline_mode: bool = os.getenv("PIPELINE_MODE", "false").lower() == "true"

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> list[str]:
        """Check required configuration. Returns list of missing items."""
        missing: list[str] = []
        if not cls.anthropic_api_key or cls.anthropic_api_key.startswith("sk-ant-xxx"):
            missing.append(
                "ANTHROPIC_API_KEY is not set or is still the placeholder value. "
                "Get your key at https://console.anthropic.com/settings/keys"
            )
        return missing


# Singleton instance
config = Config()
