"""
Configuration management for the Travel Agent application.
"""
import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration with validation."""
    
    # API Keys
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY", description="OpenAI API key")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY", description="Tavily API key")
    
    # Model Configuration
    travel_agent_model_name: str = Field(default="gpt-4.1", env="TRAVEL_AGENT_MODEL_NAME", description="OpenAI model name for the travel agent")
    memory_model_name: str = Field(default="gpt-4.1-mini", env="MEMORY_MODEL_NAME", description="OpenAI model name for memory operations")
    max_tool_iterations: int = Field(default=8, env="MAX_TOOL_ITERATIONS", description="Maximum tool iterations")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL", description="Redis connection URL")
    
    # Server Configuration
    server_name: str = Field(default="0.0.0.0", env="SERVER_NAME", description="Server host")
    server_port: int = Field(default=7860, env="SERVER_PORT", description="Server port")
    share: bool = Field(default=False, env="SHARE", description="Enable public sharing")

    # Azure APIM Gen-AI Gateway (optional)
    # When enabled, all OpenAI-compatible calls will be routed through the gateway.
    genai_gateway_enabled: bool = Field(default=False, env="GENAI_GATEWAY_ENABLED", description="Enable Azure APIM Gen-AI Gateway routing")
    genai_gateway_base_url: str | None = Field(default=None, env="GENAI_GATEWAY_BASE_URL", description="Base URL for the OpenAI-compatible gateway endpoint (e.g., https://<apim-name>.azure-api.net/v1)")
    genai_gateway_api_key: str | None = Field(default=None, env="GENAI_GATEWAY_API_KEY", description="Subscription/API key for the gateway (used instead of OPENAI_API_KEY when enabled)")
    genai_gateway_api_version: str | None = Field(default=None, env="GENAI_GATEWAY_API_VERSION", description="Optional API version query string required by the gateway, if any")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    # Per-field validation for OPENAI_API_KEY is skipped so that gateway-only configs can omit it.

    @model_validator(mode="after")
    def validate_gateway_vs_openai(self) -> "AppConfig":
        """Cross-field validation for gateway configuration vs direct OpenAI usage."""
        if self.genai_gateway_enabled:
            if not self.genai_gateway_base_url:
                raise ValueError("GENAI_GATEWAY_ENABLED is true but GENAI_GATEWAY_BASE_URL is not set")
            if not self.genai_gateway_api_key:
                raise ValueError("GENAI_GATEWAY_ENABLED is true but GENAI_GATEWAY_API_KEY is not set")
        else:
            # When not using the gateway, enforce typical OpenAI key format.
            if not self.openai_api_key or not isinstance(self.openai_api_key, str):
                raise ValueError("OPENAI_API_KEY is required when GENAI_GATEWAY_ENABLED=false")
            if not self.openai_api_key.startswith("sk-"):
                raise ValueError("OpenAI API key must start with 'sk-' when GENAI_GATEWAY_ENABLED=false")
        return self
    



def get_config() -> AppConfig:
    """Get application configuration with proper error handling."""
    try:
        return AppConfig()
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📝 Please check your environment variables or create a .env file with:")
        print("OPENAI_API_KEY=sk-your-key-here")
        print("TAVILY_API_KEY=your-key-here")
        raise SystemExit(1)


def validate_dependencies() -> bool:
    """Validate that required services are available."""
    from openai import OpenAI
    
    config = get_config()
    
    # Test OpenAI API
    try:
        # Route through gateway if enabled (OpenAI-compatible endpoint)
        if config.genai_gateway_enabled:
            base_url = config.genai_gateway_base_url
            # Ensure no trailing slash inconsistencies
            if base_url and not base_url.rstrip().endswith("/v1") and "/v1" not in base_url:
                print("ℹ️ Note: Expected an OpenAI-compatible base URL that includes '/v1'. Current:", base_url)
            client = OpenAI(api_key=config.genai_gateway_api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=config.openai_api_key)
        # Just test the client creation, not making an actual API call
        print("✅ OpenAI client configured")
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False
    
    print("✅ All dependencies validated")
    return True
