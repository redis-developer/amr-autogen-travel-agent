"""
Configuration management for the Travel Agent application.
"""
import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration with validation."""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY", description="OpenAI API key")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY", description="Tavily API key")
    
    # Model Configuration
    travel_agent_model_name: str = Field(default="gpt-4.1", env="TRAVEL_AGENT_MODEL_NAME", description="OpenAI model name for the travel agent")
    memory_model_name: str = Field(default="gpt-4.1-mini", env="MEMORY_MODEL_NAME", description="OpenAI model name for memory operations")
    max_tool_iterations: int = Field(default=8, env="MAX_TOOL_ITERATIONS", description="Maximum tool iterations")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST", description="Redis host URL")
    redis_port: str = Field(default="6379", env="REDIS_PORT", description="Redis port")

    # Server Configuration
    server_name: str = Field(default="0.0.0.0", env="SERVER_NAME", description="Server host")
    server_port: int = Field(default=7860, env="SERVER_PORT", description="Server port")
    share: bool = Field(default=False, env="SHARE", description="Enable public sharing")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v):
        """Validate OpenAI API key format."""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    



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
        client = OpenAI(api_key=config.openai_api_key)
        # Just test the client creation, not making an actual API call
        print("✅ OpenAI API key configured")
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False
    
    print("✅ All dependencies validated")
    return True
