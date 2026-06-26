from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    log_level: str = "INFO"
    environment: str = "production"
