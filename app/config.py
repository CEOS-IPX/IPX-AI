# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "patent_db"
    postgres_user: str = "ipx_patent_user"
    postgres_password: str

    # OpenSearch
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200

    # 외부 API
    # gemini_api_key: str
    # claude_api_key: str
    # kipris_api_key: str


settings = Settings()