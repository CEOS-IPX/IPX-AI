"""
============================================================
환경 설정 (config.py)
============================================================
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ===== PostgreSQL =====
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "patent_db"
    postgres_user: str = "ipx_patent_user"
    postgres_password: str

    # ===== OpenSearch =====
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "patents"

    # ===== 외부 LLM API =====
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash-lite"

    # claude_api_key: str = ""
    # claude_model: str = "claude-sonnet-4-20250514"
    #
    # # ===== KIPRIS API =====
    # kipris_api_key: str = ""
    # kipris_base_url: str = ""


settings = Settings()