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

    # ===== Redis =====
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # ===== 외부 LLM API =====
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash-lite"

    claude_api_key: str = ""
    claude_model: str = "claude-haiku-4-5-20251001"  # 성능 안나오면 "claude-sonnet-4-6"

    claude_summary_model: str = "claude-haiku-4-5-20251001"

    claude_component_model: str = "claude-haiku-4-5-20251001"

    claude_novelty_model: str = "claude-haiku-4-5-20251001"

    # ===== KIPRIS API =====
    kipris_api_key: str
    kipris_base_url: str = "http://plus.kipris.or.kr/kipo-api/kipi"

    # ===== 동의어 사전 =====
    synonyms_file_path: str = "app/resources/synonyms_patent.txt"


settings = Settings()