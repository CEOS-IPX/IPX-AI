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

    claude_inventive_step_model: str = "claude-haiku-4-5-20251001"

    # ===== KIPRIS API =====
    kipris_api_key: str

    # 상세 조회용 (기존 kipris_base_url이 이 값이었음)
    kipris_kipo_base_url: str = "http://plus.kipris.or.kr/kipo-api/kipi"

    # 검색 API용 (신규)
    kipris_openapi_base_url: str = "http://plus.kipris.or.kr/openapi/rest"

    # ===== 동의어 사전 =====
    synonyms_file_path: str = "app/resources/synonyms_patent.txt"

    # ===== 증분 배치 관련 =====
    # MPV 도메인 IPC prefix (G06N, G06T)
    target_ipc_prefixes: list[str] = ["G06N", "G06T"]

    # 배치 실행 요일/시간 (KST) : 매주 월요일 04:00
    ingestion_cron_day_of_week: str = "mon"
    ingestion_cron_hour: int = 4
    ingestion_cron_minute: int = 0

    # 배치 조회 기간 (일)
    ingestion_lookback_days: int = 7

    # KIPRIS API rate limiting (초당 호출 수)
    kipris_rate_limit_per_second: int = 5

    # 실패 재시도 최대 횟수 (이후 영구 제거)
    ingestion_max_retry: int = 3

    # 실패 로그 파일 경로
    ingestion_failure_log_path: str = "/data/failed_ingestions.txt"

settings = Settings()