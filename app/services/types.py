"""
============================================================
검색 파이프라인 공통 타입
: 세 소스(OpenSearch, pgvector, KIPRIS)의 동일한 반환 결과
============================================================
"""

from typing import Literal
from pydantic import BaseModel, Field


SourceType = Literal["opensearch", "pgvector", "kipris"]


class PatentScore(BaseModel):
    """검색 결과 1건의 점수 정보 (RRF 병합 전)"""

    application_number: str = Field(description="특허 출원번호 (공통 식별자)")
    score: float = Field(description="해당 소스의 원본 점수")
    source: SourceType = Field(description="결과를 만든 소스")
    rank: int = Field(description="해당 소스 내 순위 (1부터 시작)")