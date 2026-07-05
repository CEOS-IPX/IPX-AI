"""
============================================================
RRF (Reciprocal Rank Fusion) 병합 서비스
============================================================
여러 검색 소스의 결과를 하나의 통합 순위로 병합한다.

RRF_score(d) = Σ  w_i × 1 / (k + rank_i(d))
               i
  - d: 특정 출원번호
  - i: 검색 소스 (opensearch, pgvector)
  - rank_i(d): i번째 소스에서 d의 순위 (1부터 시작)
  - k: 60 (표준 상수, 논문에서 검증된 값)
  - w_i: 소스별 가중치 (현재는 모두 1.0 동등)

특징:
  - 각 특허의 RRF 점수는 다른 특허와 무관하게 독립 계산
  - 각 소스의 원본 점수를 무시하고 순위만 사용
  - 서로 다른 점수 체계(BM25, 코사인)의 비교 불가능 문제 해결
============================================================
"""

import logging
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field

from app.services.types import PatentScore, SourceType

logger = logging.getLogger(__name__)


# ============================================================
# 상수 (외부 사용을 위해 노출)
# ============================================================

# RRF 표준 상수
RRF_K = 60

# 소스별 가중치 (MVP: 동등, 추후 튜닝 가능)
SOURCE_WEIGHTS: dict[SourceType, float] = {
    "opensearch": 1.0,
    "pgvector": 1.0,
}


# ============================================================
# 병합 결과 모델
# ============================================================

class MergedPatent(BaseModel):
    """RRF 병합 후 단일 특허 결과"""

    application_number: str = Field(description="출원번호")
    rrf_score: float = Field(description="RRF 병합 점수")
    rank: int = Field(description="병합 후 최종 순위 (1부터)")
    sources: list[SourceType] = Field(
        description="이 특허가 발견된 검색 소스 (예: ['opensearch', 'pgvector'])"
    )
    source_ranks: dict[str, int] = Field(
        default_factory=dict,
        description="소스별 원본 순위 (디버깅용)"
    )


# ============================================================
# RRF 병합
# ============================================================

def merge_with_rrf(
    opensearch_results: list[PatentScore],
    pgvector_results: list[PatentScore],
    top_n: int,
    k: int = RRF_K,
    weights: Optional[dict[SourceType, float]] = None,
) -> list[MergedPatent]:
    """
    여러 검색 소스의 결과를 RRF로 병합.

    Args:
        opensearch_results: OpenSearch 검색 결과 (rank 오름차순)
        pgvector_results: pgvector 검색 결과 (rank 오름차순)
        top_n: 반환할 최종 결과 개수
        k: RRF 상수 (기본 60)
        weights: 소스별 가중치 (기본: 모두 1.0)

    Returns:
        RRF 점수 내림차순 MergedPatent 리스트 (최대 top_n개)
    """
    if weights is None:
        weights = SOURCE_WEIGHTS

    rrf_scores: dict[str, float] = defaultdict(float)
    sources_by_patent: dict[str, list[SourceType]] = defaultdict(list)
    ranks_by_patent: dict[str, dict[str, int]] = defaultdict(dict)

    _accumulate(
        results=opensearch_results,
        source="opensearch",
        weight=weights.get("opensearch", 1.0),
        k=k,
        rrf_scores=rrf_scores,
        sources_by_patent=sources_by_patent,
        ranks_by_patent=ranks_by_patent,
    )

    _accumulate(
        results=pgvector_results,
        source="pgvector",
        weight=weights.get("pgvector", 1.0),
        k=k,
        rrf_scores=rrf_scores,
        sources_by_patent=sources_by_patent,
        ranks_by_patent=ranks_by_patent,
    )

    sorted_patents = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    merged: list[MergedPatent] = []
    for new_rank, (app_num, score) in enumerate(sorted_patents[:top_n], start=1):
        merged.append(MergedPatent(
            application_number=app_num,
            rrf_score=score,
            rank=new_rank,
            sources=sources_by_patent[app_num],
            source_ranks=ranks_by_patent[app_num],
        ))

    logger.info(
        f"[RRF 병합] OpenSearch {len(opensearch_results)}건 + "
        f"pgvector {len(pgvector_results)}건 → "
        f"고유 {len(rrf_scores)}건 → 상위 {len(merged)}건 반환"
    )

    return merged


# ============================================================
# 내부 헬퍼
# ============================================================

def _accumulate(
    results: list[PatentScore],
    source: SourceType,
    weight: float,
    k: int,
    rrf_scores: dict[str, float],
    sources_by_patent: dict[str, list[SourceType]],
    ranks_by_patent: dict[str, dict[str, int]],
) -> None:
    """한 검색 소스의 결과를 누적 점수에 합산"""
    for result in results:
        app_num = result.application_number
        contribution = weight * (1.0 / (k + result.rank))

        rrf_scores[app_num] += contribution
        sources_by_patent[app_num].append(source)
        ranks_by_patent[app_num][source] = result.rank