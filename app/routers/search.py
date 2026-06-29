"""
============================================================
검색 API 라우터
============================================================
Spring 서버로부터 검색 요청을 받아 검색 파이프라인을 실행한다.

검색 파이프라인 (현재 단계까지):
  1. LLM 의도 해석 (Gemini)
  2. 동의어 사전으로 키워드 확장
  3. HyDE 가상 초록 생성 (Claude)
  4. BGE-M3 임베딩
  5-1. OpenSearch 키워드 검색
  5-2. (다음) pgvector 벡터 검색
  5-3. (다음) KIPRIS API 호출
  6. (다음) RRF 병합 + 랭킹
  7. (다음) 추천 이유/요약 생성
============================================================
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.intent import interpret_intent, IntentResult
from app.services.synonym import synonym_expander
from app.services.hyde import generate_hypothetical_abstract
from app.services.embedding import embedding_service
from app.services.opensearch_client import opensearch_service
from app.services.types import PatentScore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])

# ============================================================
# 요청/응답 모델
# ============================================================

class SearchRequest(BaseModel):
    """Spring → Python 검색 요청"""

    query: str = Field(description="사용자 자연어 입력")
    result_count: int = Field(default=10, ge=1, le=50, description="결과 개수 (1~50)")
    legal_status: Optional[list[str]] = Field(
        default=None,
        description="기술 성숙도 필터. 예: ['공개', '등록']"
    )
    domain: Optional[str] = Field(
        default=None,
        description="도메인 (예: '이차전지', '반도체')"
    )


class SearchDebugInfo(BaseModel):
    """디버깅용 중간 결과 (운영 시 제거 가능)"""
    expanded_keywords: list[str]
    hypothetical_abstract: str
    embedding_dim: int       # 무조건 1024
    opensearch_results: list[PatentScore]


class SearchResponse(BaseModel):
    """Python → Spring 검색 응답"""

    is_valid: bool
    reason_invalid: Optional[str] = None
    intent: Optional[IntentResult] = None
    debug: Optional[SearchDebugInfo] = None
    # 이후 단계에서 추가될 필드:
    # results: list[PatentResult] = []


# ============================================================
# 검색 엔드포인트
# ============================================================

@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    검색 파이프라인 진입점
    """

    # === Step 1: LLM 의도 해석 ===
    logger.info(f"[검색 요청] query='{request.query}', domain={request.domain}, filters={request.legal_status}")

    try:
        intent = await interpret_intent(request.query, domain=request.domain)
    except ValueError as e:
        logger.warning(f"의도 해석 실패: {e}")
        raise HTTPException(status_code=502, detail="검색 의도 해석에 실패했습니다.")
    except Exception as e:
        logger.exception(f"의도 해석 중 예상치 못한 오류")
        raise HTTPException(status_code=500, detail="서버 내부 오류")

    # === 유효하지 않은 입력 처리 ===
    if not intent.is_valid:
        return SearchResponse(
            is_valid=False,
            reason_invalid=intent.reason_invalid,
            intent=intent
        )

    # === 의도 해석 성공 ===
    logger.info(f"[의도 해석 완료] keywords={intent.keywords}, ipc={intent.ipc_codes}")

    # ===== Step 2: 동의어 사전으로 키워드 확장 =====
    expanded_keywords = synonym_expander.expand(intent.keywords)
    logger.info(f"[동의어 확장] {len(intent.keywords)}개 → {len(expanded_keywords)}개")

    # ===== Step 3: HyDE 가상 초록 생성 =====
    try:
        hypothetical_abstract = await generate_hypothetical_abstract(
            keywords=expanded_keywords,
            ipc_codes=intent.ipc_codes
        )
    except Exception:
        logger.exception("HyDE 가상 초록 생성 실패")
        raise HTTPException(status_code=502, detail="가상 초록 생성 실패")

    # ===== Step 4: BGE-M3 임베딩 =====
    try:
        query_vector = embedding_service.embed(hypothetical_abstract)
    except Exception:
        logger.exception("임베딩 생성 실패")
        raise HTTPException(status_code=500, detail="임베딩 생성 실패")

    logger.info(f"[임베딩] dim={len(query_vector)}")

    # ===== Step 5-1: OpenSearch 키워드 검색 =====
    # 결과 개수는 RRF를 위해 사용자 요청보다 더 많이 가져옴
    candidate_size = max(request.result_count * 3, 30)

    opensearch_results = await opensearch_service.search(
        user_query=request.query,
        expanded_keywords=expanded_keywords,
        hypothetical_abstract=hypothetical_abstract,
        ipc_codes=intent.ipc_codes,
        legal_status=request.legal_status,
        size=candidate_size,
    )

    # 다음 단계
    # - 병렬 탐색 (pgvector, KIPRIS)
    # - RRF 병합 및 랭킹
    # - 추천/요약 생성 (Claude)

    return SearchResponse(
        is_valid=True,
        intent=intent,
        debug=SearchDebugInfo(
            expanded_keywords=expanded_keywords,
            hypothetical_abstract=hypothetical_abstract,
            embedding_dim=len(query_vector),
            opensearch_results=opensearch_results,
        )
    )