"""
============================================================
검색 API 라우터
============================================================
Spring 서버로부터 검색 요청을 받아 검색 파이프라인을 실행한다.

엔드포인트:
  POST /search
============================================================
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.intent import interpret_intent, IntentResult

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


class SearchResponse(BaseModel):
    """Python → Spring 검색 응답"""

    is_valid: bool
    reason_invalid: Optional[str] = None
    intent: Optional[IntentResult] = None
    # 이후 단계에서 추가될 필드:
    # results: list[PatentResult] = []


# ============================================================
# 검색 엔드포인트
# ============================================================

@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    검색 파이프라인 진입점

    1. LLM 의도 해석
    2. (다음 단계) 가상 초록 생성 + HyDE 임베딩
    3. (다음 단계) OpenSearch + pgvector + KIPRIS 병렬 탐색
    4. (다음 단계) RRF 병합 및 랭킹
    5. (다음 단계) LLM 추천/요약 생성
    """

    # === Step 1: LLM 의도 해석 ===
    logger.info(f"[검색 요청] query='{request.query}', filters={request.legal_status}")

    try:
        intent = await interpret_intent(request.query)
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

    # 다음 단계
    # - HyDE 가상 초록 생성 (Claude)
    # - BGE-M3 임베딩
    # - 병렬 탐색 (OpenSearch, pgvector, KIPRIS)
    # - RRF 병합 및 랭킹
    # - 추천/요약 생성 (Claude)

    return SearchResponse(
        is_valid=True,
        intent=intent
    )