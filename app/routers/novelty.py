"""
============================================================
신규성 분석 API 라우터
============================================================
Spring이 사건 정보 + 구성요소 + 상위 3건 선행기술을 전달하면
Python이 LLM으로 신규성 분석을 수행한다.

Spring 흐름:
  1. 변리사 UI에서 "신규성 분석" 버튼 클릭
  2. Spring: cases, invention_components 조회
  3. Spring: prior_arts에서 rrf_score 내림차순 상위 3건 조회
     (각 특허의 청구항 원문은 OpenSearch에서 조회)
  4. Spring: Python /analyze/novelty 호출
  5. Python: LLM 3회 병렬 호출 → 가장 유사한 1건 반환
  6. Spring: novelty_analyses + novelty_comparisons 테이블 저장
  7. Spring: 프론트에 응답
============================================================
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.novelty_analyzer import (
    analyze_novelty,
    InventionComponent,
    PriorArtForAnalysis,
    ComponentComparison,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])


# ============================================================
# 요청 모델
# ============================================================

class NoveltyAnalyzeRequest(BaseModel):
    """Spring → Python 신규성 분석 요청"""
    invention_title: str = Field(description="사용자 발명 명칭")
    invention_description: str = Field(description="사용자 발명 설명")
    components: list[InventionComponent] = Field(
        description="사용자 발명 구성요소 (label 포함, 3~7개)"
    )
    prior_arts: list[PriorArtForAnalysis] = Field(
        description="분석 대상 선행기술 (상위 3건 권장, 각 특허의 청구항 원문 포함)"
    )


# ============================================================
# 응답 모델
# ============================================================

class NoveltyAnalyzeResponse(BaseModel):
    """Python → Spring 신규성 분석 결과 (가장 유사한 1건)"""
    d1_application_number: str = Field(description="주인용발명 D1의 출원번호")
    overall_similarity: str = Field(description="'매우 높음' | '높음' | '보통' | '낮음'")
    conclusion_text: str = Field(description="신규성 판단 결론 문구")
    component_results: list[ComponentComparison] = Field(
        description="구성요소별 대비 결과"
    )


# ============================================================
# 엔드포인트
# ============================================================

@router.post("/novelty", response_model=NoveltyAnalyzeResponse)
async def analyze_novelty_endpoint(
    request: NoveltyAnalyzeRequest,
) -> NoveltyAnalyzeResponse:
    """
    사용자 발명의 신규성 분석.

    상위 3건 선행기술과 병렬 비교 후 가장 유사한 1건을 D1으로 선정하여 반환.

    Raises:
        400: 필수 데이터 부족
        502: 모든 LLM 분석 실패
    """
    # 유효성 검증
    if not request.invention_title.strip() or not request.invention_description.strip():
        raise HTTPException(
            status_code=400,
            detail="발명 명칭과 설명이 필요합니다.",
        )
    if not request.components:
        raise HTTPException(
            status_code=400,
            detail="분석할 구성요소가 없습니다. 먼저 구성요소를 등록해주세요.",
        )
    if not request.prior_arts:
        raise HTTPException(
            status_code=400,
            detail="분석할 선행기술이 없습니다. 먼저 선행기술 탐색을 수행해주세요.",
        )

    logger.info(
        f"[Novelty API] 요청: components={len(request.components)}개, "
        f"prior_arts={len(request.prior_arts)}건"
    )

    # 분석 수행
    try:
        result = await analyze_novelty(
            invention_title=request.invention_title,
            invention_description=request.invention_description,
            components=request.components,
            prior_arts=request.prior_arts,
        )
    except Exception:
        logger.exception("[Novelty API] 분석 중 예상치 못한 오류")
        raise HTTPException(status_code=500, detail="서버 내부 오류")

    if result is None:
        raise HTTPException(
            status_code=502,
            detail="신규성 분석에 실패했습니다. 잠시 후 다시 시도해주세요.",
        )

    return NoveltyAnalyzeResponse(
        d1_application_number=result.application_number,
        overall_similarity=result.overall_similarity,
        conclusion_text=result.conclusion_text,
        component_results=result.component_results,
    )