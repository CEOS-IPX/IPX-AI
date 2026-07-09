"""
============================================================
진보성 분석 API 라우터
============================================================
5가지 엔드포인트:
  1. /analyze/inventive-step/select-secondary        - D2 자동 선정
  2. /analyze/inventive-step/generate/numerical-limit
  3. /analyze/inventive-step/generate/combination-motivation
  4. /analyze/inventive-step/generate/common-technique
  5. /analyze/inventive-step/generate/simple-design

Spring 흐름:
  - 변리사가 선행문헌함에서 주인용(D1) 선택 + "기술 진보성 분석" 클릭
    → Spring이 select-secondary 호출 → D2 자동 결정
    → inventive_step_analyses INSERT
  - 각 카테고리 카드에서 "AI 자동 생성" 버튼 클릭 시
    → Spring이 해당 카테고리의 generate 엔드포인트 호출
    → inventive_arguments의 content JSONB 업데이트
============================================================
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.inventive_step_analyzer import (
    InventionComponent,
    PriorArtInfo,
    select_secondary_art,
    generate_numerical_limit,
    generate_combination_motivation,
    generate_common_technique,
    generate_simple_design,
    NumericalLimitResult,
    CombinationMotivationResult,
    CommonTechniqueResult,
    SimpleDesignResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze/inventive-step", tags=["inventive-step"])


# ============================================================
# 1. 부인용 D2 자동 선정
# ============================================================

class SelectSecondaryRequest(BaseModel):
    invention_title: str
    invention_description: str
    components: list[InventionComponent]
    primary_art: PriorArtInfo
    candidates: list[PriorArtInfo] = Field(description="D2 후보 (D1 제외한 상위 N건, included=true)")


class SelectSecondaryResponse(BaseModel):
    d2_application_number: str


@router.post("/select-secondary", response_model=SelectSecondaryResponse)
async def select_secondary_endpoint(
    request: SelectSecondaryRequest,
) -> SelectSecondaryResponse:
    """
    D1이 주어졌을 때 D2(부인용)를 후보 중에서 자동 선정.

    Raises:
        400: 후보가 비어있음
        502: LLM 판단 실패
    """
    if not request.candidates:
        raise HTTPException(
            status_code=400,
            detail="D2 후보가 없습니다. 선행기술 목록을 확인해주세요.",
        )
    if not request.components:
        raise HTTPException(
            status_code=400,
            detail="구성요소가 필요합니다. 먼저 구성요소를 등록해주세요.",
        )

    result = await select_secondary_art(
        invention_title=request.invention_title,
        invention_description=request.invention_description,
        components=request.components,
        primary_art=request.primary_art,
        candidates=request.candidates,
    )

    if result is None:
        raise HTTPException(status_code=502, detail="D2 선정에 실패했습니다.")

    # LLM이 후보에 없는 출원번호를 반환한 경우 방어
    candidate_nums = {c.application_number for c in request.candidates}
    if result.d2_application_number not in candidate_nums:
        logger.warning(
            f"[InventiveStep] LLM이 후보에 없는 D2 반환: {result.d2_application_number}, "
            f"후보={candidate_nums}. 첫 번째 후보로 대체."
        )
        result.d2_application_number = request.candidates[0].application_number

    return SelectSecondaryResponse(
        d2_application_number=result.d2_application_number,
    )


# ============================================================
# 2. 수치한정 논리 생성
# ============================================================

class GenerateNumericalLimitRequest(BaseModel):
    invention_title: str
    invention_description: str
    primary_art: PriorArtInfo


@router.post("/generate/numerical-limit", response_model=NumericalLimitResult)
async def generate_numerical_limit_endpoint(
    request: GenerateNumericalLimitRequest,
) -> NumericalLimitResult:
    """수치한정 (발명의 효과 표) 자동 생성"""

    result = await generate_numerical_limit(
        invention_title=request.invention_title,
        invention_description=request.invention_description,
        primary_art=request.primary_art,
    )

    if result is None:
        raise HTTPException(status_code=502, detail="수치한정 논리 생성에 실패했습니다.")

    return result


# ============================================================
# 3. 복수인용발명결합 논리 생성
# ============================================================

class GenerateCombinationMotivationRequest(BaseModel):
    invention_title: str
    invention_description: str
    primary_art: PriorArtInfo
    secondary_art: PriorArtInfo


@router.post("/generate/combination-motivation", response_model=CombinationMotivationResult)
async def generate_combination_motivation_endpoint(
    request: GenerateCombinationMotivationRequest,
) -> CombinationMotivationResult:
    """복수인용발명결합 (Teaching Away) 논리 자동 생성"""

    result = await generate_combination_motivation(
        invention_title=request.invention_title,
        invention_description=request.invention_description,
        primary_art=request.primary_art,
        secondary_art=request.secondary_art,
    )

    if result is None:
        raise HTTPException(status_code=502, detail="복수인용발명결합 논리 생성에 실패했습니다.")

    return result


# ============================================================
# 4. 주지관용기술 반박 생성
# ============================================================

class GenerateCommonTechniqueRequest(BaseModel):
    invention_title: str
    invention_description: str
    components: list[InventionComponent]
    primary_art: PriorArtInfo


@router.post("/generate/common-technique", response_model=CommonTechniqueResult)
async def generate_common_technique_endpoint(
    request: GenerateCommonTechniqueRequest,
) -> CommonTechniqueResult:
    """주지관용기술 반박 논리 자동 생성"""

    if not request.components:
        raise HTTPException(status_code=400, detail="구성요소가 필요합니다.")

    result = await generate_common_technique(
        invention_title=request.invention_title,
        invention_description=request.invention_description,
        components=request.components,
        primary_art=request.primary_art,
    )

    if result is None:
        raise HTTPException(status_code=502, detail="주지관용기술 반박 생성에 실패했습니다.")

    return result


# ============================================================
# 5. 단순설계변경 논리 생성
# ============================================================

class GenerateSimpleDesignRequest(BaseModel):
    invention_title: str
    invention_description: str
    components: list[InventionComponent]
    primary_art: PriorArtInfo


@router.post("/generate/simple-design", response_model=SimpleDesignResult)
async def generate_simple_design_endpoint(
    request: GenerateSimpleDesignRequest,
) -> SimpleDesignResult:
    """단순설계변경 비자명성 논리 자동 생성"""

    if not request.components:
        raise HTTPException(status_code=400, detail="구성요소가 필요합니다.")

    result = await generate_simple_design(
        invention_title=request.invention_title,
        invention_description=request.invention_description,
        components=request.components,
        primary_art=request.primary_art,
    )

    if result is None:
        raise HTTPException(status_code=502, detail="단순설계변경 논리 생성에 실패했습니다.")

    return result