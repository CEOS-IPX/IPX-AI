"""
============================================================
발명 구성요소 자동 추출 API 라우터
============================================================
Spring 서버가 "AI 자동 생성" 버튼 클릭 시 호출.

Spring 흐름:
  1. 변리사 UI에서 "AI 자동 생성" 버튼 클릭
  2. Spring: cases 테이블에서 title, description, technical_field 조회
  3. Spring: Python /components/extract 호출
  4. Python: LLM으로 구성요소 추출 후 반환
  5. Spring: invention_components 테이블에서 기존 데이터 삭제
  6. Spring: 새 구성요소 INSERT (display_order 부여)
  7. Spring: 프론트에 응답 → UI 리스트 갱신 (기존 덮어쓰기)
============================================================
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.component_extractor import extract_components, Component

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/components", tags=["components"])


# ============================================================
# 요청/응답 모델
# ============================================================

class ComponentExtractRequest(BaseModel):
    """Spring → Python 구성요소 추출 요청"""
    title: str = Field(description="발명의 명칭")
    description: str = Field(description="핵심 기능 설명")
    technical_field: Optional[str] = Field(default=None, description="기술 분야 (선택)")


class ComponentExtractResponse(BaseModel):
    """Python → Spring 응답"""
    components: list[Component] = Field(description="추출된 구성요소 리스트 (3~7개)")


# ============================================================
# 엔드포인트
# ============================================================

@router.post("/extract", response_model=ComponentExtractResponse)
async def extract(request: ComponentExtractRequest) -> ComponentExtractResponse:
    """
    발명 정보로부터 청구항 구성요소를 자동 추출.

    Args:
        request: 발명의 명칭, 핵심 기능 설명, 기술 분야 (선택)

    Returns:
        추출된 구성요소 리스트 (3~7개)

    Raises:
        400: 필수 필드 부족
        502: LLM 호출 실패
    """
    if not request.title.strip() or not request.description.strip():
        raise HTTPException(
            status_code=400,
            detail="발명의 명칭과 핵심 기능 설명은 필수입니다.",
        )

    logger.info(
        f"[Components] 추출 요청: title='{request.title}', "
        f"technical_field='{request.technical_field}'"
    )

    try:
        components = await extract_components(
            title=request.title,
            description=request.description,
            technical_field=request.technical_field,
        )
    except ValueError as e:
        logger.warning(f"[Components] LLM 응답 처리 실패: {e}")
        raise HTTPException(status_code=502, detail=f"구성요소 추출 실패: {e}")
    except Exception:
        logger.exception("[Components] 예상치 못한 오류")
        raise HTTPException(status_code=500, detail="서버 내부 오류")

    return ComponentExtractResponse(components=components)