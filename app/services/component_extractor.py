"""
============================================================
발명 구성요소 자동 추출 서비스
============================================================
변리사가 입력한 발명 정보(명칭, 기술 분야, 핵심 기능 설명)를 받아
청구항 구성요소 단위로 분해한다.

이 결과는:
  - UI의 "구성요소 분석" 리스트에 표시됨
  - invention_components 테이블에 저장됨
  - 신규성/진보성 분석 시 선행기술과 대비할 단위로 사용됨
============================================================
"""

import json
import logging
from typing import Optional, List

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.services.llm_json import strip_code_fence
from app.services.llm_retry import post_with_retry

logger = logging.getLogger(__name__)

CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"


# ============================================================
# 응답 모델
# ============================================================

class Component(BaseModel):
    """청구항 구성요소 1개"""
    name: str = Field(description="구성요소 명칭 (20자 이내 명사구)")
    description: str = Field(description="구성요소 상세 설명 (50~100자)")

class ComponentResponse(BaseModel):
    """LLM 응답 스키마 (개수 검증 포함)"""
    components: List[Component] = Field(min_items=1, max_items=7)


# ============================================================
# 커스텀 예외
# ============================================================

class InvalidInventionError(Exception):
    """발명 설명이 특허 대상이 아닌 경우 (광고성 문구, 감정 표현 등)"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)

# ============================================================
# 프롬프트
# ============================================================

SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 변리사입니다.
변리사가 출원 준비 중인 발명 정보를 받아, 청구항으로 작성될 핵심 기술 요소(구성요소) 단위로 분해합니다.

이 결과는 다음 목적에 사용됩니다:
- 신규성 분석: 각 구성요소가 선행기술에 이미 개시되었는지 판단
- 진보성 분석: 각 구성요소의 조합이 통상의 기술자에게 자명한지 판단

출력 규칙:
1. JSON 형식만 출력. 마크다운 코드 블록(```), 설명 텍스트, 인사말 금지.
2. 한국어로 작성.
3. 1~7개의 구성요소로 분해. 너무 잘게 쪼개거나 너무 크게 뭉치지 말 것.
4. 각 구성요소는 신규성/진보성 판단 시 선행기술과 독립적으로 대비할 수 있는 단위여야 함.
5. 발명 설명에 명시된 수치, 조건, 재료, 알고리즘명, 모델명 등 구체적 정보는 그대로 반영.

반드시 지킬 것 (부정 명령):
- 발명 설명에 없는 내용을 추측하거나 창작 금지
- 여러 기능을 하나의 구성요소에 뭉치지 말 것 (예: "데이터 수집 및 처리부"는 두 개로 분리)
- 일반적/추상적 명칭 금지 (예: "처리부"보다 "이미지 노이즈 검출 모듈"이 구체적)
- 발명 설명이 광고성 문구, 감정 표현, 기술적 실체가 없는 아이디어만 담고 있다면
  다음 형식으로 반환: {"error": "invalid_invention", "reason": "<구체적 사유>"}

출력 형식:
{
  "components": [
    {"name": "구성요소 명칭", "description": "구성요소 상세 설명"},
    ...
  ]
}

각 필드 작성 가이드:

- name (20자 이내 명사구):
  * 청구항 관례에 따라 "○○부", "○○수단", "○○모듈", "○○장치" 등 기능적 접미사 사용
  * 구성요소의 역할을 함축적으로 표현
  * 좋은 예: "저온 침출 반응조", "BERT 임베딩 모듈", "노이즈 검출부"
  * 나쁜 예: "부분1", "구성요소A", "데이터 처리부" (일반적)

- description (50~100자):
  * "기능(무엇을 하는지) → 구성/조건(어떻게 구성되는가)" 순서
  * 발명 설명의 수치, 화학식, 알고리즘명, 조건 등을 최대한 보존
  * 좋은 예: "사용자 발화 데이터를 BERT 기반 사전학습 모델로 임베딩하여 고차원 특징 벡터를 추출하는 처리 모듈. 개인화 학습의 기반을 제공한다."
  * 나쁜 예: "데이터를 처리합니다" (너무 추상적)
  * 나쁜 예: "BERT 임베딩" (기능 없음, 구성만)

===== 예시 1: 머신러닝 도메인 (G06N) =====

발명의 명칭: 사용자 발화 데이터를 활용한 개인화 음성 인식 시스템
기술 분야: 딥러닝 기반 음성 인식
핵심 기능 설명: 사용자별 발화 데이터를 수집하고, BERT 기반 임베딩 모델로
              특징 벡터를 추출한 뒤, 사용자별 파인튜닝 모델을 적용하여
              개인화된 음성 명령 인식 정확도를 향상시킨다.

예시 출력:
{
  "components": [
    {
      "name": "사용자 발화 데이터 수집부",
      "description": "사용자의 음성 명령을 지속적으로 수집하는 모듈. 개인별 발화 패턴 데이터베이스를 구축하여 개인화 학습의 기반을 제공한다."
    },
    {
      "name": "BERT 기반 임베딩 모듈",
      "description": "수집된 발화 데이터를 BERT 기반 사전학습 모델로 처리하여 고차원 특징 벡터로 변환하는 처리부. 음성의 의미론적 표현을 확보한다."
    },
    {
      "name": "사용자별 파인튜닝 모델",
      "description": "각 사용자의 발화 특징 벡터를 활용하여 개별적으로 미세조정된 인식 모델. 개인화된 음성 명령 인식 정확도를 향상시킨다."
    },
    {
      "name": "개인화 명령 인식부",
      "description": "파인튜닝된 모델을 이용해 실시간 발화를 특정 사용자의 명령으로 분류하는 추론 엔진. 최종 인식 결과를 반환한다."
    }
  ]
}

===== 예시 2: 이미지 처리 도메인 (G06T) =====

발명의 명칭: 저조도 환경에서의 실시간 이미지 노이즈 제거 시스템
기술 분야: 딥러닝 기반 이미지 향상
핵심 기능 설명: 저조도 이미지를 U-Net 기반 노이즈 검출 네트워크로 분석하고,
              검출된 노이즈 영역에 GAN 기반 복원 알고리즘을 적용하여
              실시간으로 노이즈를 제거한다. 처리 속도는 30fps 이상.

예시 출력:
{
  "components": [
    {
      "name": "저조도 이미지 입력 모듈",
      "description": "카메라 또는 저장 매체로부터 저조도 이미지를 실시간으로 수신하는 입력 인터페이스. 30fps 이상의 처리 속도를 보장한다."
    },
    {
      "name": "U-Net 기반 노이즈 검출 네트워크",
      "description": "입력 이미지의 노이즈 영역을 픽셀 단위로 세그멘테이션하는 U-Net 구조의 딥러닝 네트워크. 검출 정확도를 극대화한다."
    },
    {
      "name": "GAN 기반 노이즈 복원 모듈",
      "description": "검출된 노이즈 영역에 GAN 기반 이미지 복원 알고리즘을 적용하여 자연스러운 픽셀 값으로 재구성하는 처리부."
    },
    {
      "name": "실시간 프레임 출력부",
      "description": "복원된 이미지 프레임을 30fps 이상 속도로 출력하는 렌더링 모듈. 실시간 처리 성능을 유지한다."
    }
  ]
}

이제 JSON만 출력하세요.
"""

USER_PROMPT_TEMPLATE = """\
발명의 명칭: {title}
기술 분야: {technical_field}
핵심 기능 설명: {description}

위 발명을 청구항 구성요소로 분해하여 JSON으로 반환하세요.
"""


# ============================================================
# 구성요소 추출 함수
# ============================================================

async def extract_components(
    title: str,
    description: str,
    technical_field: Optional[str] = None,
) -> list[Component]:
    """
    발명 정보로부터 청구항 구성요소를 자동 추출.

    Args:
        title: 발명의 명칭
        description: 핵심 기능 설명
        technical_field: 기술 분야 (선택)

    Returns:
        Component 리스트 (1~7개)

    Raises:
        ValueError: LLM 응답 파싱 실패
        httpx.HTTPError: API 호출 실패
    """
    user_message = USER_PROMPT_TEMPLATE.format(
        title=title,
        technical_field=technical_field or "(미제공)",
        description=description,
    )

    payload = {
        "model": settings.claude_component_model,
        "max_tokens": 2048,   # 구성요소 최대 7개 여유
        "temperature": 0.2,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    }

    headers = {
        "x-api-key": settings.claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # LLM 호출
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await post_with_retry(
            client, CLAUDE_ENDPOINT, headers=headers, json=payload, log_prefix="[Components]",
        )
        data = response.json()

    # max_tokens 도달 감지 (응답 잘림 방지)
    stop_reason = data.get("stop_reason")
    if stop_reason == "max_tokens":
        logger.error(f"[Components] max_tokens 도달로 응답 잘림. "
                     f"max_tokens 증량 필요: {payload['max_tokens']}")
        raise ValueError("LLM 응답이 최대 토큰 초과로 잘렸습니다.")

    # 응답 텍스트 추출
    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError):
        logger.error(f"[Components] Claude 응답 구조 이상: {data}")
        raise ValueError("Claude 응답 구조가 예상과 다릅니다.")

    # JSON 파싱
    try:
        parsed = json.loads(strip_code_fence(text))
    except json.JSONDecodeError:
        logger.error(f"[Components] JSON 파싱 실패: {text[:300]}")
        raise ValueError("Claude가 유효한 JSON을 반환하지 않았습니다.")

    # Invalid 입력 케이스 처리
    if isinstance(parsed, dict) and parsed.get("error") == "invalid_invention":
        reason = parsed.get("reason", "특허 등록 가능한 기술적 특징 부족")
        logger.warning(f"[Components] Invalid 발명 감지: {reason}")
        raise InvalidInventionError(reason)

    try:
        result = ComponentResponse(**parsed)
    except ValidationError as e:
        logger.error(f"[Components] 응답 스키마 검증 실패: {e}, parsed={parsed}")
        raise ValueError(f"응답 스키마 검증 실패: {e}")

    logger.info(f"[Components] 추출 완료: {len(result.components)}개")
    return result.components