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
from typing import Optional

import httpx
from pydantic import BaseModel, Field

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
    description: str = Field(description="구성요소 상세 설명 (50~150자)")


# ============================================================
# 프롬프트
# ============================================================

SYSTEM_PROMPT = """\
당신은 특허 청구항 작성 전문가입니다.
변리사가 출원하려는 발명 정보를 받아, 청구항의 구성요소 단위로 분해합니다.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. 3~7개의 구성요소로 분해. 너무 잘게 쪼개거나 너무 크게 뭉치지 말 것.
4. 각 구성요소는 청구항에서 독립적으로 판단할 수 있는 단위여야 함.
5. 신규성/진보성 분석 시 선행기술과 대비할 수 있는 수준으로 구체적으로 작성.
6. 발명 설명에 명시된 수치, 조건, 재료 등 구체적 정보는 그대로 반영.

출력 형식:
{
  "components": [
    {"name": "구성요소 명칭", "description": "구성요소 상세 설명"},
    ...
  ]
}

각 필드 작성 가이드:

- name: 20자 이내의 명사구.
        구성요소의 역할을 함축적으로 표현.
        예: "저온 침출 반응기", "황산 농도 제어부", "환원제 투입 수단"

- description: 50~150자.
                이 구성요소가 무엇을 하는지, 어떤 조건에서 동작하는지, 어떤 구성인지 구체적으로.
                발명 설명의 수치, 화학식, 조건 등을 가능한 한 보존.

예시:

발명의 명칭: 폐리튬이온전지로부터 니켈과 코발트를 저온에서 회수하는 습식제련 공정
기술 분야: 이차전지 재활용, 습식 야금
핵심 기능 설명: 50-60°C 저온에서 H₂SO₄ 1.0~1.5 M과 환원제 H₂O₂를 단계적으로
              투입하여 폐리튬이온전지 양극재로부터 니켈과 코발트를 92% 이상 회수한다.

예시 출력:
{
  "components": [
    {
      "name": "저온 침출 반응조",
      "description": "50-60°C의 저온 조건에서 폐리튬이온전지 양극재를 침출하는 반응 용기. 저온 유지를 통해 에너지 소비를 최소화하고 부반응을 억제한다."
    },
    {
      "name": "황산 투입부",
      "description": "H₂SO₄ 1.0~1.5 M 농도로 침출 반응조에 주입되는 산 공급 수단. 니켈과 코발트의 선택적 침출을 유도한다."
    },
    {
      "name": "환원제 단계 투입 수단",
      "description": "H₂O₂를 침출 반응 과정에서 단계적으로 투입하는 장치. 침출 선택성을 향상시키고 회수율을 92% 이상으로 유지한다."
    },
    {
      "name": "니켈·코발트 동시 회수부",
      "description": "침출 용액에서 니켈과 코발트를 동시에 분리·회수하는 후속 처리 단계. 별도 공정 없이 두 금속을 함께 회수한다."
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
        Component 리스트 (3~7개)

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
        "temperature": 0.3,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    }

    headers = {
        "x-api-key": settings.claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await post_with_retry(
            client, CLAUDE_ENDPOINT, headers=headers, json=payload, log_prefix="[Components]",
        )
        data = response.json()

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

    # 컴포넌트 리스트 검증
    if "components" not in parsed or not isinstance(parsed["components"], list):
        logger.error(f"[Components] 응답에 components 리스트 없음: {parsed}")
        raise ValueError("응답에 components 리스트가 없습니다.")

    try:
        components = [Component(**c) for c in parsed["components"]]
    except Exception as e:
        logger.error(f"[Components] Component 검증 실패: {e}, parsed={parsed}")
        raise ValueError(f"구성요소 필드 검증 실패: {e}")

    if not components:
        raise ValueError("구성요소가 하나도 추출되지 않았습니다.")

    logger.info(f"[Components] 추출 완료: {len(components)}개")
    return components