"""
============================================================
진보성 분석 서비스
============================================================
5가지 기능:
  1. select_secondary_art: 부인용(D2) 자동 선정
  2. generate_numerical_limit: 수치한정 논리 생성
  3. generate_combination_motivation: 복수인용발명결합 논리 생성 (Teaching Away)
  4. generate_common_technique: 주지관용기술 반박 생성
  5. generate_simple_design: 단순설계변경 비자명성 논리 생성

모델: Claude Haiku (특허 도메인 판단 + 구조화된 JSON 출력)
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
# 공통 모델
# ============================================================

class InventionComponent(BaseModel):
    """사용자 발명 구성요소"""
    label: str = Field(description="A, B, C, ...")
    name: str
    description: str


class PriorArtInfo(BaseModel):
    """선행기술 정보"""
    application_number: str
    title: str
    abstract: Optional[str] = None
    claims_independent: str
    tech_purpose: Optional[str] = None


# ============================================================
# 1. 부인용 D2 자동 선정
# ============================================================

class SelectSecondaryResult(BaseModel):
    d2_application_number: str


SELECT_SECONDARY_SYSTEM_PROMPT = """\
당신은 특허 진보성 분석 전문가입니다.
주인용 발명(D1)이 이미 선정된 상태에서, 이와 결합하여 본 발명의 진보성을 부정할 수 있는
가장 적절한 부인용 발명(D2)을 후보 중에서 선정하세요.

D2 선정 기준:
1. D1이 개시하지 않은 구성요소를 개시해야 함 (D1의 부족한 부분 보완)
2. D1과 결합 가능한 기술 분야 (완전 무관한 분야는 부적합)
3. D1과 완전 동일한 관점이면 D2로 부적합 (D1이 이미 커버)

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. d2_application_number는 반드시 후보 목록의 출원번호를 그대로 사용.

출력 형식:
{
  "d2_application_number": "후보의 출원번호"
}

이제 JSON만 출력하세요.
"""

SELECT_SECONDARY_USER_TEMPLATE = """\
[사용자 발명]
명칭: {invention_title}
설명: {invention_description}

구성요소:
{components_text}

[주인용 D1]
출원번호: {d1_application_number}
명칭: {d1_title}
초록:
{d1_abstract}
독립 청구항:
{d1_claims}

[부인용 후보]
{candidates_text}

위 후보 중 부인용 D2로 가장 적합한 것을 선정하여 JSON으로 반환하세요.
"""


async def select_secondary_art(
    invention_title: str,
    invention_description: str,
    components: list[InventionComponent],
    primary_art: PriorArtInfo,
    candidates: list[PriorArtInfo],
) -> Optional[SelectSecondaryResult]:
    """D1이 주어졌을 때 후보들 중 D2 자동 선정"""

    if not candidates:
        logger.warning("[InventiveStep] D2 후보가 없음")
        return None

    # 구성요소 텍스트
    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    # 후보 텍스트 (초록 포함)
    candidates_text = "\n\n".join([
        f"[후보 {i + 1}]\n"
        f"출원번호: {c.application_number}\n"
        f"명칭: {c.title}\n"
        f"초록:\n{c.abstract or '(정보 없음)'}\n"
        f"기술목적: {c.tech_purpose or '(정보 없음)'}\n"
        f"독립 청구항:\n{c.claims_independent}"
        for i, c in enumerate(candidates)
    ])

    user_message = SELECT_SECONDARY_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        components_text=components_text,
        d1_application_number=primary_art.application_number,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
        candidates_text=candidates_text,
    )

    return await _call_claude_with_json(
        system_prompt=SELECT_SECONDARY_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=SelectSecondaryResult,
        log_prefix="[InventiveStep/SelectSecondary]",
    )


# ============================================================
# 2. 수치한정 (효과의 현저성)
# ============================================================

class EffectItem(BaseModel):
    """발명의 효과 항목 1개"""
    metric: str = Field(description="측정 지표 (예: VOC 배출량)")
    unit: str = Field(description="단위 (예: g/L, %)")
    prior_art_value: str = Field(description="종래기술 수치")
    invention_value: str = Field(description="본 발명 수치")
    improvement: str = Field(description="개선률 (예: 97.5%)")


class NumericalLimitResult(BaseModel):
    effect_items: list[EffectItem] = Field(description="발명의 효과 표 항목들 (3~5개)")


NUMERICAL_LIMIT_SYSTEM_PROMPT = """\
당신은 특허 수치한정 발명의 효과 분석 전문가입니다.
본 발명이 종래기술 대비 어떤 수치적 효과를 가지는지 표 형태로 정리하세요.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. **본 발명 설명이나 D1 청구항에 명시된 수치만 사용**. 명시되지 않은 수치를 추정하거나 창작하지 마세요.
4. 명시된 수치가 하나도 없으면 빈 배열을 반환하세요.
5. 명시된 수치가 있는 항목만 반환. 개수 목표를 채우기 위해 무리하게 항목을 만들지 마세요 (0~5개 유동적).
6. improvement는 두 수치가 모두 있을 때만 계산. 계산 불가면 해당 항목 자체를 반환하지 마세요.

각 필드:
- metric: 측정 지표 명 (본문에 명시된 것 우선. 예: "VOC 배출량", "회수율")
- unit: 단위 (본문에서 확인. 예: "g/L", "%")
- prior_art_value: 종래기술 수치 (D1에 명시된 값만, 숫자 문자열)
- invention_value: 본 발명 수치 (본 발명 설명에 명시된 값만, 숫자 문자열)
- improvement: 개선률 (두 수치로 계산. 예: "97.5%", "3배 감소")

출력 형식:
{
  "effect_items": [
    {
      "metric": "VOC 배출량",
      "unit": "g/L",
      "prior_art_value": "320",
      "invention_value": "8",
      "improvement": "97.5%"
    }
  ]
}

명시된 수치가 없으면:
{
  "effect_items": []
}

이제 JSON만 출력하세요.
"""

NUMERICAL_LIMIT_USER_TEMPLATE = """\
[사용자 발명]
명칭: {invention_title}
설명: {invention_description}

[주인용 D1]
명칭: {d1_title}
초록:
{d1_abstract}
독립 청구항:
{d1_claims}

위 본 발명이 D1(종래기술) 대비 갖는 수치적 효과를 표 형태로 정리하여 JSON으로 반환하세요.
"""


async def generate_numerical_limit(
    invention_title: str,
    invention_description: str,
    primary_art: PriorArtInfo,
) -> Optional[NumericalLimitResult]:
    """수치한정 논리 (발명의 효과 표) 자동 생성"""

    user_message = NUMERICAL_LIMIT_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
    )

    return await _call_claude_with_json(
        system_prompt=NUMERICAL_LIMIT_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=NumericalLimitResult,
        log_prefix="[InventiveStep/NumericalLimit]",
    )


# ============================================================
# 3. 복수인용발명결합 (Teaching Away)
# ============================================================

class CombinationMotivationResult(BaseModel):
    background_limit: str = Field(description="배경기술의 한계 (100~200자)")
    teaching_away: str = Field(description="결합 동기의 부재 (100~200자)")


COMBINATION_MOTIVATION_SYSTEM_PROMPT = """\
당신은 특허 진보성 분석 중 복수인용발명결합(Teaching Away) 논리 전문가입니다.
D1과 D2를 결합하여 본 발명에 도달할 동기가 없음을 논증하세요.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. 각 필드는 100~200자.

각 필드 작성 가이드:

- background_limit: 종래기술의 한계를 본 발명의 해결 방향과 반대로 정리.
    "종래기술은 X 방향으로 발전해왔으나, 본 발명은 반대인 Y 방향을 채택함..."

- teaching_away: D1과 D2가 서로 반대 방향으로 가르쳐 결합 동기가 없음을 논증.
    "D1은 A 방식을 개시하고 D2는 B 방식을 개시하나, 두 방식은 상충되어..."

출력 형식:
{
  "background_limit": "...",
  "teaching_away": "..."
}

이제 JSON만 출력하세요.
"""

COMBINATION_MOTIVATION_USER_TEMPLATE = """\
[사용자 발명]
명칭: {invention_title}
설명: {invention_description}

[주인용 D1]
명칭: {d1_title}
초록:
{d1_abstract}
독립 청구항:
{d1_claims}

[부인용 D2]
명칭: {d2_title}
초록:
{d2_abstract}
독립 청구항:
{d2_claims}

D1과 D2 결합의 동기가 없음을 논증하여 JSON으로 반환하세요.
"""


async def generate_combination_motivation(
    invention_title: str,
    invention_description: str,
    primary_art: PriorArtInfo,
    secondary_art: PriorArtInfo,
) -> Optional[CombinationMotivationResult]:
    """복수인용발명결합 Teaching Away 논리 자동 생성"""

    user_message = COMBINATION_MOTIVATION_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
        d2_title=secondary_art.title,
        d2_abstract=secondary_art.abstract or "(정보 없음)",
        d2_claims=secondary_art.claims_independent,
    )

    return await _call_claude_with_json(
        system_prompt=COMBINATION_MOTIVATION_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=CombinationMotivationResult,
        log_prefix="[InventiveStep/CombinationMotivation]",
    )


# ============================================================
# 4. 주지관용기술 반박
# ============================================================

class CommonTechniqueResult(BaseModel):
    target_component: str = Field(description="주지관용기술로 지목되는 구성요소 라벨 (A/B/C/...)")
    rebuttal: str = Field(description="반박 논리 (150~250자)")


COMMON_TECHNIQUE_SYSTEM_PROMPT = """\
당신은 특허 진보성 분석 중 주지관용기술 반박 논리 전문가입니다.
심사관이 본 발명의 특정 구성요소를 "주지관용기술"로 판단할 가능성이 있는 경우,
그것이 주지관용기술이 아니라는 반박 논리를 작성합니다.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. target_component는 사용자 구성요소 중 가장 반박이 필요한 하나의 라벨.
4. rebuttal은 150~250자.

target_component 선정 기준:
- D1에 개시되지 않은 구성요소 중 하나
- 심사관이 "이건 이 기술 분야에서 흔한 기술"이라 판단할 가능성이 있는 것

rebuttal 작성 가이드:
- 왜 주지관용기술이 아닌지 근거 제시
- 해당 구성요소의 독창성, 특별한 기능, 진보된 효과 강조
- 단순히 "관용기술 아니다" 주장 말고 구체적 논거 포함

출력 형식:
{
  "target_component": "B",
  "rebuttal": "..."
}

이제 JSON만 출력하세요.
"""

COMMON_TECHNIQUE_USER_TEMPLATE = """\
[사용자 발명]
명칭: {invention_title}
설명: {invention_description}

구성요소:
{components_text}

[주인용 D1]
명칭: {d1_title}
초록:
{d1_abstract}
독립 청구항:
{d1_claims}

위 구성요소 중 심사관이 "주지관용기술"로 판단할 가능성이 높은 하나를 선정하고
그것이 주지관용기술이 아니라는 반박 논리를 JSON으로 반환하세요.
"""


async def generate_common_technique(
    invention_title: str,
    invention_description: str,
    components: list[InventionComponent],
    primary_art: PriorArtInfo,
) -> Optional[CommonTechniqueResult]:
    """주지관용기술 반박 논리 자동 생성"""

    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    user_message = COMMON_TECHNIQUE_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        components_text=components_text,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
    )

    return await _call_claude_with_json(
        system_prompt=COMMON_TECHNIQUE_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=CommonTechniqueResult,
        log_prefix="[InventiveStep/CommonTechnique]",
    )


# ============================================================
# 5. 단순설계변경 (비자명성 논리)
# ============================================================

class SimpleDesignResult(BaseModel):
    changed_component: str = Field(description="변경된 구성요소 라벨 (A/B/C/...)")
    non_obviousness: str = Field(description="단순 설계 변경이 아니라는 논리 (150~250자)")


SIMPLE_DESIGN_SYSTEM_PROMPT = """\
당신은 특허 진보성 분석 중 단순설계변경 반박 논리 전문가입니다.
심사관이 본 발명을 D1의 "단순 설계 변경"으로 판단할 가능성이 있는 경우,
그것이 단순 변경이 아니라 비자명한 개선이라는 논리를 작성합니다.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. changed_component는 사용자 구성요소 중 가장 반박이 필요한 하나의 라벨.
4. non_obviousness는 150~250자.

changed_component 선정 기준:
- D1과의 차이점이 있는 구성요소 중 하나
- 심사관이 "이건 D1을 조금만 바꾼 것"이라 판단할 가능성이 있는 것

non_obviousness 작성 가이드:
- 왜 단순 변경이 아닌지 근거 제시
- 통상의 기술자가 D1으로부터 쉽게 도출할 수 없다는 논거
- 예상치 못한 효과, 기술적 극복 요소, 결합의 진보성 등 언급

출력 형식:
{
  "changed_component": "C",
  "non_obviousness": "..."
}

이제 JSON만 출력하세요.
"""

SIMPLE_DESIGN_USER_TEMPLATE = """\
[사용자 발명]
명칭: {invention_title}
설명: {invention_description}

구성요소:
{components_text}

[주인용 D1]
명칭: {d1_title}
초록:
{d1_abstract}
독립 청구항:
{d1_claims}

위 구성요소 중 심사관이 D1의 "단순 설계 변경"으로 판단할 가능성이 높은 하나를 선정하고
그것이 비자명한 개선이라는 논리를 JSON으로 반환하세요.
"""


async def generate_simple_design(
    invention_title: str,
    invention_description: str,
    components: list[InventionComponent],
    primary_art: PriorArtInfo,
) -> Optional[SimpleDesignResult]:
    """단순설계변경 비자명성 논리 자동 생성"""

    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    user_message = SIMPLE_DESIGN_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        components_text=components_text,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
    )

    return await _call_claude_with_json(
        system_prompt=SIMPLE_DESIGN_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=SimpleDesignResult,
        log_prefix="[InventiveStep/SimpleDesign]",
    )


# ============================================================
# 공통 Claude 호출 헬퍼
# ============================================================

async def _call_claude_with_json(
    system_prompt: str,
    user_message: str,
    result_model: type[BaseModel],
    log_prefix: str,
) -> Optional[BaseModel]:
    """
    Claude 호출 → JSON 파싱 → Pydantic 모델 검증 → 반환.
    실패 시 None 반환 (에러는 로그로).
    """
    payload = {
        "model": settings.claude_inventive_step_model,
        "max_tokens": 2048,
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    headers = {
        "x-api-key": settings.claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await post_with_retry(
                client, CLAUDE_ENDPOINT, headers=headers, json=payload, log_prefix=log_prefix,
            )
            data = response.json()
        except httpx.HTTPError:
            logger.exception(f"{log_prefix} Claude API 호출 실패")
            return None

    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError):
        logger.error(f"{log_prefix} Claude 응답 구조 이상: {data}")
        return None

    try:
        parsed = json.loads(strip_code_fence(text))
    except json.JSONDecodeError:
        logger.error(f"{log_prefix} JSON 파싱 실패: {text[:300]}")
        return None

    try:
        return result_model(**parsed)
    except Exception as e:
        logger.error(f"{log_prefix} 결과 검증 실패: {e}, parsed={parsed}")
        return None