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
# 0. 부인용 D2 자동 선정
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
# 1. 카테고리 자동 선정
# ============================================================

class SelectCategoriesResult(BaseModel):
    categories: list[str]


SELECT_CATEGORIES_SYSTEM_PROMPT = """\
당신은 특허 진보성 분석 전문가입니다.
본 발명과 주인용(D1), 부인용(D2)를 분석해 진보성 논쟁에서 이슈가 될 만한 카테고리를 선정하세요.

가능한 4개 카테고리:
1. numerical_limit (수치한정):
   - 본 발명이 정량적 성능 지표를 개시하고, D1 대비 개선률 명확한 경우
   - 사용자가 measurement_conditions, measurement_results를 입력했다면 강력한 근거

2. combination_motivation (복수인용발명결합):
   - D1과 D2가 서로 다른 방향의 기술이거나 결합 동기가 없어 보이는 경우
   - 사용자가 prior_art_reference, differentiation_notes를 입력했다면 참고

3. common_technique (주지관용기술):
   - D1에 없는 구성요소 중 관용기술로 오해받을 만한 것이 있는 경우
   - 특별한 기능·효과가 있는 구성요소가 있는지 판단

4. simple_design (단순설계변경):
   - D1과 본 발명의 차이가 미묘하거나 파라미터 조정으로 보일 위험이 있는 경우
   - 실제로는 비자명한 개선인지 판단

선정 규칙:
1. 관련성이 명확한 카테고리만 선정. 억지로 4개 다 넣지 마세요.
2. 최소 1개, 최대 4개. 보통 1~3개가 적절.
3. 사용자가 입력한 추가 정보(measurement_*, differentiation_*)가 있다면 관련 카테고리 우선.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. categories 배열에 위 4개 문자열만 사용.
3. reasoning은 100~200자.

출력 형식:
{
  "categories": ["numerical_limit", "combination_motivation"]
}

이제 JSON만 출력하세요.
"""

SELECT_CATEGORIES_USER_TEMPLATE = """\
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

[부인용 D2]
명칭: {d2_title}
초록:
{d2_abstract}
독립 청구항:
{d2_claims}
{additional_info_text}

위 정보를 바탕으로 진보성 논리 카테고리를 선정하여 JSON으로 반환하세요.
"""


async def select_relevant_categories(
        invention_title: str,
        invention_description: str,
        components: list[InventionComponent],
        primary_art: PriorArtInfo,
        secondary_art: PriorArtInfo,
        prior_art_reference: Optional[str] = None,
        differentiation_notes: Optional[str] = None,
        measurement_conditions: Optional[str] = None,
        measurement_results: Optional[str] = None,
) -> Optional[SelectCategoriesResult]:
    """4개 카테고리 중 관련 있는 것들 자동 선정"""

    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    additional_parts = []
    if prior_art_reference or differentiation_notes:
        additional_parts.append(
            f"\n[사용자가 언급한 선행기술 대비 차별점]\n"
            f"선행기술: {prior_art_reference or '(없음)'}\n"
            f"차이점: {differentiation_notes or '(없음)'}"
        )
    if measurement_conditions or measurement_results:
        additional_parts.append(
            f"\n[사용자가 제공한 측정 데이터]\n"
            f"측정 조건: {measurement_conditions or '(없음)'}\n"
            f"측정 결과: {measurement_results or '(없음)'}"
        )
    additional_info_text = "\n".join(additional_parts) if additional_parts else ""

    user_message = SELECT_CATEGORIES_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        components_text=components_text,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
        d2_title=secondary_art.title,
        d2_abstract=secondary_art.abstract or "(정보 없음)",
        d2_claims=secondary_art.claims_independent,
        additional_info_text=additional_info_text,
    )

    return await _call_claude_with_json(
        system_prompt=SELECT_CATEGORIES_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=SelectCategoriesResult,
        log_prefix="[InventiveStep/SelectCategories]",
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
3. **본 발명 설명, D1 청구항, 또는 [사용자가 제공한 실험 데이터] 섹션에 명시된 수치만 사용**. 명시되지 않은 수치를 추정하지 마세요.
4. 사용자 제공 데이터가 있으면 이를 우선 활용.
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
{measurement_info}

위 본 발명이 D1(종래기술) 대비 갖는 수치적 효과를 표 형태로 정리하여 JSON으로 반환하세요.
"""


async def generate_numerical_limit(
    invention_title: str,
    invention_description: str,
    primary_art: PriorArtInfo,
    measurement_conditions: Optional[str] = None,
    measurement_results: Optional[str] = None,
) -> Optional[NumericalLimitResult]:
    """수치한정 논리 (발명의 효과 표) 자동 생성"""

    # 사용자 실험 데이터 조립
    measurement_text = ""
    if measurement_conditions or measurement_results:
        measurement_text = (
            f"\n[사용자가 제공한 실험 데이터]\n"
            f"측정 조건: {measurement_conditions or '(없음)'}\n"
            f"측정 결과: {measurement_results or '(없음)'}\n"
            f"\n위 사용자 제공 데이터의 수치를 우선 활용하세요."
        )

    user_message = NUMERICAL_LIMIT_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
        measurement_info=measurement_text,
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
4. 사용자가 [사용자가 언급한 선행기술 대비 차별점] 섹션을 제공한 경우,
   해당 내용을 활용하여 D1/D2가 본 발명과 다른 방향임을 뒷받침하세요.

각 필드 작성 가이드:

- background_limit: 종래기술이 가진 근본적 한계. D1과 D2가 각자 그 한계를 완전히 해결하지 못하고,
    본 발명이 등장할 필연성이 있음을 정리.
    "종래기술은 X 문제를 안고 있으며, D1과 D2 각각 접근은 이를 근본적으로 해결하지 못함..."

- teaching_away: D1과 D2가 서로 다른 방향으로 가르쳐, 통상의 기술자가 두 문헌을 결합할 동기가 없음을 논증.
    "D1은 A 방식을 개시하고 D2는 B 방식을 개시하나, 두 방식은 상충되어 결합 시 오히려 성능 저하가 예상됨..."

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
{differentiation_info}

D1과 D2 결합의 동기가 없음을 논증하여 JSON으로 반환하세요.
"""


async def generate_combination_motivation(
    invention_title: str,
    invention_description: str,
    primary_art: PriorArtInfo,
    secondary_art: PriorArtInfo,
    prior_art_reference: Optional[str] = None,
    differentiation_notes: Optional[str] = None,
) -> Optional[CombinationMotivationResult]:
    """복수인용발명결합 Teaching Away 논리 자동 생성"""

    differentiation_text = ""
    if prior_art_reference or differentiation_notes:
        differentiation_text = (
            f"\n[사용자가 언급한 선행기술 대비 차별점]\n"
            f"선행기술: {prior_art_reference or '(없음)'}\n"
            f"차이점: {differentiation_notes or '(없음)'}"
        )

    user_message = COMBINATION_MOTIVATION_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
        d2_title=secondary_art.title,
        d2_abstract=secondary_art.abstract or "(정보 없음)",
        d2_claims=secondary_art.claims_independent,
        differentiation_info=differentiation_text,
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
    target_label: str = Field(description="주지관용기술로 지목되는 구성요소 라벨 (A/B/C/...)")
    target_name: str = Field(description="주지관용기술로 지목되는 구성요소 이름")
    rebuttal: str = Field(description="반박 논리 (150~250자)")


COMMON_TECHNIQUE_SYSTEM_PROMPT = """\
당신은 특허 진보성 분석 중 주지관용기술 반박 논리 전문가입니다.
심사관이 본 발명의 특정 구성요소를 "주지관용기술"로 판단할 가능성이 있는 경우,
그것이 주지관용기술이 아니라는 반박 논리를 작성합니다.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. target_label과 targe_name은 각각 사용자 구성요소 중 가장 반박이 필요한 하나의 라벨과 그 이름.
4. rebuttal은 150~250자.
5. 사용자가 [사용자가 언급한 선행기술 대비 차별점] 섹션을 제공한 경우,
   해당 내용을 반박 논거로 활용하세요.

target_label 선정 기준:
- D1에 개시되지 않은 구성요소 중 하나
- 심사관이 "이건 이 기술 분야에서 흔한 기술"이라 판단할 가능성이 있는 것

rebuttal 작성 가이드:
다음 중 하나 이상의 논거를 포함하세요:
1. 해당 구성요소가 D1에 없음 → 새로 도입된 기술적 특징
2. 특별한 기능·효과가 있음 (사용자 발명 설명, 차별점 참조)
3. 통상의 기술자가 이 기술 분야의 관용기술만으로는 자연스럽게 도출하기 어려움
4. 유사 기술 분야에 존재하더라도, 본 발명의 다른 구성요소와의 결합으로 특유의 상승효과

단순히 "관용기술 아니다" 주장 말고, 위 논거 중 최소 하나 이상 구체적으로 기술하세요.

출력 형식:
{
  "target_label": "B",
  "target_name": "...",
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
{differentiation_info}

위 구성요소 중 심사관이 "주지관용기술"로 판단할 가능성이 높은 하나를 선정하고
그것이 주지관용기술이 아니라는 반박 논리를 JSON으로 반환하세요.
"""


async def generate_common_technique(
    invention_title: str,
    invention_description: str,
    components: list[InventionComponent],
    primary_art: PriorArtInfo,
    prior_art_reference: Optional[str] = None,
    differentiation_notes: Optional[str] = None,
) -> Optional[CommonTechniqueResult]:
    """주지관용기술 반박 논리 자동 생성"""

    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    differentiation_text = ""
    if prior_art_reference or differentiation_notes:
        differentiation_text = (
            f"\n[사용자가 언급한 선행기술 대비 차별점]\n"
            f"선행기술: {prior_art_reference or '(없음)'}\n"
            f"차이점: {differentiation_notes or '(없음)'}"
        )

    user_message = COMMON_TECHNIQUE_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        components_text=components_text,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
        differentiation_info=differentiation_text,
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
    changed_component_label: str = Field(description="변경된 구성요소 라벨 (A/B/C/...)")
    changed_component_name: str = Field(description="변경된 구성요소 이름")
    non_obviousness: str = Field(description="단순 설계 변경이 아니라는 논리 (150~250자)")


SIMPLE_DESIGN_SYSTEM_PROMPT = """\
당신은 특허 진보성 분석 중 단순설계변경 반박 논리 전문가입니다.
심사관이 본 발명을 D1의 "단순 설계 변경"으로 판단할 가능성이 있는 경우,
그것이 단순 변경이 아니라 비자명한 개선이라는 논리를 작성합니다.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. changed_component_label과 changed_component_name은 각각 사용자 구성요소 중 가장 반박이 필요한 하나의 라벨과 그 이름.
4. non_obviousness는 150~250자.
5. 사용자가 [사용자가 언급한 선행기술 대비 차별점] 섹션을 제공한 경우,
   해당 내용을 비자명성 논거로 활용하세요.

changed_component_label 선정 기준:
- D1과의 차이점이 있는 구성요소 중 하나
- 심사관이 "이건 D1을 조금만 바꾼 것"이라 판단할 가능성이 있는 것

non_obviousness 작성 가이드:
다음 중 하나 이상의 논거를 포함하세요:
1. 통상의 기술자가 D1으로부터 쉽게 도출할 수 없다는 구체적 이유
2. 예상치 못한 효과 또는 특유의 상승효과
3. 기술적 극복 요소 (해결하려는 문제와 접근 방식의 차이)
4. 사용자가 차별점에서 언급한 특유의 접근법 (있는 경우)

단순히 "설계 변경 아니다" 주장 말고, 위 논거 중 최소 하나 이상 구체적으로 기술하세요.

출력 형식:
{
  "changed_component_label": "C",
  "changed_component_name": "...",
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
{differentiation_info}

위 구성요소 중 심사관이 D1의 "단순 설계 변경"으로 판단할 가능성이 높은 하나를 선정하고
그것이 비자명한 개선이라는 논리를 JSON으로 반환하세요.
"""


async def generate_simple_design(
    invention_title: str,
    invention_description: str,
    components: list[InventionComponent],
    primary_art: PriorArtInfo,
    prior_art_reference: Optional[str] = None,
    differentiation_notes: Optional[str] = None,
) -> Optional[SimpleDesignResult]:
    """단순설계변경 비자명성 논리 자동 생성"""

    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    differentiation_text = ""
    if prior_art_reference or differentiation_notes:
        differentiation_text = (
            f"\n[사용자가 언급한 선행기술 대비 차별점]\n"
            f"선행기술: {prior_art_reference or '(없음)'}\n"
            f"차이점: {differentiation_notes or '(없음)'}"
        )

    user_message = SIMPLE_DESIGN_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        components_text=components_text,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
        differentiation_info=differentiation_text,
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