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
from typing import Optional, List, Literal

import httpx
from pydantic import BaseModel, Field, field_validator, ValidationError

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
    """D2 선정 결과."""

    d2_application_number: Optional[str] = Field(
        default=None,
        description="선정된 D2 출원번호. 적합한 D2 없으면 None."
    )

    @field_validator("d2_application_number", mode="before")
    @classmethod
    def normalize_empty(cls, v):
        """빈 문자열이나 'null' 문자열을 None으로 정규화."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() in ("null", "none"):
                return None
        return v


SELECT_SECONDARY_SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 변리사이자 진보성 분석 전문가입니다.
주인용 발명(D1)이 이미 선정된 상태에서, 이와 결합하여 본 발명의 진보성을 
부정할 수 있는 가장 적절한 부인용 발명(D2)을 후보 중에서 선정하세요.

## D2 선정 기준

1. **D1의 부족한 부분 보완**: D1이 개시하지 않은 구성요소를 개시해야 함
2. **결합 가능성**: D1과 결합 가능한 기술 분야여야 함
3. **관점 차별성**: D1과 완전 동일한 관점이면 D2로 부적합 (새 정보 없음)

### 결합 가능성 판단 상세

- **동일/인접 기술 분야**: 결합 가능
  * 예: 얼굴 인식(G06V) + 음성 인식(G10L) — 둘 다 인식 시스템
  * 예: 이미지 분할(G06T) + 이미지 분류(G06V) — 둘 다 컴퓨터 비전
- **응용 도메인 다르지만 알고리즘 공통**: 결합 가능
  * 예: 의료 영상 세그멘테이션 + 자율주행 세그멘테이션 — U-Net 공통
- **완전 무관한 분야**: 결합 불가
  * 예: 이미지 처리 + 화학 촉매 합성
- **판단 기준**: 통상의 기술자가 두 특허를 결합할 동기가 있는가?

## D2 선정 실패 케이스 (중요!)

다음 경우 d2_application_number를 null로 반환:
- 모든 후보가 D1과 매우 유사한 관점 (결합해도 새 정보 없음)
- 모든 후보가 D1이 이미 커버하는 구성요소만 개시
- 발명 도메인과 완전 무관한 후보들만 있음

**억지로 D2를 선정하지 마세요.** 
D2 없이 D1 단독으로 진보성을 분석하는 것이 더 정확합니다.

## 출력 규칙

1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. d2_application_number는 반드시 후보 목록의 출원번호를 그대로 사용 (없으면 null).

## 출력 형식

{
  "d2_application_number": "출원번호 또는 null"
}

## 예시

### 예시 1: 적합한 D2 선정

[사용자 발명]
명칭: 딥러닝 기반 얼굴 감정 및 음성 감정 결합 분석 시스템
구성요소:
A: CNN 얼굴 검출부
B: BERT 음성 임베딩부
C: 감정 융합 분류부

[D1]
출원번호: 1020200011111
명칭: CNN 기반 얼굴 감정 인식
청구항 1: CNN으로 얼굴을 검출하고, FC 레이어로 감정을 분류하는 시스템.

[후보]
- 1020200022222: BERT 기반 음성 감정 인식 (음성만 처리)
- 1020200033333: CNN 기반 이미지 분류 (D1과 유사)
- 1020200044444: 화학 촉매 합성 (무관)

출력:
{
  "d2_application_number": "1020200022222"
}

### 예시 2: 적합한 D2 없음

[사용자 발명]
명칭: U-Net 기반 저조도 이미지 노이즈 제거

[D1]
청구항 1: U-Net으로 이미지 세그멘테이션을 수행하는 시스템.

[후보]
- 1020200055555: FCN 이미지 세그멘테이션 (D1과 매우 유사)
- 1020200066666: CNN 이미지 분류 (D1과 유사 관점)
- 1020200077777: 로봇 매니퓰레이터 제어 (무관)

출력:
{
  "d2_application_number": null
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
적합한 D2가 없다면 d2_application_number를 null로 반환하세요.
"""


async def select_secondary_art(
    invention_title: str,
    invention_description: str,
    components: list[InventionComponent],
    primary_art: PriorArtInfo,
    candidates: list[PriorArtInfo],
) -> Optional[SelectSecondaryResult]:
    """
        D1이 주어졌을 때 후보들 중 D2를 자동 선정.

        Args:
            invention_title: 사용자 발명 명칭
            invention_description: 사용자 발명 설명
            components: 사용자 발명 구성요소 리스트
            primary_art: D1 (이미 선정됨)
            candidates: D2 후보 리스트 (D1 제외)

        Returns:
            SelectSecondaryResult: D2 선정 결과 (또는 None if LLM 호출 실패)
            - d2_application_number가 None이면 D2 없이 D1 단독 분석 진행
        """

    if not candidates:
        logger.warning("[InventiveStep] D2 후보가 없음")
        return SelectSecondaryResult(
            d2_application_number=None
        )

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

    # LLM 호출
    result = await _call_claude_with_json(
        system_prompt=SELECT_SECONDARY_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=SelectSecondaryResult,
        log_prefix="[InventiveStep/SelectSecondary]",
    )

    # 타입 안정성 보장 (BaseModel -> SelectSecondaryResult)
    if not isinstance(result, SelectSecondaryResult):
        logger.error("[InventiveStep/SelectSecondary] LLM 호출 실패 또는 타입 불일치")
        return None

    if result is None:
        # LLM 호출 자체 실패
        logger.error("[InventiveStep/SelectSecondary] LLM 호출 실패")
        return None

    # ============================================================
    # LLM 반환 번호 검증
    # ============================================================
    if result.d2_application_number is not None:
        valid_numbers = {c.application_number for c in candidates}

        if result.d2_application_number not in valid_numbers:
            logger.warning(
                f"[InventiveStep/SelectSecondary] LLM이 후보에 없는 번호 반환: "
                f"'{result.d2_application_number}'. D2 없음으로 처리."
            )
            # 안전하게 D2 없음 처리 (억지 매칭 방지)
            result.d2_application_number = None

    # 로깅
    if result.d2_application_number:
        logger.info(
            f"[InventiveStep/SelectSecondary] D2 선정: "
            f"{result.d2_application_number}"
        )
    else:
        logger.info(
            f"[InventiveStep/SelectSecondary] D2 미선정 (D1 단독 분석). "
        )

    return result

# ============================================================
# 1. 카테고리 자동 선정
# ============================================================

# ============================================================
# 카테고리 타입 정의
# ============================================================

CategoryType = Literal[
    "numerical_limit",
    "combination_motivation",
    "common_technique",
    "simple_design",
]

ALL_CATEGORIES: set[str] = {
    "numerical_limit",
    "combination_motivation",
    "common_technique",
    "simple_design",
}

# D2가 필수인 카테고리 (D2 없으면 자동 제외)
D2_REQUIRED_CATEGORIES: set[str] = {
    "combination_motivation",
}

class SelectCategoriesResult(BaseModel):
    """카테고리 선정 결과."""

    categories: List[CategoryType] = Field(
        max_length=4,
        description="선정된 진보성 논리 카테고리 (0~4개, 빈 경우 fallback 적용)"
    )

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, v):
        """
        카테고리 값 정규화:
        - 앞뒤 공백 제거
        - 소문자 통일
        - 중복 제거 (순서 유지)
        - 유효하지 않은 값 로그로 필터링
        """
        if not isinstance(v, list):
            return v

        seen = set()
        normalized = []
        for item in v:
            if not isinstance(item, str):
                continue
            cleaned = item.strip().lower()
            if cleaned in ALL_CATEGORIES and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
            elif cleaned not in ALL_CATEGORIES:
                logger.warning(
                    f"[SelectCategories] 유효하지 않은 카테고리 무시: '{item}'"
                )
        return normalized


SELECT_CATEGORIES_SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 변리사이자 진보성 분석 전문가입니다.
본 발명과 주인용(D1), 부인용(D2)를 분석하여 진보성 논쟁에서 이슈가 될 만한 
카테고리를 선정하세요.

## 4개 진보성 논리 카테고리

### 1. numerical_limit (수치한정)

**언제 선정?**
- 본 발명 청구항에 수치 범위, 파라미터, 성능 지표가 명시됨
- D1과 비교하여 정량적 개선(%, 배수 등)이 있음
- 사용자가 measurement_conditions, measurement_results를 입력함

**선정하면 안 되는 경우**:
- 순수 알고리즘/구성 발명 (수치 없음)
- D1과의 수치 비교가 불가능

### 2. combination_motivation (복수인용발명결합)

**언제 선정?**
- D1과 D2가 서로 다른 방향의 기술이거나 결합 동기가 없어 보이는 경우
- 통상의 기술자가 D1과 D2를 결합할 이유가 없음
- 사용자가 prior_art_reference, differentiation_notes 입력함

**중요 제약**:
- **D2가 미제공된 경우 절대 선정 금지** (D2 없이 결합 논의 불가)

### 3. common_technique (주지관용기술)

**언제 선정?**
- D1에 없는 본 발명의 구성요소 중 관용기술로 오해받을 만한 것이 있음
- 하지만 실제로는 특별한 기능·효과가 있는 구성요소
- 심사관이 "주지관용기술"이라 부정할 여지가 있는 경우

### 4. simple_design (단순설계변경)

**언제 선정?**
- D1과 본 발명의 차이가 미묘하거나 파라미터 조정으로 보일 위험이 있음
- 하지만 실제로는 비자명한 개선이거나 예상치 못한 효과 있음
- 심사관이 "단순설계변경"이라 부정할 여지가 있는 경우

## 사용자 입력 → 카테고리 매핑

- **measurement_conditions, measurement_results 있음**:
  → numerical_limit 우선순위 높음
- **prior_art_reference, differentiation_notes 있음**:
  → combination_motivation 또는 common_technique 참고

## 선정 규칙

1. **관련성이 명확한 카테고리만 선정**. 억지로 4개 다 넣지 마세요.
2. **최소 1개, 최대 4개**. 보통 1~3개가 적절.
3. **D2가 미제공된 경우**: combination_motivation은 절대 선정 금지.
   나머지 3개(numerical_limit, common_technique, simple_design)만 검토.
4. 사용자 입력 정보가 있다면 관련 카테고리 우선.

## 출력 규칙

1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. categories 배열에 정확한 카테고리 이름 사용:
   "numerical_limit" | "combination_motivation" | "common_technique" | "simple_design"

## 출력 형식

{
  "categories": ["numerical_limit", "combination_motivation"]
}

## 예시

### 예시 1: 수치한정 + 결합 동기 (D2 있음)

[사용자 발명]
명칭: 딥러닝 기반 멀티모달 감정 분석 시스템
설명: CNN 얼굴 인식 + BERT 텍스트 분석 결합으로 정확도 94% 달성 
     (D1 대비 7% 향상)
측정 조건: FER2013 데이터셋
측정 결과: 정확도 94% (D1: 87%)

[D1]: CNN 얼굴 감정 인식 (정확도 87%)
[D2]: BERT 텍스트 감정 인식 (다른 도메인)

출력:
{
  "categories": ["numerical_limit", "combination_motivation"]
}

### 예시 2: D2 없음, 수치한정 + 단순설계변경 방어

[사용자 발명]
명칭: 저조도 이미지 노이즈 제거 시스템
설명: U-Net 아키텍처 개선으로 노이즈 제거 정확도 95% 달성 (D1 대비 10% 향상)
측정 결과: PSNR 32.5dB (D1: 29.8dB)

[D1]: 기존 U-Net 노이즈 제거 (PSNR 29.8dB)
[D2]: 없음 (선정 실패)

출력:
{
  "categories": ["numerical_limit", "simple_design"]
}

### 예시 3: 순수 구성 발명, 결합 동기만

[사용자 발명]
명칭: Transformer + GNN 결합 그래프 학습 시스템
설명: Transformer의 어텐션 메커니즘과 GNN의 그래프 임베딩을 결합
(수치 데이터 없음)

[D1]: Transformer 기반 자연어 처리
[D2]: GNN 기반 그래프 분석 (완전 다른 도메인)

출력:
{
  "categories": ["combination_motivation", "common_technique"]
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
{d2_section}
{additional_info_text}

위 정보를 바탕으로 진보성 논리 카테고리를 선정하여 JSON으로 반환하세요.
"""


async def select_relevant_categories(
        invention_title: str,
        invention_description: str,
        components: list[InventionComponent],
        primary_art: PriorArtInfo,
        secondary_art: Optional[PriorArtInfo] = None,
        prior_art_reference: Optional[str] = None,
        differentiation_notes: Optional[str] = None,
        measurement_conditions: Optional[str] = None,
        measurement_results: Optional[str] = None,
) -> Optional[SelectCategoriesResult]:
    """
        4개 진보성 논리 카테고리 중 관련 있는 것들 자동 선정.

        Args:
            invention_title: 사용자 발명 명칭
            invention_description: 사용자 발명 설명
            components: 사용자 발명 구성요소 리스트
            primary_art: D1 (주인용, 필수)
            secondary_art: D2 (부인용, 없을 수 있음)
            prior_art_reference: 사용자가 언급한 유사 선행기술 (선택)
            differentiation_notes: 사용자가 명시한 차이점 (선택)
            measurement_conditions: 측정 조건 (선택)
            measurement_results: 측정 결과 (선택)

        Returns:
            SelectCategoriesResult 또는 None (LLM 호출 실패 시)

        Note:
            - D2가 None이면 combination_motivation 카테고리 자동 제외
            - 유효하지 않은 카테고리 값은 자동 필터링
        """

    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    # ============================================================
    # D2 섹션 조립 (있음/없음 분기)
    # ============================================================
    if secondary_art is not None:
        d2_section = (
            f"명칭: {secondary_art.title}\n"
            f"초록:\n{secondary_art.abstract or '(정보 없음)'}\n"
            f"독립 청구항:\n{secondary_art.claims_independent}"
        )
    else:
        d2_section = (
            "(D2 미선정 - D1 단독 분석 모드)\n"
            "**중요**: combination_motivation 카테고리는 D2 필수이므로 선정 금지."
        )


    # ============================================================
    # 사용자 추가 정보 조립
    # ============================================================

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
        d2_section=d2_section,
        additional_info_text=additional_info_text,
    )

    result = await _call_claude_with_json(
        system_prompt=SELECT_CATEGORIES_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=SelectCategoriesResult,
        log_prefix="[InventiveStep/SelectCategories]",
    )

    if result is None:
        logger.error("[InventiveStep/SelectCategories] LLM 호출 실패")
        return None

    # 타입 안정성 보장
    if not isinstance(result, SelectCategoriesResult):
        logger.error("[InventiveStep/SelectCategoriesResult] LLM 호출 실패 또는 타입 불일치")
        return None

    # ============================================================
    # D2 없으면 COMBINATION_MOTIVATION 강제 제외 (방어 코드)
    # ============================================================
    if secondary_art is None:
        original_categories = list(result.categories)
        result.categories = [
            c for c in result.categories
            if c not in D2_REQUIRED_CATEGORIES
        ]

        removed = set(original_categories) - set(result.categories)
        if removed:
            logger.warning(
                f"[SelectCategories] D2 없어서 자동 제외된 카테고리: {removed}. "
                f"프롬프트 지시 위반. 프롬프트 개선 필요."
            )

    # ============================================================
    # 카테고리 최소 1개 보장
    # ============================================================
    if not result.categories:
        logger.warning(
            "[SelectCategories] 카테고리 선정 결과가 비어있음. "
            "fallback: simple_design 사용."
        )
        # Fallback: simple_design은 D1만 있어도 항상 적용 가능
        result.categories = ["simple_design"]

    # ============================================================
    # 로깅
    # ============================================================
    logger.info(
        f"[SelectCategories] 선정 완료: {result.categories}"
    )

    return result

# ============================================================
# 2. 수치한정 (효과의 현저성)
# ============================================================

class EffectItem(BaseModel):
    """발명의 효과 항목 1개"""
    metric: str = Field(
        min_length=1,
        max_length=100,
        description="측정 지표 (예: 정확도, VOC 배출량)"
    )
    unit: str = Field(
        max_length=20,
        description="단위 (예: %, fps, g/L). 무단위 지표는 빈 문자열."
    )
    prior_art_value: str = Field(
        min_length=1,
        max_length=50,
        description="종래기술 수치 (문자열, 범위/근사값 허용)"
    )
    invention_value: str = Field(
        min_length=1,
        max_length=50,
        description="본 발명 수치 (문자열, 범위/근사값 허용)"
    )
    improvement: str = Field(
        min_length=1,
        max_length=50,
        description="개선률 (예: 97.5% 감소, 5배 증가)"
    )

    @field_validator("metric", "prior_art_value", "invention_value", "improvement")
    @classmethod
    def strip_whitespace(cls, v):
        """앞뒤 공백 제거."""
        return v.strip() if isinstance(v, str) else v


class NumericalLimitResult(BaseModel):
    """수치한정 논리 결과."""

    effect_items: List[EffectItem] = Field(
        default_factory=list,
        max_length=10,
        description="발명의 효과 표 항목들 (0~10개, 명시된 수치가 있는 것만)"
    )


NUMERICAL_LIMIT_SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 변리사이자 수치한정 발명의 효과 분석 전문가입니다.
본 발명이 종래기술(D1) 대비 어떤 수치적 효과를 가지는지 표 형태로 정리하세요.

## 출력 규칙

1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. 명시된 수치만 사용:
   - 본 발명 설명, D1 청구항/초록, 사용자 제공 실험 데이터에 명시된 수치만 사용
   - 명시되지 않은 수치를 추정하거나 창작 금지
4. **사용자 제공 실험 데이터가 있으면 최우선 활용**.
5. 명시된 수치가 있는 항목만 반환. 개수 채우려 무리하게 만들지 말 것.
6. `improvement`는 두 수치가 모두 있고 계산 가능할 때만.

## 반드시 지킬 것 (부정 명령)

- 청구항이나 명세서에 없는 수치를 추측 금지
- D1 청구항에 수치가 없으면 그 항목은 반환하지 말 것
- 애매한 표현("좋아짐", "빠름")만 있으면 항목 자체를 만들지 말 것

## improvement 계산 공식

### 감소 지표 (낮을수록 좋음)
- 예: 오차율, 배출량, 처리 시간, 손실률
- 공식: `((prior - invention) / prior) × 100`
- 표기: "N% 감소" 또는 "N배 감소"
- 예: 320 → 8 → 97.5% 감소

### 증가 지표 (높을수록 좋음)
- 예: 정확도, 회수율, 처리 속도, F1 스코어
- 공식: `((invention - prior) / prior) × 100`
- 표기: "N% 향상" 또는 "N배 증가"
- 예: 87 → 94 → 8.05% 향상

### 배수 표현
- 변화폭이 큰 경우 (2배 이상)
- 예: 20fps → 100fps → "5배 증가"

## 각 필드 가이드

- **metric**: 측정 지표 명 (본문에 명시된 것 우선)
  * 예: "정확도", "처리 속도", "VOC 배출량", "회수율"
  * 명시되지 않으면 "성능", "품질" 같은 일반 지표 사용 금지

- **unit**: 단위 (본문에서 확인)
  * 표준 기호 사용: "%", "fps", "ms", "MB", "g/L", "℃"
  * 무단위 지표는 빈 문자열 허용

- **prior_art_value**: 종래기술 수치 (D1에 명시된 값)
  * 순수 숫자 문자열 우선: "320"
  * 범위인 경우 대표값: "1.0~1.5" → "1.25" 또는 "1.0"
  * 근사값 허용: "약 320"

- **invention_value**: 본 발명 수치 (본 발명 설명 또는 사용자 데이터에 명시된 값)
  * 동일한 형식

- **improvement**: 개선률 (위 공식으로 계산)
  * 두 수치가 모두 명확할 때만 계산
  * 계산 불가 시 해당 항목 자체를 반환하지 말 것

## 출력 형식

```
{
  "effect_items": [
    {
      "metric": "지표명",
      "unit": "단위",
      "prior_art_value": "숫자",
      "invention_value": "숫자",
      "improvement": "개선률"
    }
  ]
}
```

명시된 수치가 하나도 없으면:
```
{
  "effect_items": []
}
```
(빈 배열도 정상 응답.)

## 예시

### 예시 1: 소프트웨어 발명 - 딥러닝 모델 (G06N)

[사용자 발명]
설명: BERT + LSTM 결합으로 감정 분석 정확도 94%, F1 스코어 0.91 달성
측정 결과: 정확도 94%, F1 0.91, 처리 속도 25fps (RTX 3090 기준)

[D1]
청구항: CNN 기반 감정 분석 (정확도 87%, F1 0.83, 처리 속도 15fps)

출력:
{
  "effect_items": [
    {
      "metric": "감정 분류 정확도",
      "unit": "%",
      "prior_art_value": "87",
      "invention_value": "94",
      "improvement": "8.05% 향상"
    },
    {
      "metric": "F1 스코어",
      "unit": "",
      "prior_art_value": "0.83",
      "invention_value": "0.91",
      "improvement": "9.64% 향상"
    },
    {
      "metric": "실시간 처리 속도",
      "unit": "fps",
      "prior_art_value": "15",
      "invention_value": "25",
      "improvement": "66.7% 향상"
    }
  ]
}

### 예시 2: 소프트웨어 발명 - 이미지 처리 (G06T)

[사용자 발명]
설명: U-Net 개선 아키텍처로 저조도 이미지 PSNR 32.5dB 달성
측정 결과: PSNR 32.5dB, SSIM 0.94, 처리 시간 45ms

[D1]
청구항: 기존 U-Net 노이즈 제거 (PSNR 29.8dB, SSIM 0.89, 처리 시간 80ms)

출력:
{
  "effect_items": [
    {
      "metric": "PSNR (화질)",
      "unit": "dB",
      "prior_art_value": "29.8",
      "invention_value": "32.5",
      "improvement": "9.06% 향상"
    },
    {
      "metric": "SSIM (구조 유사도)",
      "unit": "",
      "prior_art_value": "0.89",
      "invention_value": "0.94",
      "improvement": "5.62% 향상"
    },
    {
      "metric": "처리 시간",
      "unit": "ms",
      "prior_art_value": "80",
      "invention_value": "45",
      "improvement": "43.75% 감소"
    }
  ]
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
명시된 수치가 없다면 빈 배열을 반환하세요.
"""


async def generate_numerical_limit(
    invention_title: str,
    invention_description: str,
    primary_art: PriorArtInfo,
    measurement_conditions: Optional[str] = None,
    measurement_results: Optional[str] = None,
) -> Optional[NumericalLimitResult]:
    """
        수치한정 논리 (발명의 효과 표) 자동 생성.

        Args:
            invention_title: 사용자 발명 명칭
            invention_description: 사용자 발명 설명
            primary_art: D1 (주인용, 수치 비교 대상)
            measurement_conditions: 사용자 제공 측정 조건 (선택, 있으면 우선 활용)
            measurement_results: 사용자 제공 측정 결과 (선택, 있으면 우선 활용)

        Returns:
            NumericalLimitResult 또는 None (LLM 호출 실패 시)
            effect_items가 빈 배열이어도 정상 응답 (수치 데이터 부재)

        Note:
            - D1 청구항에는 수치가 없는 경우가 많음
            - 사용자 measurement_* 입력이 강력한 근거
            - 빈 결과 시 프론트가 사용자에게 직접 입력 유도
        """

    # ============================================================
    # 사용자 데이터 확인
    # ============================================================
    has_user_data = bool(measurement_conditions or measurement_results)

    if not has_user_data:
        logger.warning(
            "[NumericalLimit] 사용자 측정 데이터 없음. "
            "D1 청구항에도 수치가 없다면 빈 결과 반환 예상. "
            "이 카테고리 선정 자체를 재검토 필요."
        )

    # ============================================================
    # 사용자 실험 데이터 조립
    # ============================================================
    measurement_text = ""
    if has_user_data:
        measurement_text = (
            f"\n[사용자가 제공한 실험 데이터]\n"
            f"측정 조건: {measurement_conditions or '(없음)'}\n"
            f"측정 결과: {measurement_results or '(없음)'}\n"
            f"\n**중요: 위 사용자 제공 데이터의 수치를 최우선 활용하세요.**"
        )

    user_message = NUMERICAL_LIMIT_USER_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        d1_title=primary_art.title,
        d1_abstract=primary_art.abstract or "(정보 없음)",
        d1_claims=primary_art.claims_independent,
        measurement_info=measurement_text,
    )

    result = await _call_claude_with_json(
        system_prompt=NUMERICAL_LIMIT_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=NumericalLimitResult,
        log_prefix="[InventiveStep/NumericalLimit]",
    )

    if result is None:
        logger.error("[NumericalLimit] LLM 호출 실패")
        return None

    # 타입 안정성 보장
    if not isinstance(result, NumericalLimitResult):
        logger.error("[InventiveStep/NumericalLimitResult] LLM 호출 실패 또는 타입 불일치")
        return None

    # ============================================================
    # 결과 로깅
    # ============================================================
    item_count = len(result.effect_items)

    if item_count == 0:
        logger.info(
            f"[NumericalLimit] 빈 결과: 명시된 수치 없음. "
            f"user_data={has_user_data}. "
        )
    else:
        logger.info(
            f"[NumericalLimit] 생성 완료: {item_count}개 항목, "
            f"user_data={has_user_data}"
        )
        # 첫 번째 항목만 디버그 로그
        first = result.effect_items[0]
        logger.debug(
            f"[NumericalLimit] 첫 항목 예시: "
            f"{first.metric} ({first.prior_art_value}{first.unit} → "
            f"{first.invention_value}{first.unit}, {first.improvement})"
        )

    return result


# ============================================================
# 3. 복수인용발명결합 (Teaching Away)
# ============================================================

class CombinationMotivationResult(BaseModel):
    """결합 동기 부재 논리 결과."""

    background_limit: str = Field(
        min_length=50,
        max_length=300,
        description="배경기술의 한계 (100~200자 권장, 최대 300자)"
    )
    teaching_away: str = Field(
        min_length=50,
        max_length=300,
        description="결합 동기의 부재 (100~200자 권장, 최대 300자)"
    )

    @field_validator("background_limit", "teaching_away")
    @classmethod
    def strip_whitespace(cls, v):
        """앞뒤 공백 제거."""
        return v.strip() if isinstance(v, str) else v


COMBINATION_MOTIVATION_SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 변리사이자 진보성 분석 전문가입니다.
D1과 D2를 결합하여 본 발명에 도달할 동기가 없음을 논증하세요.
이는 한국 특허법 실무상 "복수인용발명 결합의 곤란성" 논리입니다.

## 결합 동기 부재의 두 가지 논리

### 1. Teaching Away (명시적 반대 가르침) - 강한 논리
- D1 또는 D2가 특정 방식을 명시적으로 부정/배척
- 통상의 기술자가 본 발명 방향으로 갈 이유가 없음
- 예: D1이 "정확도를 위해 깊은 네트워크 필수" → 본 발명 "얕은 네트워크로 고정확도"

### 2. 결합 동기 부재 (일반 논리)
- D1과 D2가 상반된 방향을 지향
- 결합 시 시너지 없거나 성능 저하 예상
- 예: D1은 정확도 우선(느림), D2는 속도 우선(부정확) → 결합 시 목적 상반

## 출력 규칙

1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. 각 필드는 100~200자.
4. 사용자가 [차별점] 섹션을 제공한 경우 해당 내용을 명시적으로 활용.

## 각 필드 작성 가이드

### background_limit (100~200자)

**목적**: 종래기술이 가진 근본적 한계 + D1/D2가 각자 이를 해결 못함을 서술.

**구조**:
1. 산업 전반의 미해결 문제 서술
2. D1의 접근 한계 지적
3. D2의 접근 한계 지적
4. (선택) 본 발명이 필요한 이유 함축

**서술 톤**: 
- "종래 [분야]는 [문제]를 안고 있으며, 
   D1은 [D1 접근]으로 [D1 한계], 
   D2는 [D2 접근]으로 [D2 한계]..."

### teaching_away (100~200자)

**목적**: D1과 D2가 서로 다른 방향으로 가르쳐, 결합 동기 부재를 논증.

**서술 방식**:
- D1의 방향성 명시
- D2의 방향성 명시
- 두 방향이 상충됨을 지적
- 결합 시 성능 저하 또는 목적 상반 예상

**서술 톤**:
- "D1은 [방식 A]를 지향하고 D2는 [방식 B]를 지향하나, 
   두 방식은 [상충 이유]로 결합 동기가 부재..."

## 사용자 입력 활용

- **prior_art_reference**: 사용자가 언급한 유사 선행기술
  → 이 선행기술의 한계를 background_limit에 반영

- **differentiation_notes**: 사용자가 명시한 차이점
  → 이 차별점이 D1/D2가 갈 수 없는 방향임을 teaching_away에 반영

## 반드시 지킬 것 (부정 명령)

- D1, D2 초록/청구항에 없는 내용을 창작 금지
- 억지 결합 부재 논리를 만들지 말 것 (실제로 결합 가능하면 정직하게 서술)
- 100자 미만은 근거 부족, 200자 초과는 장황함

## 출력 형식

```
{
  "background_limit": "종래기술 한계 및 D1/D2 각각의 접근 한계 (100~200자)",
  "teaching_away": "D1과 D2 결합 동기 부재 논증 (100~200자)"
}
```

## 예시

### 예시 1: 상반된 방향 (소프트웨어, G06N/G06T)

[사용자 발명]
CNN + LSTM 결합으로 실시간 감정 인식 (30fps + 정확도 94%)

[D1]
청구항: CNN 기반 정지 이미지 감정 인식 (정확도 88%, 처리 시간 200ms)
초록: 정확도를 위해 깊은 네트워크 사용

[D2]
청구항: LSTM 기반 텍스트 감정 인식 (시퀀스 분석 특화)
초록: 텍스트 시퀀스 처리 특화

출력:
{
  "background_limit": "종래 감정 인식은 정확도와 실시간성이 상충 관계에 있어 두 조건 동시 만족이 곤란. D1은 정확도 향상에 집중하여 처리 시간이 200ms에 달해 실시간 응용 곤란하며, D2는 텍스트 시퀀스에 특화되어 이미지 도메인에는 직접 적용 불가한 근본적 한계 존재.",
  "teaching_away": "D1은 정지 이미지 기반 심층 CNN을 지향하여 실시간성을 명시적으로 포기하는 방향이며, D2는 텍스트 시퀀스 처리 특화 아키텍처로 이미지 도메인에 대한 결합 동기 부재. 두 특허 결합해도 시간 축 이미지 처리라는 본 발명 목표와 상반된 방향으로 가르치고 있음."
}

### 예시 2: 도메인 불일치 (소프트웨어)

[사용자 발명]
멀티모달 음성-이미지 감정 인식 (BERT + CNN 결합)

[D1]
청구항: BERT 기반 음성 인식 (음성 도메인 특화)

[D2]
청구항: CNN 기반 얼굴 인식 (이미지 도메인 특화)

출력:
{
  "background_limit": "종래 감정 인식은 단일 모달리티(음성 또는 이미지)에 국한되어 종합적 감정 판단이 어려웠음. D1은 음성 특징 추출에 최적화되어 시각 정보를 다룰 수 없고, D2는 이미지 얼굴 분석에 특화되어 음성 신호를 처리할 수 없는 각자의 도메인 한계 보유.",
  "teaching_away": "D1은 음성 언어 처리, D2는 얼굴 이미지 분석으로 상이한 모달리티에 특화되어, 통상의 기술자가 두 특허를 결합할 기술적 동기가 부재. 결합 시 각 모델의 특화 성능을 오히려 저해할 우려가 있어 상반된 방향으로 가르치고 있음."
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
    secondary_art: Optional[PriorArtInfo],
    prior_art_reference: Optional[str] = None,
    differentiation_notes: Optional[str] = None,
) -> Optional[CombinationMotivationResult]:
    """
    복수인용발명결합 (Teaching Away) 논리 자동 생성.

    Args:
        invention_title: 사용자 발명 명칭
        invention_description: 사용자 발명 설명
        primary_art: D1 (주인용, 필수)
        secondary_art: D2 (부인용, 이 카테고리는 D2 필수)
        prior_art_reference: 사용자가 언급한 유사 선행기술 (선택)
        differentiation_notes: 사용자가 명시한 차이점 (선택)

    Returns:
        CombinationMotivationResult 또는 None (LLM 호출 실패 or D2 없음)

    Note:
        이 카테고리는 D2가 필수입니다.
        select_categories에서 D2 없으면 이 카테고리 자동 제외되어야 함.
        여기서는 방어 코드로 이중 체크.
    """

    # ============================================================
    # D2 필수 방어 (앞서 select_categories에서 걸러졌어야 함)
    # ============================================================
    if secondary_art is None:
        logger.error(
            "[CombinationMotivation] D2 없이 호출됨. "
            "이 카테고리는 D2 필수. select_categories 로직 재검토 필요."
        )
        return None

    # ============================================================
    # 사용자 차별점 정보 조립
    # ============================================================
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

    result = await _call_claude_with_json(
        system_prompt=COMBINATION_MOTIVATION_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=CombinationMotivationResult,
        log_prefix="[InventiveStep/CombinationMotivation]",
    )

    if result is None:
        logger.error("[CombinationMotivation] LLM 호출 실패")
        return None

    # 타입 안정성 보장
    if not isinstance(result, CombinationMotivationResult):
        logger.error("[InventiveStep/CombinationMotivationResult] LLM 호출 실패 또는 타입 불일치")
        return None

    # ============================================================
    # 결과 로깅
    # ============================================================
    logger.info(
        f"[CombinationMotivation] 생성 완료: "
        f"background={len(result.background_limit)}자, "
        f"teaching_away={len(result.teaching_away)}자, "
        f"user_input={bool(prior_art_reference or differentiation_notes)}"
    )

    # 길이 경고 (100자 미만 또는 200자 초과)
    for field_name in ["background_limit", "teaching_away"]:
        text = getattr(result, field_name)
        text_len = len(text)
        if text_len < 100:
            logger.warning(
                f"[CombinationMotivation] {field_name} 짧음: {text_len}자 (권장 100~200)"
            )
        elif text_len > 200:
            logger.debug(
                f"[CombinationMotivation] {field_name} 길이: {text_len}자 (권장 100~200)"
            )

    return result



# ============================================================
# 4. 주지관용기술 반박
# ============================================================

class CommonTechniqueResult(BaseModel):
    """주지관용기술 반박 논리 결과."""

    target_label: str = Field(
        min_length=1,
        max_length=5,
        description="주지관용기술로 지목되는 구성요소 라벨 (A/B/C/...)"
    )
    target_name: str = Field(
        min_length=1,
        max_length=100,
        description="주지관용기술로 지목되는 구성요소 이름"
    )
    rebuttal: str = Field(
        min_length=50,
        max_length=350,
        description="반박 논리 (150~250자 권장)"
    )

    @field_validator("target_label", "target_name", "rebuttal")
    @classmethod
    def strip_whitespace(cls, v):
        """앞뒤 공백 제거."""
        return v.strip() if isinstance(v, str) else v


COMMON_TECHNIQUE_SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 변리사이자 주지관용기술 반박 논리 전문가입니다.
심사관이 본 발명의 특정 구성요소를 "주지관용기술"로 판단할 가능성이 있는 경우,
그것이 주지관용기술이 아니라는 반박 논리를 작성합니다.
 
## 출력 규칙
 
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. **target_label**과 **target_name**은 사용자 구성요소 중 반박이 가장 필요한 하나의 라벨과 그 이름.
4. **rebuttal**은 150~250자.
5. 사용자가 [차별점] 섹션을 제공한 경우 반박 논거에 명시적으로 반영.
 
## target_label 선정 기준
 
### 우선순위 1 (강한 반박 논리)
- D1에 개시되지 않은 새 구성요소
- 심사관이 "이 기술 분야의 관용기술"이라 판단할 가능성 큼
- 예: BERT, GAN, Transformer 같은 널리 알려진 아키텍처
 
### 우선순위 2 (부분적 반박 논리)
- D1에 있어도 심사관이 "관용 방식"이라 폄하할 가능성
- 예: CNN, LSTM 같은 기본 아키텍처
 
**우선순위 1을 우선 선정**. 없으면 2 검토.
 
## rebuttal 작성 가이드
 
다음 중 **최소 하나 이상의 논거**를 구체적으로 서술 (단순히 "관용기술 아니다" 주장 금지):
 
### 논거 1: 새로 도입된 기술적 특징
- 해당 구성요소가 D1에 없음
- 본 발명이 처음 도입한 것임을 강조
 
### 논거 2: 특별한 기능·효과
- 이 구성요소의 특유 효과 명시
- 사용자 발명 설명, 차별점 참조
 
### 논거 3: 관용기술과의 차별성
- 이 기술 분야의 통상적 관용기술만으로는 도출 어려움
- 특유의 응용 방식이나 조건 강조
 
### 논거 4: 다른 구성요소와의 상승효과
- 본 발명의 다른 구성요소(A, B, C 등)와 결합
- 특유의 상승효과 (synergy) 도출
 
**논거 4가 가장 강력**: 개별 구성요소로만 보지 말고 결합 효과 강조.
 
## 사용자 입력 활용
 
- **prior_art_reference**: 사용자가 언급한 유사 선행기술
  → 반박 논거 3, 4에 반영 ("논문 X에서 유사 시도했으나 본 발명은...")
 
- **differentiation_notes**: 사용자가 명시한 차이점
  → 반박 논거 2에 반영 (특유 효과 근거)
 
## 반드시 지킬 것 (부정 명령)
 
- target_label은 반드시 실제 사용자 구성요소 라벨 중 하나 (A/B/C/D...)
- target_name은 해당 구성요소의 실제 이름 그대로 사용
- 발명 설명에 없는 효과나 특징을 창작 금지
- 150자 미만은 근거 부족, 250자 초과는 장황함
 
## 출력 형식
 
```
{
  "target_label": "B",
  "target_name": "정확한 구성요소 이름",
  "rebuttal": "반박 논리 (150~250자)"
}
```
 
## 예시
 
### 예시 1: 새 구성요소 반박 + 상승효과 (G06N, 소프트웨어)
 
[사용자 발명]
명칭: 딥러닝 기반 실시간 얼굴 감정 인식
구성요소:
A. CNN 얼굴 검출부: CNN으로 얼굴 영역 검출
B. BERT 임베딩 모듈: 얼굴 특징을 BERT로 시맥락 표현
C. 감정 분류부: 시맥락 특징으로 감정 5개 카테고리 분류
 
[D1]
청구항: CNN 기반 얼굴 인식 (FC 레이어 사용, 정지 이미지)
 
출력:
{
  "target_label": "B",
  "target_name": "BERT 임베딩 모듈",
  "rebuttal": "BERT 임베딩 모듈은 D1에 개시되지 않은 새 구성요소로, 심사관이 자연어 처리 분야의 관용기술이라 판단할 여지가 있음. 그러나 본 발명은 텍스트가 아닌 얼굴 특징에 BERT 어텐션 메커니즘을 적용하여 시간 축의 표정 변화 맥락을 포착하는 특유의 효과가 있으며, CNN(A)의 정지 이미지 검출과 결합하여 D1의 단일 프레임 분류로는 도출 곤란한 시맥락 감정 분석을 실현함."
}
 
### 예시 2: GAN 특유 결합 효과 (G06T)
 
[사용자 발명]
명칭: 저조도 이미지 노이즈 제거 시스템
구성요소:
A. U-Net 노이즈 검출 네트워크: 저조도 이미지에서 노이즈 영역 세그멘테이션
B. GAN 기반 복원 모듈: 검출된 영역에 특화된 GAN으로 자연스러운 복원
C. 실시간 프레임 출력부: 30fps 이상 출력
 
[D1]
청구항: 기존 U-Net 노이즈 제거 (인페인팅 사용, GAN 미사용)
 
출력:
{
  "target_label": "B",
  "target_name": "GAN 기반 복원 모듈",
  "rebuttal": "GAN 기반 복원 모듈은 D1에 개시되지 않은 새 구성요소로, GAN 자체가 생성 모델 분야에서 알려져 있어 심사관이 관용기술이라 판단할 여지가 있음. 그러나 본 발명은 U-Net(A)이 세그멘테이션한 노이즈 영역에 특화된 조건부 GAN 학습을 적용하여 저조도 특유의 텍스처를 자연스럽게 복원하며, 통상의 인페인팅으로는 도출 곤란한 상승효과가 있음."
}
 
### 예시 3: 특유 응용 방식 (G06Q, 비즈니스)
 
[사용자 발명]
명칭: 시청 이력 기반 개인화 콘텐츠 추천
구성요소:
A. 시청 이력 수집부: 사용자별 시청 데이터 수집
B. Transformer 임베딩부: 시청 시퀀스를 어텐션으로 임베딩
C. 추천 랭킹부: 유사 사용자 클러스터 기반 추천
 
[D1]
청구항: 협업 필터링 기반 추천 (행렬 분해 사용, Transformer 미사용)
 
출력:
{
  "target_label": "B",
  "target_name": "Transformer 임베딩부",
  "rebuttal": "Transformer 임베딩부는 D1에 개시되지 않은 새 구성요소로, Transformer가 NLP 분야 관용기술로 알려져 있어 심사관이 관용기술이라 판단할 여지가 있음. 그러나 본 발명은 텍스트 시퀀스가 아닌 시청 이력의 시간적 순서 정보에 자기 어텐션 메커니즘을 적용하여 사용자 관심 변화의 장기 의존성을 포착하는 특유의 효과가 있어, D1의 정적 행렬 분해와 차별됨."
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
 
위 구성요소 중 심사관이 "주지관용기술"로 판단할 가능성이 높은 하나를 선정하고, 
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
    """
    주지관용기술 반박 논리 자동 생성.

    Args:
        invention_title: 사용자 발명 명칭
        invention_description: 사용자 발명 설명
        components: 사용자 발명 구성요소 리스트
        primary_art: D1 (주인용, 필수)
        prior_art_reference: 사용자가 언급한 유사 선행기술 (선택)
        differentiation_notes: 사용자가 명시한 차이점 (선택)

    Returns:
        CommonTechniqueResult 또는 None (LLM 호출 실패 or 구성요소 없음)

    Note:
        - D2 불필요 (D1 단독으로 논리 성립)
        - target_label 검증 후 존재하지 않으면 첫 구성요소로 fallback
        - target_name은 실제 구성요소 이름으로 정정
    """

    # ============================================================
    # 구성요소 필수 확인
    # ============================================================
    if not components:
        logger.error("[CommonTechnique] 구성요소가 없음. 반박 논리 생성 불가.")
        return None

    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    # 라벨 → 컴포넌트 매핑 (사후 검증용)
    label_to_component = {c.label: c for c in components}

    # ============================================================
    # 사용자 차별점 정보 조립
    # ============================================================
    differentiation_text = ""
    if prior_art_reference or differentiation_notes:
        differentiation_text = (
            f"\n[사용자가 언급한 선행기술 대비 차별점]\n"
            f"선행기술: {prior_art_reference or '(없음)'}\n"
            f"차이점: {differentiation_notes or '(없음)'}\n"
            f"\n**중요: 위 사용자 제공 차별점을 반박 논거에 반영하세요.**"
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

    result = await _call_claude_with_json(
        system_prompt=COMMON_TECHNIQUE_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=CommonTechniqueResult,
        log_prefix="[InventiveStep/CommonTechnique]",
    )

    if result is None:
        logger.error("[CommonTechnique] LLM 호출 실패")
        return None

    # 타입 안정성 보장
    if not isinstance(result, CommonTechniqueResult):
        logger.error("[InventiveStep/CommonTechniqueResult] LLM 호출 실패 또는 타입 불일치")
        return None

    # ============================================================
    # target_label 검증 (사용자 구성요소에 존재하는지)
    # ============================================================
    if result.target_label not in label_to_component:
        logger.warning(
            f"[CommonTechnique] LLM이 존재하지 않는 라벨 반환: "
            f"'{result.target_label}'. 첫 구성요소로 fallback."
        )
        # Fallback: 첫 구성요소로 대체 (rebuttal은 유지)
        fallback = components[0]
        result.target_label = fallback.label
        result.target_name = fallback.name
    else:
        # ============================================================
        # target_name 정정 (실제 구성요소 이름으로)
        # ============================================================
        actual_component = label_to_component[result.target_label]
        if result.target_name != actual_component.name:
            logger.info(
                f"[CommonTechnique] target_name 정정: "
                f"'{result.target_name}' → '{actual_component.name}'"
            )
            result.target_name = actual_component.name

    # ============================================================
    # rebuttal 길이 로깅
    # ============================================================
    rebuttal_len = len(result.rebuttal)
    if rebuttal_len < 150:
        logger.warning(
            f"[CommonTechnique] rebuttal 짧음: {rebuttal_len}자 (권장 150~250)"
        )
    elif rebuttal_len > 250:
        logger.debug(
            f"[CommonTechnique] rebuttal 길이: {rebuttal_len}자 (권장 150~250)"
        )

    logger.info(
        f"[CommonTechnique] 생성 완료: "
        f"target={result.target_label}({result.target_name}), "
        f"rebuttal={rebuttal_len}자, "
        f"user_input={bool(prior_art_reference or differentiation_notes)}"
    )

    return result

# ============================================================
# 5. 단순설계변경 (비자명성 논리)
# ============================================================

class SimpleDesignResult(BaseModel):
    """단순설계변경 반박 논리 결과."""

    changed_component_label: str = Field(
        min_length=1,
        max_length=5,
        description="변경된 구성요소 라벨 (A/B/C/...)"
    )
    changed_component_name: str = Field(
        min_length=1,
        max_length=100,
        description="변경된 구성요소 이름"
    )
    non_obviousness: str = Field(
        min_length=100,
        max_length=350,
        description="단순 설계 변경이 아니라는 논리 (150~250자 권장)"
    )

    @field_validator("changed_component_label", "changed_component_name", "non_obviousness")
    @classmethod
    def strip_whitespace(cls, v):
        """앞뒤 공백 제거."""
        return v.strip() if isinstance(v, str) else v


SIMPLE_DESIGN_SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 변리사이자 단순설계변경 반박 논리 전문가입니다.
심사관이 본 발명을 D1의 "단순 설계 변경"으로 판단할 가능성이 있는 경우,
그것이 단순 변경이 아니라 비자명한 개선임을 논증합니다.

## COMMON_TECHNIQUE와의 차이 (중요)

- **COMMON_TECHNIQUE**: 이 구성요소 자체가 관용기술이 아님을 방어
  * 관점: "구성요소 존재의 정당화"
- **SIMPLE_DESIGN (본 카테고리)**: D1에서 본 발명으로의 변경이 자명하지 않음
  * 관점: "D1에서 본 발명으로의 진화 곤란성"
  * 강조: 도출 곤란성, 통상 기술자 관점, 극복한 기술 장벽

## 출력 규칙

1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. **changed_component_label**과 **changed_component_name**은 사용자 구성요소 중 반박이 가장 필요한 하나의 라벨과 이름.
4. **non_obviousness**는 150~250자.
5. 사용자가 [차별점] 섹션을 제공한 경우 비자명성 논거에 반영.

## changed_component_label 선정 기준

### 우선순위 1 (강한 반박 논리)
- D1과 파라미터/방식이 미묘하게 다른 구성요소
- 심사관이 "단순 파라미터 조정"이라 폄하할 가능성 큼
- 예: D1의 CNN 필터 3x3 → 본 발명 5x5 (단순 변경 오해)

### 우선순위 2 (구조 변경)
- D1과 아키텍처가 다른 구성요소
- 심사관이 "설계 선택의 문제"라 폄하할 가능성
- 예: D1의 FC 레이어 → 본 발명 어텐션 메커니즘

**우선순위 1을 우선 선정**. 없으면 2 검토.

## non_obviousness 작성 가이드

다음 중 **최소 하나 이상의 논거**를 구체적으로 서술:

### 논거 1: 도출 곤란성
- 통상의 기술자가 D1으로부터 본 발명 변경을 시도할 이유 없음
- 시행착오, 기술적 편견 극복 필요
- 예: "D1의 CNN 구조에서 어텐션 추가는 당시 관행에 반하는 접근"

### 논거 2: 예상치 못한 효과 (가장 강력)
- 변경으로 얻은 효과가 D1으로부터 예상 불가
- 정량적 근거 (수치 비교) 있으면 더 강력
- 예: "필터 크기 변경만으로 정확도 7% 향상은 통상 예상 밖"

### 논거 3: 기술적 장벽 극복
- 변경 시 발생하는 기술적 문제를 본 발명이 해결
- 관용 방법으로 도출 시 실패했을 것임을 논증
- 예: "어텐션 추가 시 발생하는 계산 비용 문제를 특유 최적화로 해결"

### 논거 4: 상승 효과
- 변경된 구성요소와 다른 구성요소의 결합
- 개별 변경으로는 도출 안 되는 시너지
- 예: "어텐션(B)과 기존 CNN(A)의 결합으로 시맥락 특징 포착이라는 새 능력"

**논거 2, 4가 특히 강력**.

## 수치 데이터 활용 (있는 경우)

사용자 발명이 D1 대비 정량적 개선이 있으면 non_obviousness의 예상치 못한 효과 근거로 활용:
- "D1 대비 X% 향상은 통상의 파라미터 조정으로 도출 곤란한 예상 밖 효과"

## 사용자 입력 활용

- **prior_art_reference**: 사용자가 언급한 유사 선행기술
  → 이 선행기술도 실패했던 접근임을 논거 3 (기술적 장벽)에 반영

- **differentiation_notes**: 사용자가 명시한 차이점
  → 논거 1, 2에 반영 (특유 접근, 예상치 못한 효과)

## 반드시 지킬 것 (부정 명령)

- changed_component_label은 반드시 실제 사용자 구성요소 라벨 중 하나
- changed_component_name은 해당 구성요소의 실제 이름 그대로
- 발명 설명에 없는 효과나 특징을 창작 금지
- 150자 미만은 근거 부족, 250자 초과는 장황함

## 출력 형식

```
{
  "changed_component_label": "C",
  "changed_component_name": "정확한 구성요소 이름",
  "non_obviousness": "비자명성 논리 (150~250자)"
}
```

## 예시

### 예시 1: 아키텍처 변경 + 예상치 못한 효과 (G06N)

[사용자 발명]
명칭: 딥러닝 기반 실시간 얼굴 감정 인식 (정확도 94%)
구성요소:
A. CNN 얼굴 검출부
B. 어텐션 기반 특징 강화 모듈 ← 반박 대상 (D1에는 FC 레이어)
C. 감정 분류부

[D1]
청구항: CNN + FC 레이어 기반 얼굴 감정 인식 (정확도 87%)

출력:
{
  "changed_component_label": "B",
  "changed_component_name": "어텐션 기반 특징 강화 모듈",
  "non_obviousness": "D1의 FC 레이어를 어텐션 메커니즘으로 변경한 것은 통상의 단순 대체 설계로 보이나, 당시 얼굴 감정 인식 분야에서 FC 레이어가 표준이었고 어텐션은 NLP에 국한된 것으로 인식되어 통상의 기술자가 시도할 이유가 없었음. 이 변경으로 정확도 87%에서 94%로 7%p 향상되어 D1으로부터 예상할 수 없었던 효과 달성."
}

### 예시 2: 구조적 극복 (G06T)

[사용자 발명]
명칭: U-Net 개선 저조도 노이즈 제거 (PSNR 32.5dB)
구성요소:
A. 개선된 U-Net 인코더 (스킵 연결 강화)
B. Attention Gate 모듈 ← 반박 대상
C. 실시간 프레임 출력부

[D1]
청구항: 기존 U-Net 노이즈 제거 (PSNR 29.8dB, Attention Gate 미사용)

출력:
{
  "changed_component_label": "B",
  "changed_component_name": "Attention Gate 모듈",
  "non_obviousness": "D1의 표준 U-Net에 Attention Gate를 추가한 것은 단순한 부품 추가로 보이나, Attention Gate 도입 시 발생하는 계산 비용 증가로 실시간 처리 곤란이라는 기술적 편견이 있었음. 본 발명은 저조도 특화 학습으로 이 장벽을 극복하여 PSNR 2.7dB 향상과 실시간성을 동시 확보하는 예상치 못한 상승효과 달성."
}

### 예시 3: 상승효과 (G06Q)

[사용자 발명]
명칭: 시청 이력 기반 개인화 추천
구성요소:
A. 시청 이력 수집부
B. Transformer 임베딩부
C. 시맥락 랭킹 알고리즘 ← 반박 대상 (D1의 정적 랭킹과 다름)

[D1]
청구항: 협업 필터링 + 정적 유사도 기반 랭킹

출력:
{
  "changed_component_label": "C",
  "changed_component_name": "시맥락 랭킹 알고리즘",
  "non_obviousness": "D1의 정적 유사도 랭킹을 시맥락 랭킹으로 변경한 것은 단순 변형으로 보이나, 종래 협업 필터링 문헌은 정적 접근을 표준으로 삼아 시간 축 도입을 시도하지 않았음. Transformer 임베딩(B)과 결합하여 사용자 관심 변화의 장기 의존성을 포착하는 상승효과가 있으며, 개별 변경으로는 도출 곤란한 시너지 창출."
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

위 구성요소 중 심사관이 D1의 "단순 설계 변경"으로 판단할 가능성이 높은 하나를 선정하고, 
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
    """
    단순설계변경 비자명성 논리 자동 생성.

    Args:
        invention_title: 사용자 발명 명칭
        invention_description: 사용자 발명 설명
        components: 사용자 발명 구성요소 리스트
        primary_art: D1 (주인용, 필수)
        prior_art_reference: 사용자가 언급한 유사 선행기술 (선택)
        differentiation_notes: 사용자가 명시한 차이점 (선택)

    Returns:
        SimpleDesignResult 또는 None (LLM 호출 실패 or 구성요소 없음)

    Note:
        - D2 불필요 (D1 단독으로 논리 성립)
        - changed_component_label 검증 후 존재하지 않으면 첫 구성요소로 fallback
        - changed_component_name은 실제 구성요소 이름으로 정정
    """

    # ============================================================
    # 구성요소 필수 확인
    # ============================================================
    if not components:
        logger.error("[SimpleDesign] 구성요소가 없음. 비자명성 논리 생성 불가.")
        return None

    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}" for c in components
    ])

    # 라벨 → 컴포넌트 매핑 (사후 검증용)
    label_to_component = {c.label: c for c in components}

    # ============================================================
    # 사용자 차별점 정보 조립
    # ============================================================
    differentiation_text = ""
    if prior_art_reference or differentiation_notes:
        differentiation_text = (
            f"\n[사용자가 언급한 선행기술 대비 차별점]\n"
            f"선행기술: {prior_art_reference or '(없음)'}\n"
            f"차이점: {differentiation_notes or '(없음)'}\n"
            f"\n**중요: 위 사용자 제공 차별점을 비자명성 논거에 반영하세요.**"
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

    result = await _call_claude_with_json(
        system_prompt=SIMPLE_DESIGN_SYSTEM_PROMPT,
        user_message=user_message,
        result_model=SimpleDesignResult,
        log_prefix="[InventiveStep/SimpleDesign]",
    )

    if result is None:
        logger.error("[SimpleDesign] LLM 호출 실패")
        return None

    # 타입 안정성 보장
    if not isinstance(result, SimpleDesignResult):
        logger.error("[InventiveStep/SimpleDesignResult] LLM 호출 실패 또는 타입 불일치")
        return None

    # ============================================================
    # changed_component_label 검증 (사용자 구성요소에 존재하는지)
    # ============================================================
    if result.changed_component_label not in label_to_component:
        logger.warning(
            f"[SimpleDesign] LLM이 존재하지 않는 라벨 반환: "
            f"'{result.changed_component_label}'. 첫 구성요소로 fallback."
        )
        # Fallback: 첫 구성요소로 대체 (non_obviousness는 유지)
        fallback = components[0]
        result.changed_component_label = fallback.label
        result.changed_component_name = fallback.name
    else:
        # ============================================================
        # changed_component_name 정정 (실제 구성요소 이름으로)
        # ============================================================
        actual_component = label_to_component[result.changed_component_label]
        if result.changed_component_name != actual_component.name:
            logger.info(
                f"[SimpleDesign] changed_component_name 정정: "
                f"'{result.changed_component_name}' → '{actual_component.name}'"
            )
            result.changed_component_name = actual_component.name

    # ============================================================
    # non_obviousness 길이 로깅
    # ============================================================
    text_len = len(result.non_obviousness)
    if text_len < 150:
        logger.warning(
            f"[SimpleDesign] non_obviousness 짧음: {text_len}자 (권장 150~250)"
        )
    elif text_len > 250:
        logger.debug(
            f"[SimpleDesign] non_obviousness 길이: {text_len}자 (권장 150~250)"
        )

    logger.info(
        f"[SimpleDesign] 생성 완료: "
        f"changed={result.changed_component_label}({result.changed_component_name}), "
        f"non_obviousness={text_len}자, "
        f"user_input={bool(prior_art_reference or differentiation_notes)}"
    )

    return result

# ============================================================
# 공통 Claude 호출 헬퍼
# ============================================================

async def _call_claude_with_json(
        system_prompt: str,
        user_message: str,
        result_model: type[BaseModel],
        log_prefix: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
) -> Optional[BaseModel]:
    """
    Claude 호출 → JSON 파싱 → Pydantic 모델 검증 → 반환.

    Args:
        system_prompt: 시스템 프롬프트 (Prompt Caching 대상)
        user_message: 사용자 메시지
        result_model: 응답 검증용 Pydantic 모델 클래스
        log_prefix: 로그 접두어 (예: "[InventiveStep/NumericalLimit]")
        max_tokens: 최대 응답 토큰 (기본 2048)
        temperature: 창의성 조절 (기본 0.3, 판단 태스크는 낮게)

    Returns:
        검증된 Pydantic 모델 인스턴스 또는 None (실패 시)

    실패 케이스별 대응:
    - HTTP 오류: None 반환 (post_with_retry가 재시도 후 최종 실패)
    - max_tokens 도달: 경고 로그 + None 반환 (부분 응답은 신뢰 불가)
    - JSON 파싱 실패: None 반환 (재시도 없음, 비용 부담)
    - Pydantic 검증 실패: None 반환 (스키마 위반)
    """
    # ============================================================
    # Payload 조립 (Prompt Caching 활성화)
    # ============================================================
    payload = {
        "model": settings.claude_inventive_step_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # system을 배열로 구성 + cache_control로 캐싱 활성화
        # 상수 부분(system)만 캐싱되어 90% 비용 절감 가능
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_message}],
    }

    headers = {
        "x-api-key": settings.claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # ============================================================
    # Claude API 호출
    # ============================================================
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await post_with_retry(
                client, CLAUDE_ENDPOINT,
                headers=headers, json=payload,
                log_prefix=log_prefix,
            )
            data = response.json()
        except httpx.HTTPError:
            logger.exception(f"{log_prefix} Claude API 호출 실패")
            return None

    # ============================================================
    # max_tokens 도달 감지 (응답 잘림 방지)
    # ============================================================
    stop_reason = data.get("stop_reason")
    if stop_reason == "max_tokens":
        logger.warning(
            f"{log_prefix} max_tokens 도달로 응답 잘림. "
            f"max_tokens={max_tokens} 증량 필요."
        )
        # 잘린 응답은 신뢰 불가. 파싱 시도해도 실패 가능성 큼.
        return None

    # ============================================================
    # Prompt Caching 모니터링 (선택)
    # ============================================================
    usage = data.get("usage", {})
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    if cache_read > 0:
        logger.debug(f"{log_prefix} Cache hit: read={cache_read} tokens")
    elif cache_create > 0:
        logger.debug(f"{log_prefix} Cache created: {cache_create} tokens")

    # ============================================================
    # 응답 텍스트 추출
    # ============================================================
    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError):
        logger.error(f"{log_prefix} Claude 응답 구조 이상: {data}")
        return None

    # ============================================================
    # JSON 파싱
    # ============================================================
    stripped_text = strip_code_fence(text)
    try:
        parsed = json.loads(stripped_text)
    except json.JSONDecodeError:
        logger.error(f"{log_prefix} JSON 파싱 실패: {text[:300]}")
        return None

    # ============================================================
    # Pydantic 검증 (예외 범위 좁힘)
    # ============================================================
    try:
        return result_model(**parsed)
    except ValidationError as e:
        logger.error(
            f"{log_prefix} Pydantic 검증 실패: {e}, "
            f"parsed={parsed}"
        )
        return None
    except TypeError as e:
        logger.error(
            f"{log_prefix} 필드 타입 오류: {e}, "
            f"parsed={parsed}"
        )
        return None