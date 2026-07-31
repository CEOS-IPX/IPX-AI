"""
============================================================
신규성 분석 서비스
============================================================
사용자 발명의 구성요소를 상위 3건의 선행기술과 각각 비교하여
가장 유사한 1건을 주인용발명(D1)으로 선정한다.

분석 흐름:
  1. 각 선행기술 특허와 구성요소별 대비 (LLM, 병렬)
  2. 각 특허의 유사도 점수 계산
  3. 최고 유사도 특허를 D1으로 선정
  4. D1과의 상세 비교 결과 반환

결과 구조:
  - 전체 유사도 (매우 높음/높음/보통/낮음)
  - 결론 문구 (신규성 충족 여부 판단)
  - 구성요소별 대비 결과:
    - 판정 (동일/유사/신규)
    - 자연스러운 서술 (disclosure_text)
    - 원문 인용 + 출처 (citation)

모델: Claude Haiku
============================================================
"""

import json
import logging
import asyncio
from typing import Optional, List, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import settings
from app.services.llm_json import strip_code_fence
from app.services.llm_retry import post_with_retry

logger = logging.getLogger(__name__)

CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"


# ============================================================
# 모델
# ============================================================

class InventionComponent(BaseModel):
    """사용자 발명의 구성요소 1개"""
    label: str = Field(description="구성요소 라벨 (A, B, C, ...)")
    name: str = Field(description="구성요소 명칭")
    description: str = Field(description="구성요소 설명")


class PriorArtForAnalysis(BaseModel):
    """분석 대상 선행기술 특허"""
    application_number: str = Field(description="출원번호")
    title: str = Field(description="발명의 명칭")
    claims_independent: str = Field(
        description="독립 청구항 텍스트 ('청구항 N: 본문' 형식)"
    )


class ComponentComparison(BaseModel):
    """구성요소 1개에 대한 대비 결과."""
    component_label: str = Field(description="구성요소 라벨 (A, B, C, ...)")
    disclosure_text: str = Field(description="선행기술의 대응 개시 내용")
    citation: Optional[str] = Field(
        default=None,
        description="원문 인용. 개시되지 않은 경우 None"
    )
    result: Literal["동일", "유사", "신규"] = Field(
        description="대비 결과"
    )

    @field_validator("citation", mode="before")
    @classmethod
    def normalize_empty_citation(cls, v):
        """빈 문자열은 None으로 정규화."""
        if isinstance(v, str) and not v.strip():
            return None
        return v


class PatentAnalysisResult(BaseModel):
    """특허 1건에 대한 분석 결과."""
    application_number: str
    overall_similarity: Literal["매우 높음", "높음", "보통", "낮음", "매우 낮음"]
    conclusion_text: str = Field(description="신규성 판단 결론 문구")
    component_results: List[ComponentComparison]

# ============================================================
# 유사도 점수 계산
# ============================================================

def _calculate_similarity_score(analysis: PatentAnalysisResult) -> float:
    """
    분석 결과로부터 유사도 점수 계산.

    점수 부여:
      - 동일: 2점
      - 유사: 1점
      - 신규: 0점

    전체 구성요소 중 평균 점수 반환 (0~2 범위).
    최고점 = 가장 유사한 특허.
    """
    if not analysis.component_results:
        return 0.0

    scores = {"동일": 2.0, "유사": 1.0, "신규": 0.0}
    total = sum(
        scores.get(r.result, 0.0)
        for r in analysis.component_results
    )
    return total / len(analysis.component_results)


# ============================================================
# overall_similarity 계산
# ============================================================

def calculate_overall_similarity(
        component_results: List[ComponentComparison]
) -> Literal["매우 높음", "높음", "보통", "낮음", "매우 낮음"]:
    """
    구성요소 대비 결과로부터 전체 유사도 등급 계산.

    점수 체계:
      - 동일: 2점 (완전 개시, 신규성 부정 사유)
      - 유사: 1점 (부분 개시, 진보성 판단 영역)
      - 신규: 0점 (미개시)

    총점 대비 최대 가능 점수 비율:
      - 95% 이상: 매우 높음 (거의 모두 동일)
      - 75% 이상: 높음 (대부분 동일 또는 동일+유사)
      - 50% 이상: 보통 (동일/유사가 반반)
      - 25% 이상: 낮음 (신규가 많음)
      - 25% 미만: 매우 낮음 (거의 모두 신규)

    예시:
      - 동일 3 + 유사 1 (4개) → 7/8 = 87.5% → 높음
      - 동일 2 + 유사 1 + 신규 1 (4개) → 5/8 = 62.5% → 보통
      - 유사 2 + 신규 2 (4개) → 2/8 = 25% → 낮음
      - 모두 유사 (4개) → 4/8 = 50% → 보통
    """
    if not component_results:
        return "매우 낮음"

    SCORE_MAP = {"동일": 2.0, "유사": 1.0, "신규": 0.0}

    total_score = sum(
        SCORE_MAP.get(r.result, 0.0)
        for r in component_results
    )
    max_score = 2.0 * len(component_results)

    if max_score == 0:
        return "매우 낮음"

    ratio = total_score / max_score

    if ratio >= 0.95:
        return "매우 높음"
    elif ratio >= 0.75:
        return "높음"
    elif ratio >= 0.50:
        return "보통"
    elif ratio >= 0.25:
        return "낮음"
    else:
        return "매우 낮음"

# ============================================================
# 프롬프트
# ============================================================

SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 심사관이자 신규성 판단 전문가입니다.
사용자가 출원하려는 발명의 구성요소와 선행기술 특허의 청구항을 대비하여,
각 구성요소가 선행기술에 개시되어 있는지 판단합니다.

## 출력 규칙

1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. 모든 구성요소를 빠짐없이 판단.
4. 판단은 청구항에 명시된 내용을 기준으로. 청구항에 없는 내용 추측 금지.
5. 청구항은 "청구항 N: 본문" 형식으로 제공됨. 인용 시 정확한 청구항 번호와 문장을 그대로 사용.
6. component_label은 사용자 발명의 구성요소 라벨(A, B, C, D ...)을 그대로 사용.

## 판정 기준 (동일 / 유사 / 신규)

### "동일" (identical)
- 청구항에 구성요소의 명칭 또는 실질적 등가물이 명시됨
- 구성요소의 기능, 구조, 파라미터가 모두 매칭됨
- 술어(용어) 차이는 무관 (예: "합성곱 신경망" = "CNN")
- **신규성 판단에서 부정 사유가 됨**

### "유사" (similar)
- 청구항에 유사 기능/목적의 구성이 있으나
- 구체적 구현 방식, 파라미터, 알고리즘이 다름
- **신규성은 인정됨** (진보성 판단은 별개)

### "신규" (novel)
- 청구항에 대응 구성이 명확히 없음
- 기능/목적이 다르거나 언급 자체가 없음
- **명백히 신규성 인정 사유**

## 반드시 지킬 것 (부정 명령)

- 청구항에 없는 내용을 추측하여 "동일"로 판정 금지
- 확실하지 않으면 "유사"로 판정 (엄격한 "동일" 판정 회피)
- disclosure_text에 "선행기술 청구항에 개시되지 않음"만 쓰지 말 것 
  (왜, 어떻게 다른지 구체적 서술 필수)
- citation은 청구항 원문 그대로. 재해석/의역 금지.

## conclusion_text 작성 규칙 (특허법 기준)

한국 특허법 제29조 제1항 (신규성) 판단은 다음과 같이 서술:

### 케이스 1: "신규"로 판정된 구성요소가 하나라도 있으면
"구성요소 [해당 라벨들]이(가) 주인용발명에 개시되어 있지 않은 차이점입니다. 
본 발명은 단일 선행문헌과 실질적으로 동일하지 않으므로, 
특허법 제29조 제1항의 신규성을 충족합니다."

라벨 여러 개면 "구성요소 A, B가" 등으로 자연스럽게 나열.

### 케이스 2: "신규"는 없고 "유사"가 하나라도 있으면
"모든 구성요소가 주인용발명에 개시되어 있으나, 
구성요소 [해당 라벨들]에 유사 개시(완전 동일하지 않음)가 있어 
특허법 제29조 제1항의 신규성은 인정됩니다. 
다만 진보성(제29조 제2항) 판단이 별도 필요합니다."

### 케이스 3: 모든 구성요소가 "동일"
"본 발명의 모든 구성요소가 주인용발명에 실질적으로 동일하게 개시되어 있어, 
특허법 제29조 제1항의 신규성이 부정될 가능성이 있습니다."

## 출력 형식

{
  "conclusion_text": "특허법 기준 판단 결론",
  "component_results": [
    {
      "component_label": "A",
      "disclosure_text": "선행기술 청구항에 어떻게 개시되어 있는지 자연스러운 서술",
      "citation": "청구항 N: 원문 그대로",
      "result": "동일|유사|신규"
    }
  ]
}

## 각 필드 작성 가이드

### disclosure_text
- 자연스러운 서술체 문장
- "동일" 케이스: "[구성 서술]이 개시됨"
- "유사" 케이스: "[선행기술 구성]이 개시되었으나 [차이점] 존재"
- "신규" 케이스: 왜 개시되지 않았는지, 선행기술이 어떤 다른 방식을 취하는지 서술

### citation
- 청구항 원문에서 관련 부분을 그대로 인용
- 반드시 "청구항 N: " 접두어와 함께 원문 표현 그대로
- 개시되지 않은 경우 null
- 관련 없는 부분 인용 금지

## 예시

### 예시 1: 모든 구성요소 "동일" (신규성 부정 여지)

[사용자 발명]
- 명칭: 딥러닝 기반 얼굴 감정 인식 시스템
- 구성요소:
  A: 이미지 입력부 (카메라 영상 실시간 수신)
  B: CNN 얼굴 검출 모듈 (CNN으로 얼굴 영역 검출)
  C: 감정 분류부 (FC 레이어로 5개 감정 분류)

[선행기술 청구항]
청구항 1: 카메라로부터 영상을 수신하는 영상 수신부; 상기 영상에서 
합성곱 신경망을 이용하여 얼굴 영역을 검출하는 검출부; 검출된 얼굴 영역을 
완전 연결 계층에 입력하여 감정을 5개 카테고리로 분류하는 분류부를 
포함하는 얼굴 감정 인식 시스템.

출력:
{
  "conclusion_text": "본 발명의 모든 구성요소가 주인용발명에 실질적으로 동일하게 개시되어 있어, 특허법 제29조 제1항의 신규성이 부정될 가능성이 있습니다.",
  "component_results": [
    {
      "component_label": "A",
      "disclosure_text": "카메라로부터 실시간 영상을 수신하는 구성이 개시됨.",
      "citation": "청구항 1: 카메라로부터 영상을 수신하는 영상 수신부",
      "result": "동일"
    },
    {
      "component_label": "B",
      "disclosure_text": "CNN(합성곱 신경망)으로 얼굴 영역을 검출하는 구성이 개시됨. 용어는 다르나 실질적 등가.",
      "citation": "청구항 1: 상기 영상에서 합성곱 신경망을 이용하여 얼굴 영역을 검출하는 검출부",
      "result": "동일"
    },
    {
      "component_label": "C",
      "disclosure_text": "FC 레이어(완전 연결 계층)로 5개 감정 카테고리를 분류하는 구성이 개시됨.",
      "citation": "청구항 1: 검출된 얼굴 영역을 완전 연결 계층에 입력하여 감정을 5개 카테고리로 분류하는 분류부",
      "result": "동일"
    }
  ]
}

### 예시 2: 일부 "신규" (신규성 인정)

[사용자 발명]
- 명칭: 시계열 얼굴 감정 분석 시스템
- 구성요소:
  A: CNN 얼굴 검출부
  B: LSTM 시계열 감정 분석부 (시간 흐름에 따른 감정 변화 분석)
  C: 개인화 감정 프로필 저장부

[선행기술 청구항]
청구항 1: CNN을 이용한 얼굴 검출부; 검출된 얼굴에서 감정을 분류하는 
FC 레이어 기반 분류부를 포함하는 얼굴 감정 인식 시스템.

출력:
{
  "conclusion_text": "구성요소 B, C가 주인용발명에 개시되어 있지 않은 차이점입니다. 본 발명은 단일 선행문헌과 실질적으로 동일하지 않으므로, 특허법 제29조 제1항의 신규성을 충족합니다.",
  "component_results": [
    {
      "component_label": "A",
      "disclosure_text": "CNN을 이용한 얼굴 검출 구성이 개시됨.",
      "citation": "청구항 1: CNN을 이용한 얼굴 검출부",
      "result": "동일"
    },
    {
      "component_label": "B",
      "disclosure_text": "시간 흐름에 따른 감정 변화 분석 구성이 선행기술 청구항에 개시되지 않음. 선행기술은 정지 영상 기반 단발성 감정 분류만 개시하며, 시간 축을 활용하지 않음.",
      "citation": null,
      "result": "신규"
    },
    {
      "component_label": "C",
      "disclosure_text": "개인화 감정 프로필 저장 구성이 선행기술 청구항에 개시되지 않음. 선행기술은 사용자별 데이터 저장/개인화 언급 없음.",
      "citation": null,
      "result": "신규"
    }
  ]
}

### 예시 3: 일부 "유사" 있음 (신규성 인정, 진보성 별도 판단)

[사용자 발명]
- 명칭: U-Net 기반 저조도 이미지 노이즈 제거
- 구성요소:
  A: 저조도 이미지 입력부
  B: U-Net 노이즈 검출 네트워크
  C: GAN 기반 노이즈 복원 모듈

[선행기술 청구항]
청구항 1: 저조도 이미지를 입력받는 입력부; FCN 기반 이미지 분할 
네트워크를 이용하여 노이즈 영역을 검출하는 검출부; 검출된 영역을 
CNN 기반 인페인팅으로 복원하는 복원부를 포함하는 이미지 처리 시스템.

출력:
{
  "conclusion_text": "모든 구성요소가 주인용발명에 개시되어 있으나, 구성요소 B, C에 유사 개시(완전 동일하지 않음)가 있어 특허법 제29조 제1항의 신규성은 인정됩니다. 다만 진보성(제29조 제2항) 판단이 별도 필요합니다.",
  "component_results": [
    {
      "component_label": "A",
      "disclosure_text": "저조도 이미지를 입력받는 구성이 개시됨.",
      "citation": "청구항 1: 저조도 이미지를 입력받는 입력부",
      "result": "동일"
    },
    {
      "component_label": "B",
      "disclosure_text": "이미지 분할 네트워크로 노이즈 영역을 검출하는 구성이 개시되었으나, U-Net이 아닌 FCN 기반이라는 아키텍처 차이가 있음.",
      "citation": "청구항 1: FCN 기반 이미지 분할 네트워크를 이용하여 노이즈 영역을 검출하는 검출부",
      "result": "유사"
    },
    {
      "component_label": "C",
      "disclosure_text": "이미지 노이즈 복원 구성이 개시되었으나, GAN이 아닌 CNN 인페인팅 방식이라는 알고리즘 차이가 있음.",
      "citation": "청구항 1: 검출된 영역을 CNN 기반 인페인팅으로 복원하는 복원부",
      "result": "유사"
    }
  ]
}

이제 JSON만 출력하세요.
"""

USER_PROMPT_TEMPLATE = """\
[사용자 발명]
명칭: {invention_title}
설명: {invention_description}

구성요소:
{components_text}

[선행기술 특허]
명칭: {patent_title}
출원번호: {patent_application_number}

독립 청구항:
{patent_claims}

위 사용자 발명의 각 구성요소가 이 선행기술 특허에 개시되어 있는지 판단하여
JSON으로 반환하세요.
"""



# ============================================================
# 단일 특허 분석
# ============================================================

async def _compare_with_patent(
    invention_title: str,
    invention_description: str,
    components: list[InventionComponent],
    patent: PriorArtForAnalysis,
    client: httpx.AsyncClient,
) -> Optional[PatentAnalysisResult]:
    """
    사용자 발명 구성요소를 선행기술 특허 1건과 대비 (LLM 호출).

    Returns:
        PatentAnalysisResult 또는 None (실패 시)
    """
    # 구성요소 텍스트 조립
    components_text = "\n".join([
        f"{c.label}. {c.name}: {c.description}"
        for c in components
    ])

    user_message = USER_PROMPT_TEMPLATE.format(
        invention_title=invention_title,
        invention_description=invention_description,
        components_text=components_text,
        patent_title=patent.title,
        patent_application_number=patent.application_number,
        patent_claims=patent.claims_independent,
    )

    payload = {
        "model": settings.claude_novelty_model,
        "max_tokens": 3072,
        "temperature": 0.1,  # 판정 재현성 향상 (0.3 → 0.1)
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
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

    # Claude API 호출
    try:
        response = await post_with_retry(
            client, CLAUDE_ENDPOINT, headers=headers, json=payload, log_prefix="[Novelty]",
        )
        data = response.json()
    except httpx.HTTPError:
        logger.exception(f"[Novelty] Claude API 호출 실패: {patent.application_number}")
        return None

    # ============================================================
    # max_tokens 도달 감지
    # ============================================================
    stop_reason = data.get("stop_reason")
    if stop_reason == "max_tokens":
        logger.warning(
            f"[Novelty] max_tokens 도달로 응답 잘림 가능: "
            f"patent={patent.application_number}"
        )

    # ============================================================
    # Prompt Caching 모니터링
    # ============================================================
    usage = data.get("usage", {})
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    if cache_read > 0:
        logger.debug(f"[Novelty] Cache hit: read={cache_read} tokens")
    elif cache_create > 0:
        logger.debug(f"[Novelty] Cache created: {cache_create} tokens")

    # ============================================================
    # 응답 텍스트 추출
    # ============================================================
    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError):
        logger.error(f"[Novelty] Claude 응답 구조 이상: {data}")
        return None

    # JSON 파싱
    try:
        parsed = json.loads(strip_code_fence(text))
    except json.JSONDecodeError:
        logger.error(f"[Novelty] JSON 파싱 실패: {text[:300]}")
        return None

    # ============================================================
    # Pydantic 검증 (Literal 타입 자동 검증)
    # ============================================================
    try:
        # 1. component_results 먼저 파싱 및 검증
        component_results = [
            ComponentComparison(**r)
            for r in parsed.get("component_results", [])
        ]

        # 2. overall_similarity를 결정론적으로 계산 (LLM 값 무시)
        overall_similarity = calculate_overall_similarity(component_results)

        # 3. 최종 결과 조립
        result = PatentAnalysisResult(
            application_number=patent.application_number,
            overall_similarity=overall_similarity,
            conclusion_text=parsed.get("conclusion_text", ""),
            component_results=component_results,
        )
    except ValidationError as e:
        logger.error(
            f"[Novelty] 결과 검증 실패: "
            f"patent={patent.application_number}, error={e}"
        )
        return None
    except KeyError as e:
        logger.error(
            f"[Novelty] 필수 필드 누락: "
            f"patent={patent.application_number}, missing={e}"
        )
        return None

    # ============================================================
    # 구성요소 개수 검증 (LLM이 누락하지 않았는지)
    # ============================================================
    if len(result.component_results) != len(components):
        logger.warning(
            f"[Novelty] 구성요소 개수 불일치: "
            f"입력 {len(components)}개, 응답 {len(result.component_results)}개, "
            f"patent={patent.application_number}"
        )
        # MVP엔 진행. 실패 처리하지 않음 (부분 결과라도 활용)

    return result

# ============================================================
# 신규성 분석 진입점 (여러 특허 비교 + 최유사 특허 선정)
# ============================================================

async def analyze_novelty(
    invention_title: str,
    invention_description: str,
    components: list[InventionComponent],
    prior_arts: list[PriorArtForAnalysis],
) -> Optional[PatentAnalysisResult]:
    """
    상위 N건 선행기술과 각각 비교 후 가장 유사한 1건 반환.

    Args:
        invention_title: 사용자 발명 명칭
        invention_description: 사용자 발명 설명
        components: 사용자 발명 구성요소 리스트 (label 포함)
        prior_arts: 분석 대상 선행기술 리스트 (최대 3건 권장)

    Returns:
        가장 유사한 특허 1건의 PatentAnalysisResult
        또는 None (모든 분석이 실패한 경우)
    """
    if not prior_arts:
        logger.warning("[Novelty] 분석 대상 선행기술이 없음")
        return None
    if not components:
        logger.warning("[Novelty] 구성요소가 없음")
        return None

    logger.info(
        f"[Novelty] 분석 시작: 구성요소 {len(components)}개, "
        f"선행기술 {len(prior_arts)}건"
    )

    # 병렬 LLM 호출 (특허별)
    async with httpx.AsyncClient(timeout=90.0) as client:
        tasks = [
            _compare_with_patent(
                invention_title=invention_title,
                invention_description=invention_description,
                components=components,
                patent=patent,
                client=client,
            )
            for patent in prior_arts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    # 실패한 것 제외
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        logger.error("[Novelty] 모든 특허 분석 실패")
        return None

    # 가장 유사한 특허 선정
    scored = [(r, _calculate_similarity_score(r)) for r in valid_results]
    best, best_score = max(scored, key=lambda x: x[1])

    logger.info(
        f"[Novelty] D1 선정: {best.application_number}, "
        f"score={best_score:.2f}, similarity={best.overall_similarity}, "
        f"성공 {len(valid_results)}/{len(prior_arts)}건"
    )

    return best