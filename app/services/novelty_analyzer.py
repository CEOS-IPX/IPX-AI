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
  - 구성요소별 대비 결과 (동일/유사/신규 + 개시 내용)

모델: Claude Haiku
============================================================
"""

import json
import logging
import asyncio
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from app.config import settings

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
    claims_independent: str = Field(description="독립 청구항 텍스트")


class ComponentComparison(BaseModel):
    """구성요소 1개에 대한 대비 결과"""
    component_label: str = Field(description="구성요소 라벨 (A, B, C, ...)")
    disclosure_text: str = Field(description="선행기술의 대응 게시 내용")
    result: str = Field(description="대비 결과: '동일' | '유사' | '신규'")


class PatentAnalysisResult(BaseModel):
    """특허 1건에 대한 분석 결과"""
    application_number: str
    overall_similarity: str = Field(description="'매우 높음' | '높음' | '보통' | '낮음'")
    conclusion_text: str = Field(description="신규성 판단 결론 문구")
    component_results: list[ComponentComparison]


# ============================================================
# 프롬프트
# ============================================================

SYSTEM_PROMPT = """\
당신은 특허 신규성 판단 전문가입니다.
사용자가 출원하려는 발명의 구성요소와 선행기술 특허의 청구항을 대비하여,
각 구성요소가 선행기술에 개시되어 있는지 판단합니다.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. 모든 구성요소를 빠짐없이 판단.
4. 판단은 청구항에 명시된 내용을 기준으로. 추측 금지.

각 구성요소의 대비 결과:
- "동일": 선행기술 청구항에 동일한 구성이 명시적으로 개시됨
- "유사": 유사한 구성이 있으나 완전히 동일하지는 않음 (기능/목적이 유사하지만 구체적 구현이 다름 등)
- "신규": 선행기술 청구항에 개시되지 않은 새로운 구성

overall_similarity 판단 기준:
- 매우 높음: 대부분(80% 이상) "동일"
- 높음: "동일"이 절반 이상
- 보통: "동일"과 "유사"가 섞여있음
- 낮음: "신규"가 절반 이상

출력 형식:
{
  "overall_similarity": "매우 높음|높음|보통|낮음",
  "conclusion_text": "...",
  "component_results": [
    {
      "component_label": "A",
      "disclosure_text": "선행기술 청구항에 어떻게 개시되어 있는지 구체적 내용. 개시되지 않았다면 '해당 구성이 선행기술 청구항에 개시되지 않음'",
      "result": "동일|유사|신규"
    },
    ...
  ]
}

disclosure_text 작성 가이드:
- 개시된 경우: 청구항 내용을 그대로 인용하지 말고, 자연스러운 서술 문장으로.
              예: "H₂SO₄ 1.0~1.5 M을 폐리튬이온전지 양극재에 투입하여 니켈과 코발트를 침출하는 구성이 개시됨."
- 개시 안 된 경우: "해당 구성이 선행기술 청구항에 개시되지 않음."

conclusion_text 작성 규칙:
- "신규"로 판정된 구성요소가 하나라도 있으면:
    "구성요소 [해당 라벨들]가(이) 주인용발명에 개시되어 있지 않은 차이점입니다. 
     본 발명은 단일 선행문헌과 실질적으로 동일하지 않으므로, 특허법 제29조 제1항의 신규성을 충족합니다."
    (라벨 여러 개면 "구성요소 A, B가", "구성요소 C·D가" 등으로 자연스럽게)

- 모두 "동일" 또는 "유사"이면:
    "본 발명의 모든 구성요소가 주인용발명에 실질적으로 개시되어 있어, 
     특허법 제29조 제1항의 신규성이 부정될 여지가 있습니다."

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
        "max_tokens": 2048,
        "temperature": 0.3,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    }

    headers = {
        "x-api-key": settings.claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        response = await client.post(CLAUDE_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError:
        logger.exception(f"[Novelty] Claude API 호출 실패: {patent.application_number}")
        return None

    # 응답 파싱
    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError):
        logger.error(f"[Novelty] Claude 응답 구조 이상: {data}")
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"[Novelty] JSON 파싱 실패: {text[:300]}")
        return None

    try:
        component_results = [
            ComponentComparison(**r)
            for r in parsed.get("component_results", [])
        ]
        return PatentAnalysisResult(
            application_number=patent.application_number,
            overall_similarity=parsed["overall_similarity"],
            conclusion_text=parsed["conclusion_text"],
            component_results=component_results,
        )
    except Exception as e:
        logger.error(f"[Novelty] 결과 검증 실패: {e}, parsed={parsed}")
        return None


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
    best = max(valid_results, key=_calculate_similarity_score)
    best_score = _calculate_similarity_score(best)

    logger.info(
        f"[Novelty] D1 선정: {best.application_number}, "
        f"score={best_score:.2f}, similarity={best.overall_similarity}"
    )

    return best