"""
============================================================
특허 상세 정보 LLM 추출 서비스
============================================================
검색된 각 특허에 대해 상세 페이지에 표시할 정보를 Claude로 추출한다.

추출 항목:
  1. 핵심 요약 (summary): 특허 자체를 1문장으로 압축
  2. 기술 목적 (purpose): 특허가 해결하려는 문제/목적
  3. 주요 특징 (features): 기술적으로 구별되는 특징 리스트
  4. 관련 키워드 (keywords): 변리사가 한눈에 파악할 수 있는 핵심 키워드
  5. 추천 이유 (reason): 사용자 발명과 왜 관련성 높은지 설명

호출 전략:
  - 검색 결과 N건에 대해 비동기 병렬 호출
  - 사용자 발명 정보(title, description, keywords)를 모든 호출에 전달
  - 1건 실패 시 해당 특허는 None 반환, 나머지 영향 X

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
# 응답 모델
# ============================================================

class PatentSummary(BaseModel):
    """LLM이 추출한 특허 상세 정보"""

    summary: str = Field(description="핵심 요약 1문장")
    purpose: str = Field(description="기술 목적 / 해결하려는 문제")
    features: list[str] = Field(description="주요 기술적 특징 3~5개")
    keywords: list[str] = Field(description="관련 키워드 5~8개")
    reason: str = Field(description="사용자 발명과의 관련성을 설명하는 추천 이유 2~3문장")


# ============================================================
# 프롬프트
# ============================================================

SYSTEM_PROMPT = """\
당신은 특허 분석 전문가입니다.
변리사가 출원하려는 발명 정보와 검색된 선행기술 특허 정보를 받아,
변리사가 한눈에 파악할 수 있도록 5가지 항목을 구조화된 JSON으로 추출합니다.

출력 규칙:
1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 금지.
2. 한국어로 작성.
3. 모든 필드는 빈 값이 아닌 의미 있는 내용으로 채울 것.
4. 분량은 짧고 압축된 형태로. 변리사가 빠르게 훑을 수 있도록.
5. 화학식, 수치, 조건 등 명세서의 구체적인 표현을 가능한 한 그대로 보존할 것.

출력 형식:
{
  "summary": "핵심 요약 1문장 (80~120자)",
  "purpose": "기술 목적 1문장 (60~100자)",
  "features": ["주요 특징 1", "주요 특징 2", "주요 특징 3", "주요 특징 4"],
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "reason": "추천 이유 2~3문장 (100~150자)"
}

각 필드 작성 가이드:

- summary: 선행기술 특허 자체의 핵심 구성과 동작을 1문장으로 압축.
           사용자 발명은 언급하지 말 것. 청구항 본질을 그대로 담을 것.
           예: "50-60°C 저온에서 H₂SO₄와 환원제를 단계적으로 투입해 폐리튬이온전지 양극재로부터
                니켈과 코발트를 동시 침출하는 습식제련 방법."

- purpose: 이 선행기술 특허가 종래 기술의 한계를 어떻게 극복하거나 어떤 문제를 해결하려는지 1문장으로.
           예: "폐리튬이온전지 양극재에서 니켈과 코발트를 고회수율로 선택적 회수하여
                배터리 원료로 재활용하는 친환경 공정 개발."

- features: 이 선행기술을 구별짓는 핵심 특징 3~5개를 짧은 명사구로.
            각 항목은 30자 이내.
            예: ["저온 공정으로 에너지 소비 최소화",
                  "산과 환원제 단계적 투입으로 침출 선택성 향상",
                  "니켈·코발트 동시 침출로 공정 단계 단축",
                  "기존 건식 공정 대비 설비 비용 절감"]

- keywords: 이 선행기술의 핵심 키워드 5~8개. 짧은 단어 또는 명사구.
            너무 일반적인 단어(예: "기술", "방법") 제외. 도메인 특화 용어 우선.
            예: ["저온 침출", "습식 제련", "황산 농도 제어", "NiCo 분리",
                  "용매 추출", "D2EHPA", "폐리튬이온전지"]

- reason: 이 선행기술이 사용자가 출원하려는 발명과 왜 관련성이 높은지 2~3문장으로 설명.
          구성 원칙 (반드시 이 순서):
            (1) 첫 문장: 선행기술의 핵심 특징을 압축 (수치, 조건 등 구체적으로).
            (2) 다음 문장: 사용자가 입력한 발명 정보에서 언급된 개념이 이 선행기술의
                청구항/구성에 어떻게 매칭되는지 명시.
                "입력하신 [키워드 A, 키워드 B, 키워드 C]가 청구항에 직접 일치합니다"
                또는 "입력하신 [X]가 이 발명의 [Y] 부분에서 실현됩니다" 같은 형태.
            (3) 변리사가 자기 발명과 비교해서 "아, 이래서 유사하구나"를 즉시 이해할 수 있어야 함.
          예: "60°C 이하 저온 환경에서 황산 농도를 단계적으로 조절해 니켈/코발트를 92% 이상 회수하는 공정.
                입력하신 저온·회수율·황산 첨감 세 조건이 모두 청구항에 직접 일치합니다."

이제 JSON만 출력하세요.
"""

USER_PROMPT_TEMPLATE = """\
[사용자가 출원하려는 발명]
명칭: {user_title}
설명: {user_description}
관련 키워드: {user_keywords}

[검색된 선행기술 특허]
발명의 명칭: {patent_title}
초록: {patent_abstract}
독립 청구항: {patent_claims}

위 선행기술 특허에 대해 핵심 요약, 기술 목적, 주요 특징, 관련 키워드,
그리고 사용자 발명과의 관련성을 설명하는 추천 이유를 JSON으로 추출하세요.
"""


# ============================================================
# 단일 특허 요약
# ============================================================

async def summarize_one(
    # 특허 정보
    patent_title: str,
    patent_abstract: str,
    patent_claims: str,
    # 사용자 발명 정보
    user_title: str,
    user_description: str,
    user_keywords: list[str],
    # 클라이언트
    client: httpx.AsyncClient,
) -> Optional[PatentSummary]:
    """
    단일 특허에 대해 Claude로 정보 추출

    Args:
        patent_title: 선행기술 특허의 명칭
        patent_abstract: 선행기술 특허의 초록 (전처리됨)
        patent_claims: 선행기술 특허의 독립 청구항 (전처리됨)
        user_title: 사용자가 출원하려는 발명의 명칭
        user_description: 사용자가 출원하려는 발명의 설명
        user_keywords: 사용자 발명에서 추출된 키워드 (원본, 동의어 확장 전)
        client: 재사용할 httpx 클라이언트

    Returns:
        PatentSummary 또는 None (실패 시)
    """
    user_keywords_text = ", ".join(user_keywords) if user_keywords else "(없음)"

    user_message = USER_PROMPT_TEMPLATE.format(
        user_title=user_title or "(없음)",
        user_description=user_description or "(없음)",
        user_keywords=user_keywords_text,
        patent_title=patent_title or "(없음)",
        patent_abstract=patent_abstract or "(없음)",
        patent_claims=patent_claims or "(없음)",
    )

    payload = {
        "model": settings.claude_model,
        "max_tokens": 1500,   # reason 필드가 추가되어 여유 있게
        "temperature": 0.3,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_message}
        ],
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
        logger.exception(f"[Summary] Claude API 호출 실패: title='{patent_title[:30] if patent_title else ''}'")
        return None
    except Exception:
        logger.exception(f"[Summary] 예상치 못한 오류: title='{patent_title[:30] if patent_title else ''}'")
        return None

    # 응답 텍스트 추출
    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError):
        logger.error(f"[Summary] Claude 응답 구조 이상: {data}")
        return None

    # JSON 파싱
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"[Summary] JSON 파싱 실패: title='{patent_title[:30] if patent_title else ''}', text='{text[:200]}'")
        return None

    # Pydantic 검증
    try:
        return PatentSummary(**parsed)
    except Exception:
        logger.exception(f"[Summary] PatentSummary 검증 실패: parsed={parsed}")
        return None


# ============================================================
# 다건 병렬 요약
# ============================================================

async def summarize_batch(
    patent_data: list[dict],
    user_title: str,
    user_description: str,
    user_keywords: list[str],
) -> list[Optional[PatentSummary]]:
    """
    여러 특허에 대해 병렬로 정보 추출.

    Args:
        patent_data: 요약 대상 선행기술 특허 정보 리스트
            각 dict는 다음 키를 포함해야 함:
              - title: str
              - abstract: str
              - claims_independent: str
        user_title: 사용자가 출원하려는 발명의 명칭
        user_description: 사용자가 출원하려는 발명의 설명
        user_keywords: 사용자 발명 키워드 (LLM 의도 해석 원본, 동의어 확장 전)

    Returns:
        각 특허의 PatentSummary 또는 None (입력 순서 유지).
        실패한 특허는 None이 들어감.
    """
    if not patent_data:
        return []

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            summarize_one(
                patent_title=p.get("title", ""),
                patent_abstract=p.get("abstract", ""),
                patent_claims=p.get("claims_independent", ""),
                user_title=user_title,
                user_description=user_description,
                user_keywords=user_keywords,
                client=client,
            )
            for p in patent_data
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)

    success_count = sum(1 for r in results if r is not None)
    logger.info(f"[Summary] 병렬 요약 완료: 성공 {success_count}/{len(patent_data)}건")

    return results