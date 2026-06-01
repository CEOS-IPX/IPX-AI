"""
============================================================
LLM 의도 해석 서비스 (Gemini 2.5 Flash-Lite)
============================================================
사용자의 자연어 입력을 분석-> 검색 파이프라인이 사용할 JSON 구조로 변환

출력 JSON 예시
{
  "is_valid": true,
  "reason_invalid": null,
  "keywords": ["급속충전", "리튬이온", "음극재"],
  "ipc_codes": ["H01M 10/052", "H01M 4/13"],
}
============================================================
"""

import json
import logging
from typing import Optional
import httpx
from pydantic import BaseModel, Field
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Pydantic 모델: LLM 응답 구조 검증
# ============================================================

class IntentResult(BaseModel):
    """LLM이 추출한 의도 해석 결과"""

    is_valid: bool = Field(description="특허 검색에 적합한 입력인지")
    reason_invalid: Optional[str] = Field(default=None, description="유효하지 않을 때의 사유")
    keywords: list[str] = Field(default_factory=list, description="핵심 기술 키워드")
    ipc_codes: list[str] = Field(default_factory=list, description="관련 IPC 코드")


# ============================================================
# 프롬프트 구성
# ============================================================

SYSTEM_INSTRUCTION = """\
당신은 특허 검색을 도와주는 의도 해석 전문가입니다.
사용자의 자연어 입력을 분석해서 특허 검색 파이프라인이 사용할 구조화된 JSON을 생성하세요.

## 입력 형식
사용자 메시지에 다음 정보가 포함될 수 있습니다:
- "검색 도메인": 사용자가 미리 선택한 기술 분야 (예: 이차전지, 반도체, AI)
- "사용자 입력": 자연어 검색어

도메인이 제공되면 키워드와 IPC 코드 추정 시 해당 도메인에 집중하세요.
도메인이 없으면 사용자 입력만으로 판단하세요.

## 출력 규칙

다음 JSON 형식만 출력하세요. 다른 설명, 마크다운 코드 블록은 절대 포함하지 마세요.

{
  "is_valid": boolean,
  "reason_invalid": string or null,
  "keywords": [string],
  "ipc_codes": [string]
}

## 각 필드 작성 가이드

- **is_valid**: 다음 중 하나라도 해당되면 false
  - 욕설, 잡담, 무의미한 입력
  - 너무 모호해서 검색할 수 없는 입력 (예: "좋은 거", "아무거나")
  - 특허와 무관한 주제 (예: "오늘 날씨", "맛집 추천")
  - 위 조건에 해당하지 않으면 true

- **reason_invalid**: is_valid=false일 때만 사용자에게 보여줄 친절한 메시지 작성. true면 null

- **keywords**: 3~10개의 단어 단위 키워드
  - 기술 키워드: 발명의 핵심 기술 (예: "리튬이온전지", "음극재", "고체전해질")
  - 문제 키워드: 사용자가 해결하려는 문제 (예: "급속충전", "배터리수명", "열화")
  - 단어 또는 짧은 복합명사 (문장 X)
  - 한글/영문 혼용 가능 (예: "LIB", "이차전지")
  - 너무 일반적인 단어 제외 (예: "기술", "방법", "장치")

- **ipc_codes**: 관련 IPC 분류 코드 (메인 그룹 또는 서브 그룹 수준)
  - 예: ["H01M 10/052"], ["B60L 53/16", "H02J 7/00"]
  - 확실한 것만 1~5개. 모르겠으면 빈 배열

## 예시

입력: "급속충전 시 배터리 열화를 줄이는 기술"
출력:
{
  "is_valid": true,
  "reason_invalid": null,
  "keywords": ["급속충전", "리튬이온전지", "배터리수명", "열화", "음극재", "사이클수명"],
  "ipc_codes": ["H01M 10/052", "H01M 4/13"]
}

입력: "전기차 충전 인프라"
출력:
{
  "is_valid": true,
  "reason_invalid": null,
  "keywords": ["전기차충전", "충전스테이션", "충전기", "커넥터", "전력제어"],
  "ipc_codes": ["B60L 53/00", "H02J 7/00"]
}

입력: "오늘 점심 뭐 먹지"
출력:
{
  "is_valid": false,
  "reason_invalid": "특허 검색과 관련된 기술 주제를 입력해 주세요.",
  "keywords": [],
  "ipc_codes": []
}
"""

# ============================================================
# Gemini API 호출
# ============================================================

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"


async def interpret_intent(user_query: str, domain: Optional[str] = None) -> IntentResult:
    """
    사용자 자연어 입력을 의도 해석 결과로 변환

    Args:
        user_query: 사용자 자연어 입력

    Returns:
        IntentResult: 의도 해석 결과

    Raises:
        ValueError: LLM 응답을 파싱할 수 없을 때
        httpx.HTTPError: API 호출 실패 시
    """

    user_text = f"검색 도메인: {domain}\n사용자 입력: {user_query}" if domain else user_query

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 1024
        }
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GEMINI_ENDPOINT,
            params={"key": settings.gemini_api_key},
            json=payload
        )
        response.raise_for_status()
        data = response.json()

    # Gemini 응답에서 텍스트 추출
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error(f"Gemini 응답 구조 이상: {data}")
        raise ValueError(f"Gemini 응답 파싱 실패: {e}")

    # JSON 파싱
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패. 응답 텍스트: {text}")
        raise ValueError(f"LLM 응답이 유효한 JSON이 아님: {e}")

    # Pydantic 모델로 검증
    try:
        return IntentResult(**parsed)
    except Exception as e:
        logger.error(f"IntentResult 검증 실패. 파싱 결과: {parsed}")
        raise ValueError(f"의도 해석 결과 검증 실패: {e}")