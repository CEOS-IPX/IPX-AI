"""
============================================================
LLM 의도 해석 서비스 (Gemini 2.5 Flash-Lite)
============================================================
변리사가 입력한 발명 정보를 분석해서 검색 파이프라인이 사용할 JSON 구조로 변환

입력:
  - title: 발명의 명칭
  - description: 핵심 기술 설명
  - technical_field: 기술 분야 (선택)

출력 JSON 예시:
{
  "is_valid": true,
  "reason_invalid": null,
  "keywords": ["생분해성수지", "나노입자", "표면개질", "코팅", "친환경"],
  "ipc_codes": ["C09D 5/00", "C08L 67/02"]
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
    ipc_codes: list[str] = Field(default_factory=list, description="LLM이 추정한 IPC 코드")


# ============================================================
# 프롬프트 구성
# ============================================================

SYSTEM_INSTRUCTION = """\
당신은 특허 검색을 도와주는 의도 해석 전문가입니다.
변리사가 출원하려는 발명의 정보를 분석해서 선행기술 검색에 사용할 구조화된 JSON을 생성하세요.

## 입력 형식
변리사가 출원하려는 발명에 대해 다음 정보를 제공합니다:
- "발명의 명칭": 발명을 한 줄로 표현한 제목 (가장 핵심 신호)
- "기술 분야": 발명이 속한 분야 (예: 고분자 화학, 이차전지) - 컨텍스트 정보
- "핵심 기술 설명": 발명의 구조, 작동 원리, 효과에 대한 자연어 설명 (가장 풍부한 정보)

세 정보를 종합해서 키워드와 IPC를 추출하세요.
- 명칭은 발명의 정체성, 설명은 발명의 디테일을 제공합니다.
- 기술 분야는 IPC 추정의 정확도를 높이는 보조 컨텍스트로 활용하세요.

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
  - 너무 모호해서 검색할 수 없는 입력 (예: 명칭/설명이 한두 단어)
  - 특허와 무관한 주제
  - 위 조건에 해당하지 않으면 true

- **reason_invalid**: is_valid=false일 때만 사용자에게 보여줄 친절한 메시지. true면 null

- **keywords**: 3~10개의 단어 단위 키워드
  - 명칭과 설명에서 핵심 기술 용어를 추출
  - 기술 키워드: 발명의 구조/재료/방식 (예: "리튬이온전지", "음극재", "PLA수지")
  - 문제/효과 키워드: 해결 과제 또는 효과 (예: "급속충전", "내수성향상", "생분해성")
  - 단어 또는 짧은 복합명사 (문장 X)
  - 한글/영문 혼용 가능 (예: "LIB", "이차전지")
  - 너무 일반적인 단어 제외 (예: "기술", "방법", "장치")

- **ipc_codes**: 관련 IPC 분류 코드 (메인 그룹 또는 서브 그룹 수준)
  - 기술 분야가 제공되었으면 그 분야에 맞는 IPC를 우선 고려
  - 예: ["H01M 10/052"], ["C09D 5/00", "C08L 67/02"]
  - 확실한 것만 1~5개. 모르겠으면 빈 배열

## 예시

입력:
발명의 명칭: 생분해성 고분자 코팅 조성물
기술 분야: 고분자 화학, 친환경 코팅
핵심 기술 설명: 생분해성 폴리에스터 수지에 표면 개질된 무기 나노입자를 분산시키고, 유기용제 없이 수계 분산 공정으로 코팅층을 형성하는 친환경 코팅 조성물.

출력:
{
  "is_valid": true,
  "reason_invalid": null,
  "keywords": ["생분해성", "폴리에스터수지", "나노입자", "표면개질", "수계분산", "코팅조성물", "PLA"],
  "ipc_codes": ["C09D 5/00", "C09D 167/00", "C08L 67/02"]
}

입력:
발명의 명칭: 급속충전 리튬이온전지 음극재
기술 분야: 이차전지
핵심 기술 설명: 흑연과 실리콘 나노복합체를 사용하여 급속충전 시 발생하는 열화를 줄이고 사이클 수명을 향상시키는 음극재.

출력:
{
  "is_valid": true,
  "reason_invalid": null,
  "keywords": ["급속충전", "리튬이온전지", "음극재", "실리콘나노복합체", "흑연", "열화", "사이클수명"],
  "ipc_codes": ["H01M 10/052", "H01M 4/38", "H01M 4/13"]
}

입력:
발명의 명칭: abc
기술 분야: (미지정)
핵심 기술 설명: 모르겠음

출력:
{
  "is_valid": false,
  "reason_invalid": "발명의 명칭과 핵심 기술 설명을 구체적으로 입력해 주세요.",
  "keywords": [],
  "ipc_codes": []
}
"""

# ============================================================
# Gemini API 호출
# ============================================================

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"


async def interpret_intent(
    title: str,
    description: str,
    technical_field: Optional[str] = None,
) -> IntentResult:
    """
    발명 정보를 의도 해석 결과로 변환

    Args:
        title: 발명의 명칭
        description: 핵심 기술 설명
        technical_field: 기술 분야 (선택)

    Returns:
        IntentResult: 의도 해석 결과

    Raises:
        ValueError: LLM 응답을 파싱할 수 없을 때
        httpx.HTTPError: API 호출 실패 시
    """

    user_text = (
        f"발명의 명칭: {title}\n"
        f"기술 분야: {technical_field or '(미지정)'}\n"
        f"핵심 기술 설명: {description}"
    )

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
            json=payload,
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