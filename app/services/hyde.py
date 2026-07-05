"""
============================================================
HyDE 서비스
============================================================
Claude로 가상 특허 초록을 생성한다.
이 초록은 BGE-M3로 임베딩되어 pgvector 유사도 검색에 사용된다.

피벗 후 입력:
  - title: 발명의 명칭
  - description: 핵심 기술 설명
  - technical_field: 기술 분야 (선택)
  - keywords: 의도 해석 + 동의어 확장된 키워드
  - ipc_codes: 사용자 입력 + LLM 추정 IPC 합집합

변리사가 직접 쓴 설명까지 LLM에 전달하므로,
기존(키워드만 사용) 대비 가상 초록의 정확도가 향상된다.
============================================================
"""

import logging
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"

PROMPT_TEMPLATE = """\
당신은 특허 명세서 작성 전문가입니다.
변리사가 출원하려는 발명 정보를 바탕으로, 그와 유사한 선행기술 특허의 초록을
가상으로 작성하세요.

이 가상 초록은 검색 엔진의 벡터 유사도 비교에 사용되므로,
실제 특허 명세서의 문체와 어휘를 충실히 따라야 합니다.

## 입력
- 발명의 명칭: {title}
- 기술 분야: {technical_field}
- 핵심 기술 설명: {description}
- 핵심 키워드: {keywords}
- 관련 IPC 코드: {ipc_codes}

## 출력 규칙
- 길이: 200~400자
- 흐름: 발명의 목적, 해결 수단, 기대 효과가 자연스럽게 이어지도록
- 입력된 명칭과 설명을 충실히 반영하되, 키워드를 자연스럽게 녹여 넣을 것
- 키워드 단순 나열은 금지
- "본 발명은 ~에 관한 것으로", "본 발명의 목적은" 같은 상투어는 사용하지 말 것
  (실제 데이터는 전처리되어 상투어가 제거되어 있음)
- 다른 설명, 마크다운, 코드블록 없이 초록 본문만 출력
- 한국어로 작성

이제 가상 초록만 출력하세요.
"""


async def generate_hypothetical_abstract(
    title: str,
    description: str,
    technical_field: Optional[str],
    keywords: list[str],
    ipc_codes: list[str],
) -> str:
    """
    발명 정보를 받아 Claude로 가상 초록을 생성한다.

    Args:
        title: 발명의 명칭
        description: 핵심 기술 설명
        technical_field: 기술 분야 (선택)
        keywords: 동의어 확장이 완료된 키워드 리스트
        ipc_codes: 사용자 입력 + LLM 추정 IPC 합집합

    Returns:
        생성된 가상 초록 텍스트 (200~400자)

    Raises:
        httpx.HTTPError: Claude API 호출 실패
        ValueError: 응답 파싱 실패
    """
    prompt = PROMPT_TEMPLATE.format(
        title=title,
        technical_field=technical_field or "(미지정)",
        description=description,
        keywords=", ".join(keywords) if keywords else "(없음)",
        ipc_codes=", ".join(ipc_codes) if ipc_codes else "(미상)",
    )

    headers = {
        "x-api-key": settings.claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": settings.claude_model,
        "max_tokens": 1024,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(CLAUDE_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        logger.error(f"Claude 응답 구조 이상: {data}")
        raise ValueError(f"Claude 응답 파싱 실패: {e}")

    logger.info(f"[HyDE] 가상 초록 {len(text)}자 생성 완료")
    return text