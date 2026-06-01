"""
============================================================
HyDE 서비스
============================================================
Claude로 가상 특허 초록을 생성한다.
이 초록은 BGE-M3로 임베딩되어 pgvector 유사도 검색에 사용된다.
============================================================
"""

import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"

PROMPT_TEMPLATE = """\
당신은 특허 명세서 작성 전문가입니다.
다음 정보를 바탕으로 실제 특허처럼 보이는 가상 초록을 작성하세요.

이 초록은 검색 엔진의 벡터 유사도 비교에 사용되므로,
실제 특허 명세서의 문체와 어휘를 충실히 따라야 합니다.

## 입력
- 핵심 키워드: {keywords}
- 관련 IPC 코드: {ipc_codes}

## 출력 규칙
- 길이: 200~400자
- 흐름: 발명의 목적, 해결 수단, 기대 효과가 자연스럽게 이어지도록
- 키워드를 자연스럽게 녹여 넣되, 단순 나열하지 말 것
- "본 발명은 ~에 관한 것으로", "본 발명의 목적은" 같은 상투어는 사용하지 말 것
  (실제 데이터는 전처리되어 상투어가 제거되어 있음)
- 다른 설명, 마크다운, 코드블록 없이 초록 본문만 출력
- 한국어로 작성

이제 가상 초록만 출력하세요.
"""


async def generate_hypothetical_abstract(
    keywords: list[str],
    ipc_codes: list[str]
) -> str:
    """
    확장된 키워드와 IPC 코드를 받아 Claude로 가상 초록을 생성한다.

    Args:
        keywords: 동의어 확장이 완료된 키워드 리스트
        ipc_codes: LLM이 추정한 IPC 코드 리스트

    Returns:
        생성된 가상 초록 텍스트 (200~400자)

    Raises:
        httpx.HTTPError: Claude API 호출 실패
        ValueError: 응답 파싱 실패
    """
    prompt = PROMPT_TEMPLATE.format(
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