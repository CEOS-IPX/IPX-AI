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
from app.services.llm_retry import post_with_retry

logger = logging.getLogger(__name__)

CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 명세서 작성 전문가입니다.
변리사가 출원하려는 발명 정보를 바탕으로, 그와 유사한 선행기술 특허의 초록을 
가상으로 작성하는 것이 당신의 역할입니다.

## 목적

이 가상 초록은 검색 엔진의 벡터 유사도 비교(HyDE 기법)에 사용됩니다.
IPX 데이터베이스에 저장된 실제 특허 초록과 벡터 공간에서 최대한 가까워지도록,
실제 저장 데이터의 문체와 어휘를 정확히 모방해야 합니다.

## 출력 규칙

- 길이: 200~600자 (한국어 기준, 실제 데이터 길이 반영)
- 흐름: 발명의 명칭 → 구성/수단 → (선택적) 효과
- 문체: 아래 참고 예시의 실제 특허 초록 스타일을 정확히 따를 것
- 형식: 다른 설명, 마크다운, 코드블록 없이 초록 본문만 출력
- 언어: 한국어

## 실제 특허 초록의 문체 특성 (필수 반영)

- 본 발명은 ~에 관한 것으로", "본 발명의 목적은" 같은 상투어 금지
  (IPX DB는 상투어가 제거된 초록이 저장되어있므로 벡터 공간 왜곡)
- 알고리즘/모델명: CNN, Transformer, U-Net, BERT, GAN, LSTM 등 구체적 명명

## 반드시 지킬 것

- 발명 정보에 없는 내용을 창작 금지 (특히 알고리즘명, 구체적 수치, 성능)
- 응답에 초록 이외 텍스트 절대 포함 금지 (인사말, 설명, 주석)
- 키워드를 단순 나열하지 말고 문장에 자연스럽게 녹여 넣을 것
- 마크다운, 코드 블록 절대 사용 금지

## 참고: IPX DB 실제 저장 초록 예시

### 예시 1 (CCTV 관제, G06V/G06N 도메인)

위험도에 따라 영상 순서를 결정하는 자동 선별 관제 방법 및 장치. 
일 실시예에 따른 자동 선별 관제 방법은 관제 수행하는 복수의 
CCTV의 영상에서 객체를 추출하여 딥 러닝 기반의 영상분석 기초데이터를 
획득하는 단계, 추출된 객체로부터 CNN 기반의 객체를 분류하여 딥 러닝 
기반의 객체 분류를 수행하는 단계, 상기 객체 및 상기 객체에 대한 이벤트 
정보에 따라 각 CCTV의 관제 채널 별 점수를 산정하여 CCTV의 영상 
내 위험도 평가를 수행하는 단계, 및 산정된 채널 별 점수에 따라 상기 관제 
채널에 우선순위를 부여하고, 이벤트가 발생한 관제 채널 중 높은 우선순위의 
관제 채널부터 전체 모니터링 화면의 상단에 배치하는 단계를 포함할 수 있다.

### 예시 2 (AI 도서 추천, G06N/G06Q 도메인)

인공지능 기반 도서 큐레이션 플랫폼 운용 서버 및 그 동작 방법을 개시한다. 
인공지능 기반 도서 큐레이션 플랫폼 운용 서버는 적어도 하나의 프로세서 
및 적어도 하나의 프로세서가 적어도 하나의 단계를 수행하도록 지시하는 
명령어들을 저장하는 메모리를 포함할 수 있다. 여기서 적어도 하나의 단계는, 
사용자 단말로부터 사용자의 독서 태양, 선호 학습 형태, 생활 정보 및 사용자 
선택 정보 중 적어도 하나를 포함하는 사용자 정보를 수신하는 단계, 수신한 
사용자 정보를 기초로 사용자에게 적어도 하나 이상의 도서를 제공하고, 제공된 
도서에 대한 시선 추적 정보, 문장 터치 정보, 페이지 체류시간, 메모 및 
하이라이트 정보 중 적어도 하나를 포함하는 반응 정보를 수신하는 단계, 
잠재 의미 분석(LSA, Latent Semantic Analysis)를 통해 토픽 모델링을 
수행하여 도서 요약 정보를 생성하는 단계를 포함할 수 있다.

위 예시들의 문체와 어휘 패턴을 참고하여, 사용자가 제공한 발명 정보에 
맞는 가상 초록을 작성하세요. 
상투어 없이 자연스럽게 실제 특허 초록의 문체로 작성하는 것이 핵심입니다.
"""

USER_PROMPT_TEMPLATE = """\
아래 발명 정보에 대한 가상 특허 초록을 작성하세요.

- 발명의 명칭: {title}
- 기술 분야: {technical_field}
- 핵심 기술 설명: {description}
- 핵심 키워드: {keywords}
- 관련 IPC 코드: {ipc_codes}

가상 초록만 출력하세요. 다른 설명은 절대 포함하지 마세요.
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
    user_prompt = USER_PROMPT_TEMPLATE.format(
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

    # system 프롬프트를 배열로 구성 + cache_control로 캐싱 활성화
    payload = {
        "model": settings.claude_model,
        "max_tokens": 1024,
        "temperature": 0.5,  # 특허 문체 안정성 (0.7 → 0.5)
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # Prompt Caching
            }
        ],
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await post_with_retry(
            client, CLAUDE_ENDPOINT, headers=headers, json=payload, log_prefix="[HyDE]",
        )
        data = response.json()

    # ============================================================
    # max_tokens 도달 감지
    # ============================================================
    stop_reason = data.get("stop_reason")
    if stop_reason == "max_tokens":
        logger.warning(f"[HyDE] max_tokens 도달로 응답 잘림 가능")

    # ============================================================
    # 응답 텍스트 추출
    # ============================================================
    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        logger.error(f"Claude 응답 구조 이상: {data}")
        raise ValueError(f"Claude 응답 파싱 실패: {e}")

    logger.info(f"[HyDE] 가상 초록 {len(text)}자 생성 완료")
    return text