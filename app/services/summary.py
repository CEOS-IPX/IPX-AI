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
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.services.llm_json import strip_code_fence
from app.services.llm_retry import post_with_retry

logger = logging.getLogger(__name__)

CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"

# 동시 실행 제한 (Anthropic Rate Limit 방어)
# Tier 1 기준 분당 50요청, 검색당 최대 30건 감안
MAX_CONCURRENT_REQUESTS = 5

# ============================================================
# 응답 모델
# ============================================================

class PatentSummary(BaseModel):
    """LLM이 추출한 특허 상세 정보"""

    relevance_score: int = Field(
        ge=0, le=100,
        description="사용자 발명과의 관련성 점수 (0~100 정수)"
    )
    summary: str = Field(description="핵심 요약 1문장")
    purpose: str = Field(description="기술 목적 / 해결하려는 문제")
    features: list[str] = Field(description="주요 기술적 특징 3~5개")
    keywords: list[str] = Field(description="관련 키워드 5~8개")
    reason: str = Field(description="사용자 발명과의 관련성을 설명하는 추천 이유 2~3문장")


# ============================================================
# 유틸
# ============================================================

def _safe_title(title: Optional[str], max_len: int = 30) -> str:
    """로그용 특허 제목 안전 슬라이싱."""
    if not title:
        return "(제목 없음)"
    return title[:max_len]

# ============================================================
# 프롬프트
# ============================================================


SYSTEM_PROMPT = """\
당신은 15년 경력의 한국 특허 변리사이자 특허 분석 전문가입니다.
변리사가 출원하려는 발명 정보와 검색된 선행기술 특허 정보를 받아,
변리사가 한눈에 파악할 수 있도록 관련성 점수와 5가지 항목을 
구조화된 JSON으로 추출합니다.

## 출력 규칙

1. JSON 형식만 출력. 마크다운, 코드 블록, 설명 텍스트 절대 금지.
2. 한국어로 작성.
3. 모든 필드는 빈 값이 아닌 의미 있는 내용으로 채울 것.
4. 분량은 짧고 압축된 형태로. 변리사가 빠르게 훑을 수 있도록.
5. 명세서의 구체적 표현(화학식, 수치, 알고리즘명, 모델명 등)을 최대한 보존.

## 반드시 지킬 것 (부정 명령)

- 없는 매칭을 억지로 만들지 말 것. 관련성이 낮으면 정직하게 낮게 판정.
- 초록/청구항에 없는 내용을 추측하거나 창작 금지.
- summary에 사용자 발명 언급 금지 (선행기술 자체만 서술).
- reason에서만 사용자 발명과의 관계 언급.

## 출력 형식

{
  "relevance_score": <0~100 정수>,
  "summary": "핵심 요약 1문장 (80~120자)",
  "purpose": "기술 목적 1문장 (60~100자)",
  "features": ["주요 특징 1", "주요 특징 2", "..."],
  "keywords": ["키워드1", "키워드2", "..."],
  "reason": "추천 이유 2~3문장 (100~150자)"
}

## 각 필드 작성 가이드

### relevance_score (0~100 정수)
사용자 발명과 이 선행기술의 관련성을 정량 평가.
판단 기준:
- 80~100 (VERY_HIGH): 핵심 구성요소와 기술 방식이 매우 유사. 신규성 위협 가능성 큼.
- 60~79 (HIGH): 주요 구성요소가 유사하나 일부 차이. 세밀한 검토 필요.
- 40~59 (MEDIUM): 일부 구성요소만 겹침. 관련성 있으나 결정적 위협 아님.
- 20~39 (LOW): 기술 분야는 유사하나 구성요소 대부분 다름. 참고용.
- 0~19 (VERY_LOW): 사실상 무관.

**관련성이 낮으면 낮게 판정하는 것이 정직한 판단**. 
검색에서 반환됐다고 반드시 관련 있는 것은 아님.

### summary (80~120자, 1문장)
- 선행기술 특허 자체의 핵심 구성과 동작을 압축
- 사용자 발명은 언급 X
- 청구항 본질을 그대로 담을 것

### purpose (60~100자, 1문장)
- 이 선행기술이 종래 기술의 한계를 어떻게 극복하는지
- 어떤 문제를 해결하려는지

### features (3~5개)
- 이 선행기술을 구별짓는 핵심 특징을 짧은 명사구로
- 각 항목 30자 이내
- 개수는 3~5개 (필요에 따라 조정)

### keywords (3~8개)
- 이 선행기술의 핵심 키워드
- 짧은 단어 또는 명사구
- 도메인 특화 용어 우선
- 너무 일반적 단어(예: "기술", "방법", "장치") 제외

### reason (100~150자, 2~3문장)
구성 원칙:
- 관련성이 높은 경우 (relevance_score >= 60):
  * (1) 첫 문장: 선행기술 핵심 특징 압축 (수치, 조건, 모델명 등 구체적으로)
  * (2) 다음 문장: 사용자 발명의 어떤 개념이 이 선행기술의 어느 부분에 매칭되는지
    예: "입력하신 [X, Y, Z]가 청구항의 [A, B] 부분에 직접 일치합니다."
  * (3) 변리사가 즉시 유사성을 이해할 수 있어야 함

- 관련성이 낮은 경우 (relevance_score < 60):
  * 억지로 매칭을 만들지 말 것
  * 정직하게 "기술 분야는 유사하나 구성요소가 다름" 또는
    "핵심 알고리즘/방식이 상이함" 등으로 서술

## 예시

### 예시 1: 관련성 높음 (소프트웨어, G06N)

[사용자 발명]
- 명칭: 딥러닝 기반 실시간 얼굴 감정 분석 시스템
- 설명: CNN으로 얼굴 표정 추출 후 시계열 LSTM 분석으로 감정 상태 분류
- 키워드: CNN, LSTM, 감정 분석, 표정 인식

[선행기술]
- 명칭: 딥러닝 기반 얼굴 감정 인식 방법 및 장치
- 초록: 카메라 영상에서 CNN 모델로 얼굴 특징을 추출하고, 추출된 특징을 
  LSTM 신경망에 입력하여 시간에 따른 감정 변화를 5개 카테고리로 분류하는 
  시스템을 개시한다.

출력:
{
  "relevance_score": 88,
  "summary": "CNN으로 얼굴 특징을 추출하고 LSTM 신경망으로 시간별 감정 변화를 5개 카테고리로 분류하는 시스템.",
  "purpose": "실시간 영상에서 얼굴 표정 기반 감정 변화를 자동 인식하여 사용자 상태 분석에 활용.",
  "features": ["CNN 기반 얼굴 특징 추출", "LSTM 시계열 분석", "5개 카테고리 감정 분류", "실시간 영상 처리"],
  "keywords": ["CNN", "LSTM", "얼굴 인식", "감정 분류", "시계열 분석", "표정 인식"],
  "reason": "CNN 기반 얼굴 특징 추출과 LSTM 시계열 분석을 결합한 감정 분류 시스템. 입력하신 CNN·LSTM·감정 분석·표정 인식 네 개념이 모두 청구항에 직접 일치합니다."
}

### 예시 2: 관련성 중간 (소프트웨어, G06T)

[사용자 발명]
- 명칭: 저조도 이미지 노이즈 제거 시스템
- 설명: U-Net으로 노이즈 검출 후 GAN으로 복원
- 키워드: U-Net, GAN, 노이즈 제거, 저조도

[선행기술]
- 명칭: 딥러닝 기반 의료 영상 세그멘테이션
- 초록: U-Net 아키텍처를 활용한 의료 영상의 관심 영역 세그멘테이션 
  방법을 개시한다. CT/MRI 영상에서 종양 영역을 픽셀 단위로 분할하며, 
  데이터 증강 기법으로 학습 정확도를 향상시킨다.

출력:
{
  "relevance_score": 52,
  "summary": "U-Net 아키텍처로 의료 영상 내 종양 영역을 픽셀 단위로 세그멘테이션하고, 데이터 증강으로 정확도를 향상시키는 방법.",
  "purpose": "제한된 의료 영상 데이터에서 종양 영역을 자동 검출하여 진단 정확도 향상.",
  "features": ["U-Net 세그멘테이션", "픽셀 단위 종양 분할", "데이터 증강 학습", "의료 영상 특화"],
  "keywords": ["U-Net", "세그멘테이션", "의료 영상", "종양 검출", "데이터 증강"],
  "reason": "U-Net 아키텍처를 활용한 픽셀 세그멘테이션 기법. 입력하신 U-Net이 청구항에 직접 일치하나, 응용 도메인(의료 영상 vs 저조도 노이즈 제거)과 GAN 사용 여부에서 차이가 있습니다."
}

### 예시 3: 관련성 낮음 (다른 도메인)

[사용자 발명]
- 명칭: 딥러닝 기반 이미지 노이즈 제거
- 설명: CNN으로 저조도 이미지의 노이즈 자동 제거

[선행기술]
- 명칭: IoT 센서 데이터를 활용한 스마트 팜 관리 시스템
- 초록: 온도, 습도, 조도 센서로 수집한 데이터를 클라우드 서버에서 
  머신러닝 분석하여 자동으로 관수/조명을 제어하는 시스템.

출력:
{
  "relevance_score": 18,
  "summary": "IoT 센서로 수집한 온도, 습도, 조도 데이터를 클라우드 서버에서 머신러닝 분석하여 관수/조명을 자동 제어하는 스마트 팜 시스템.",
  "purpose": "농장 환경을 실시간 모니터링하고 자동 제어하여 작물 재배 효율 향상.",
  "features": ["IoT 센서 기반 환경 모니터링", "클라우드 머신러닝 분석", "자동 관수/조명 제어", "스마트 팜 통합 관리"],
  "keywords": ["IoT", "스마트 팜", "머신러닝", "클라우드 분석", "자동 제어"],
  "reason": "IoT 센서 기반 스마트 팜 자동 제어 시스템으로 머신러닝을 활용. 사용자 발명(이미지 노이즈 제거)과는 응용 도메인과 핵심 알고리즘이 상이하여 관련성이 낮습니다."
}

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

위 선행기술 특허의 관련성 점수와 5가지 항목을 JSON으로 추출하세요.
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
        "temperature": 0.1,
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # Prompt Caching
            }
        ],
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
        response = await post_with_retry(
            client, CLAUDE_ENDPOINT, headers=headers, json=payload, log_prefix="[Summary]",
        )
        data = response.json()
    except httpx.HTTPError:
        logger.exception(f"[Summary] Claude API 호출 실패: title='{patent_title[:30] if patent_title else ''}'")
        return None
    except Exception:
        logger.exception(f"[Summary] 예상치 못한 오류: title='{patent_title[:30] if patent_title else ''}'")
        return None

    # ============================================================
    # max_tokens 도달 감지
    # ============================================================
    stop_reason = data.get("stop_reason")
    if stop_reason == "max_tokens":
        logger.warning(
            f"[Summary] max_tokens 도달로 응답 잘림 가능: "
            f"title='{_safe_title(patent_title)}'"
        )

    # ============================================================
    # Prompt Caching 히트 로깅 (모니터링)
    # ============================================================
    usage = data.get("usage", {})
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    if cache_read > 0:
        logger.debug(f"[Summary] Cache hit: read={cache_read} tokens")
    elif cache_create > 0:
        logger.debug(f"[Summary] Cache created: {cache_create} tokens")

    # ============================================================
    # 응답 텍스트 추출
    # ============================================================
    try:
        text = data["content"][0]["text"].strip()
    except (KeyError, IndexError):
        logger.error(f"[Summary] Claude 응답 구조 이상: {data}")
        return None

    # JSON 파싱
    try:
        parsed = json.loads(strip_code_fence(text))
    except json.JSONDecodeError:
        logger.error(f"[Summary] JSON 파싱 실패: title='{patent_title[:30] if patent_title else ''}', text='{text[:200]}'")
        return None

    # ============================================================
    # Pydantic 검증 (relevance_score 범위 자동 검증)
    # ============================================================
    try:
        return PatentSummary(**parsed)
    except ValidationError as e:
        logger.error(
            f"[Summary] PatentSummary 검증 실패: "
            f"title='{_safe_title(patent_title)}', "
            f"error={e}, parsed={parsed}"
        )
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

    Semaphore로 동시 실행 수를 제한하여 Anthropic Rate Limit 방어.

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

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def _run_with_limit(patent: dict) -> Optional[PatentSummary]:
            async with semaphore:
                return await summarize_one(
                    patent_title=patent.get("title", ""),
                    patent_abstract=patent.get("abstract", ""),
                    patent_claims=patent.get("claims_independent", ""),
                    user_title=user_title,
                    user_description=user_description,
                    user_keywords=user_keywords,
                    client=client,
                )

        tasks = [_run_with_limit(p) for p in patent_data]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    success_count = sum(1 for r in results if r is not None)
    logger.info(
        f"[Summary] 병렬 요약 완료: "
        f"성공 {success_count}/{len(patent_data)}건, "
        f"동시 실행 제한={MAX_CONCURRENT_REQUESTS}"
    )

    return results