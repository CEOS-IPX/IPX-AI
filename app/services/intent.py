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
from app.services.llm_retry import post_with_retry

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
당신은 15년 경력의 한국 특허 변리사이며, 특허 검색을 위한 의도 해석 전문가입니다.
변리사가 출원하려는 발명의 정보를 분석해서 선행기술 검색에 사용할 구조화된 JSON을 생성하세요.

## 입력 형식
변리사가 출원하려는 발명에 대해 다음 정보를 제공합니다:
- "발명의 명칭": 발명을 한 줄로 표현한 제목 (가장 핵심 신호)
- "기술 분야": 발명이 속한 분야 (예: 딥러닝, 이차전지) - 컨텍스트 정보
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

## 반드시 지킬 것 (부정 명령)

- 발명 설명에 없는 내용을 추측하거나 창작 금지
- 확실하지 않은 IPC는 빈 배열 반환. 억지로 채우지 말 것
- keywords는 발명 정보에 실제로 등장한 단어 또는 명확한 파생 용어만
- 응답은 순수 JSON만. ```json 마크다운 절대 금지
- is_valid=false여도 keywords, ipc_codes는 빈 배열 []로 반드시 포함 (null이나 필드 생략 금지)

## 각 필드 작성 가이드

### is_valid
다음 기준으로 판단:

**false로 판정** (검색 진행 불가):
- 욕설, 잡담, 무의미한 입력
- 광고성 문구, 감정 표현만 있고 기술적 실체 없음
- 극단적으로 짧음 (명칭 5자 이하 AND 설명 20자 이하)
- 특허와 무관한 주제 (요리 레시피, 일기, 시 등)

**true로 판정** (검색 진행):
- 짧지만 기술 용어 포함 (예: "AI 자동차 진단")
- 아이디어 단계라 완성도 낮아도 검색 가능한 최소 정보 있음

### reason_invalid
- is_valid=true일 때는 반드시 null
- is_valid=false일 때만 사용
- 실행 가능한 조언 포함

### keywords (3~8개 권장)
- 명칭과 설명에서 핵심 기술 용어를 추출
- 기술 키워드: 발명의 구조/재료/방식/알고리즘
  (예: "리튬이온전지", "BERT", "U-Net", "PLA수지")
- 문제/효과 키워드: 해결 과제 또는 효과
  (예: "급속충전", "노이즈제거", "생분해성")
- 형식:
  * 단어 또는 짧은 복합명사 (문장 X)
  * 한글/영문 혼용 가능 (예: "LIB", "이차전지", "CNN", "GAN")
- 제외 대상:
  * 너무 일반적인 단어 (예: "기술", "방법", "장치", "시스템")
  * 발명 정보에 없는 추측 용어

### ipc_codes (0~5개)
- 관련 IPC 분류 코드 (메인 그룹 또는 서브 그룹 수준)
- 기술 분야가 제공되었으면 그 분야 IPC를 우선 고려
- 확실한 것만 반환. 모르겠으면 빈 배열

### ipc_codes (0~5개)
- 아래 "참고 IPC" 목록에서 선택 우선
- 확실한 것만. 모르겠으면 빈 배열
- 서브그룹까지 명시 (예: "G06N 20/00")
 
## 참고 IPC (IPX DB 실제 커버 범위)
 
IPX 데이터베이스는 주로 다음 분야의 약 15만 건 특허를 보유합니다.
발명 성격에 맞는 IPC를 이 목록에서 우선 선택하세요.
 
### 인공지능 및 머신러닝 (G06N) — 최다 비중
- `G06N 20/00`: 머신러닝 (일반) — 지도/비지도/강화학습 등
- `G06N 3/08`: 신경망 학습 방법 — 역전파, 최적화
- `G06N 3/0464`: CNN 계열 (합성곱 신경망)
- `G06N 3/045`: 신경망 세부 구조
- `G06N 3/04`: 신경망 아키텍처 (일반)
→ 사용 예: 딥러닝 모델, 분류/예측, GAN, Transformer, 강화학습
 
### 컴퓨터 그래픽스 & 영상 처리 (G06T) — 최다 비중
- `G06T 7/00`: 이미지 분석 (일반)
- `G06T 7/11`: 이미지 분할 (segmentation)
- `G06T 19/00`: 3D 모델링, AR/VR
- `G06T 5/00`: 이미지 향상/복원
- `G06T 3/00`: 기하학적 이미지 변환
→ 사용 예: 이미지 분할, 3D 렌더링, AR/VR, 화질 개선, 필터링
 
### 컴퓨터 비전 & 객체 인식 (G06V)
- `G06V 10/82`: 딥러닝 기반 이미지 인식
- `G06V 40/16`: 얼굴 인식
- `G06V 40/20`: 제스처/움직임 인식
→ 사용 예: 얼굴 인식, 객체 검출, 행동 분석, 생체 인식
 
### 비즈니스 & ICT 서비스 (G06Q)
- `G06Q 50/10`: 서비스 산업 ICT (미디어, 엔터테인먼트)
- `G06Q 30/02`: 마케팅, 광고
- `G06Q 30/06`: 전자상거래
→ 사용 예: 콘텐츠 추천, 온라인 쇼핑, 맞춤형 광고, 결제 시스템
 
### 디지털 헬스케어 & 의료 (G16H, A61B)
- `A61B 5/00`: 생체 신호 측정 (심박, 뇌파 등)
- `G16H 50/20`: AI 기반 의료 진단 예측
- `G16H 10/60`: 환자 데이터 관리
→ 사용 예: AI 질병 진단, 의료 영상 분석, 환자 모니터링
 
### 데이터 처리 & 인터페이스 (G06F)
- `G06F 3/01`: HCI (Human-Computer Interface), 사용자 입력
- `G06F 18/00`: 데이터 마이닝, 패턴 분류
- `G06F 16/33`: 자연어 처리, 텍스트 검색
→ 사용 예: NLP, 질의응답, UI/UX, 메타버스 입력
 
### 기타 응용 분야
- `H04N 7/18`: 영상 통신, CCTV, 보안 관제
- `G01N 21/88`: 광학 비파괴 검사, 산업 센서
- `G02B 27/01`: AR/VR 헤드셋, 스마트 글래스
- `B25J 9/16`: 로봇 제어
- `G05B`, `G08B`: 산업 제어, 이상 감지
 
**중요**: 위 목록에 명확히 매칭되지 않으면 빈 배열 반환 허용.
발명 자체의 유효성(is_valid)은 IPC 매칭과 무관하게 판단.

## 예시

### 예시 1: 머신러닝 (G06N)

입력:
발명의 명칭: 사용자 발화 데이터를 활용한 개인화 음성 인식 시스템
기술 분야: 딥러닝 기반 음성 인식
핵심 기술 설명: 사용자별 발화 데이터를 수집하고 BERT 기반 임베딩으로 
              특징 벡터를 추출한 뒤, 개인별 파인튜닝 모델로 
              음성 명령 인식 정확도를 향상시킨다.

출력:
{
  "is_valid": true,
  "reason_invalid": null,
  "keywords": ["음성인식", "발화데이터", "BERT", "임베딩", "파인튜닝", "개인화", "특징벡터"],
  "ipc_codes": ["G06N 3/08", "G10L 15/16", "G06N 20/00"]
}

### 예시 2: 이미지 처리 (G06T)

입력:
발명의 명칭: 저조도 환경에서의 실시간 이미지 노이즈 제거 시스템
기술 분야: 딥러닝 기반 이미지 향상
핵심 기술 설명: U-Net 기반 노이즈 검출 네트워크로 저조도 이미지를 분석하고,
              GAN 기반 복원 알고리즘으로 실시간 노이즈 제거.

출력:
{
  "is_valid": true,
  "reason_invalid": null,
  "keywords": ["저조도", "이미지노이즈", "U-Net", "노이즈검출", "GAN", "이미지복원", "실시간"],
  "ipc_codes": ["G06T 5/00", "G06N 3/04", "G06T 7/00"]
}

### 예시 3: 정보 부족 (invalid)

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

USER_PROMPT_TEMPLATE = """\
발명의 명칭: {title}
기술 분야: {technical_field}
핵심 기술 설명: {description}
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

    user_text = USER_PROMPT_TEMPLATE.format(
        title=title,
        technical_field=technical_field or "(미지정)",
        description=description,
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
        response = await post_with_retry(
            client,
            GEMINI_ENDPOINT,
            params={"key": settings.gemini_api_key},
            json=payload,
            log_prefix="[Intent]",
        )
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