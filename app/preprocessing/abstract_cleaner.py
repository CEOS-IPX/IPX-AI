"""
============================================================
초록/청구항 텍스트 정제 (공통)
============================================================
KIPRIS BULK TXT, KIPRIS API 응답, 증분 업데이트 모두에서 사용하는
공통 텍스트 정제 함수.

적용 순서:
  1. 유니코드 정규화 (NFKC: 전각 → 반각)
  2. 상투어 제거 ("본 발명은", "에 관한 것이다" 등)
  3. 노이즈 제거 (문단번호, 도면 참조, 특수기호 등)
  4. 연속 공백 정리

주의:
  XML/HTML 태그 제거는 포함하지 않음.
  BULK TXT는 XML 태그가 섞여 있어 strip_xml_tags() 별도 적용 필요.
  KIPRIS API는 응답에서 이미 평문이라 strip 불필요.
============================================================
"""

import re
import unicodedata


# ============================================================
# 정규식 패턴
# ============================================================

# 상투어 패턴: 특허 명세서에 반복적으로 등장하는 표현
# 검색/임베딩 정확도를 떨어뜨리는 노이즈 신호로 제거
BOILERPLATE_PATTERNS = [
    re.compile(r"본\s*발명은\s*"),
    re.compile(r"본\s*고안은\s*"),
    re.compile(r"에\s*관한\s*것으로[서,.]?\s*"),
    re.compile(r"에\s*관한\s*것이다[.]?\s*"),
    re.compile(r"상기\s*구성에\s*의하면\s*"),
    re.compile(r"이하[,]?\s*첨부된?\s*도면을?\s*참조하여\s*상세히\s*설명한다[.]?\s*"),
    re.compile(r"것을\s*특징으로\s*하는\s*"),
]

# 노이즈 패턴: 문단번호, 도면 참조, 특수기호 등
NOISE_PATTERNS = [
    re.compile(r"\[\d{4}\]"),                       # 문단번호 [0001]
    re.compile(r"도\s*\d+[a-zA-Z]?"),               # 도면 참조 "도 1", "도 1a"
    re.compile(r"FIG\.\s*\d+", re.IGNORECASE),      # FIG. 1
    re.compile(r"\(주\)"),                          # (주)
    re.compile(r"\(주식회사\)"),                    # (주식회사)
    re.compile(r"[※■▶▷●○◆◇]"),                   # 특수기호
]


# ============================================================
# 정제 함수
# ============================================================

def clean_text(text: str) -> str:
    """
    초록/청구항 텍스트 정제

    Args:
        text: 정제할 원본 텍스트

    Returns:
        정제된 텍스트. 입력이 비어있으면 빈 문자열.
    """
    if not text:
        return ""

    # 1. 유니코드 정규화 (전각 → 반각, 호환 문자 표준화)
    text = unicodedata.normalize("NFKC", text)

    # 2. 상투어 제거
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    # 3. 노이즈 제거
    for pattern in NOISE_PATTERNS:
        text = pattern.sub("", text)

    # 4. 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text