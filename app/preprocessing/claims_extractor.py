
"""
============================================================
독립 청구항 추출 (공통)
============================================================
두 가지 입력 형식 지원:
  1. BULK TXT의 XML 형식 (extract_independent_from_bulk_xml)
     '<claim num="1">...</claim><claim num="2">...</claim>'
  2. KIPRIS API의 평문 청구항 리스트 (extract_independent_from_plain_list)
     ['1. ...', '2. ...', '3. 삭제']

판별 규칙 (두 입력 모두 동일):
  - "삭제"만 있는 청구항 제외
  - "제N항에 있어서" 표현이 포함된 청구항 제외 (종속항)
  - 위 조건에 해당하지 않으면 독립항으로 판단

반환 형식 (두 입력 모두 동일):
  - 청구항 번호 prefix("1.", "2." 등)는 제거된 본문만 반환
============================================================
"""

import re


# ============================================================
# 정규식 패턴
# ============================================================

# 종속항 판별 패턴: "제N항에 있어서"
_DEPENDENT_PATTERN = re.compile(r"제\s*\d+\s*항에\s*있어서")

# BULK XML에서 <claim num="N">...</claim> 블록 추출
_CLAIM_BLOCK_PATTERN = re.compile(
    r'<claim\s+num="(\d+)">(.*?)</claim>',
    re.DOTALL
)

# 삭제 표시 청구항 판별 (BULK XML 전용)
_DELETED_CLAIM_PATTERN = '<AmendStatus status="D">'

# 일반 XML 태그 (BULK XML 내부 텍스트 추출용)
_XML_TAG_PATTERN = re.compile(r"<[^>]+>")

# 평문 청구항 번호 prefix ("1.", "제 1항", "청구항 1" 등)
# KIPRIS API 응답에서 청구항 본문 앞에 붙는 번호를 제거하기 위함
_CLAIM_NUMBER_PREFIX = re.compile(
    r"^\s*(제\s*\d+\s*항\.?|청구항\s*\d+\.?|\d+\.)\s*"
)


# ============================================================
# 1. BULK XML에서 독립 청구항 추출
# ============================================================

def extract_independent_from_bulk_xml(claims_xml: str) -> str:
    """
    BULK TXT의 XML 형식 청구항에서 독립항만 추출.

    Args:
        claims_xml: '<claim num="1">...</claim>' 형식의 XML 텍스트

    Returns:
        독립 청구항을 공백으로 join한 단일 문자열
        (청구항 번호 prefix는 이미 XML num 속성으로 분리되어 있어 본문만 포함)
    """
    if not claims_xml:
        return ""

    claim_blocks = _CLAIM_BLOCK_PATTERN.findall(claims_xml)

    independent: list[str] = []
    for _num, content in claim_blocks:
        # 삭제 표시된 청구항 제외
        if _DELETED_CLAIM_PATTERN in content:
            continue

        # 내부 텍스트만 추출 (XML 태그 제거)
        text = _XML_TAG_PATTERN.sub("", content).strip()

        # 종속 청구항 제외
        if _DEPENDENT_PATTERN.search(text):
            continue

        independent.append(text)

    return " ".join(independent)


# ============================================================
# 2. KIPRIS API의 평문 청구항 리스트에서 독립항 추출
# ============================================================

def extract_independent_from_plain_list(claims: list[str]) -> list[str]:
    """
    KIPRIS API에서 받은 평문 청구항 리스트에서 독립항만 추출.

    Args:
        claims: 평문 청구항 리스트
                예: ['1. 음향기기에서 ...', '2. 제1 항에 있어서, ...', '3. 삭제']

    Returns:
        독립 청구항 본문 리스트 (번호 prefix 제거됨)
        예: ['음향기기에서 ...', '음향기기의 출력 오디오신호를 ...']
    """
    if not claims:
        return []

    independent: list[str] = []
    for claim in claims:
        claim_clean = claim.strip()

        # "삭제" 단독 청구항 제외 (예: "3. 삭제")
        # 청구항 번호 prefix 제거 후 비교
        stripped = _CLAIM_NUMBER_PREFIX.sub("", claim_clean).strip()
        if stripped == "삭제":
            continue

        # 종속 청구항 제외 (원본 텍스트에서 검사)
        if _DEPENDENT_PATTERN.search(claim_clean):
            continue

        # 번호 prefix 제거하고 본문만 저장
        # (BULK XML 결과와 형식 일치를 위해)
        body_only = _CLAIM_NUMBER_PREFIX.sub("", claim_clean).strip()
        if body_only:
            independent.append(body_only)

    return independent