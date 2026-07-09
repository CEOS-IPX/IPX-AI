"""
청구항 추출 모듈

두 가지 입력 형식 지원 (형식이 서로 다르므로 별도 처리):

1. BULK 파일 (KIPRIS 다운로드)
   형식:
     <claim num="1">
       <claim-text>아우터 케이스;<br></br>상기 아우터 케이스...</claim-text>
     </claim>
     <claim num="2">
       <claim-text>제 1 항에 있어서,<br></br>...</claim-text>
     </claim>
     <claim num="13">
       <AmendStatus status="D">삭제</AmendStatus>
     </claim>
   - 청구항 번호는 <claim num="N"> 속성에 있음
   - 본문은 <claim-text> 안에
   - 줄바꿈은 <br></br> 태그
   - 삭제는 <AmendStatus status="D">삭제</AmendStatus>

2. KIPRIS API 개별 조회
   형식:
     <claim>1. 음향기기에서 출력되는...</claim>
     <claim>2. 제1 항에 있어서,...</claim>
     <claim>3. 삭제</claim>
   - 청구항 번호가 본문 시작에 있음 ("1.", "2." 등)
   - 삭제는 본문 텍스트 "삭제"

두 방식 모두 최종 출력은:
  ["청구항 1: 본문...", "청구항 5: 본문...", ...]
  (독립항만, 종속항과 삭제 제외)
"""

import html
import logging
import re

logger = logging.getLogger(__name__)


# ============================================================
# HTML/XML 정리 유틸
# ============================================================

def _clean_html_entities(text: str) -> str:
    """HTML 엔티티(&nbsp;, &amp; 등)를 실제 문자로 변환"""
    return html.unescape(text)


def _strip_xml_tags(text: str) -> str:
    """XML/HTML 태그 제거 (내용은 유지)"""
    return re.sub(r"<[^>]+>", " ", text)


def _normalize_whitespace(text: str) -> str:
    """연속 공백을 하나로, 앞뒤 공백 제거"""
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# 종속항 감지
# ============================================================

def _is_dependent_claim(claim_body: str) -> bool:
    """
    종속항인지 판단.
    "제N항에 있어서" 또는 "제N항 및 제M항에 있어서" 등의 패턴 감지.
    """
    return bool(re.search(r"제\s*\d+\s*항.*있어서", claim_body[:100]))


# ============================================================
# 통일된 형식 조립
# ============================================================

def _format_claim(number: int, body: str) -> str:
    """통일된 형식 '청구항 N: 본문' 으로 반환"""
    return f"청구항 {number}: {body}"


# ============================================================
# 청구항 번호 추출 및 prefix 제거 (KIPRIS API용)
# ============================================================

def _extract_claim_number_from_text(claim_text: str) -> int | None:
    """
    청구항 본문 텍스트에서 번호 추출 (KIPRIS API 형식).
    지원 형식: "1.", "청구항 1", "제 1항", "[청구항 1]" 등
    """
    head = claim_text[:100].strip()

    match = re.match(r"^\[?\s*청구항\s*(\d+)\s*\]?", head)
    if match:
        return int(match.group(1))

    match = re.match(r"^제\s*(\d+)\s*항", head)
    if match:
        return int(match.group(1))

    match = re.match(r"^(\d+)\s*\.", head)
    if match:
        return int(match.group(1))

    return None


def _remove_claim_prefix(claim_text: str) -> str:
    """청구항 번호 prefix 제거 (본문만 반환)"""
    stripped = claim_text.strip()

    stripped = re.sub(r"^\[?\s*청구항\s*\d+\s*\]?\s*[:.：]?\s*", "", stripped)
    stripped = re.sub(r"^제\s*\d+\s*항\s*[:.：]?\s*", "", stripped)
    stripped = re.sub(r"^\d+\s*\.\s*", "", stripped)

    return stripped.strip()


# ============================================================
# BULK 파일 파싱
# ============================================================

# BULK 형식: <claim num="N">...</claim>
# ()를 통해 캡쳐 그룹 지정
_BULK_CLAIM_PATTERN = re.compile(
    r'<claim\s+num\s*=\s*"(\d+)"\s*>(.*?)</claim>',
    re.IGNORECASE | re.DOTALL,
)

# 삭제 청구항 감지: <AmendStatus status="D">
_AMEND_DELETED_PATTERN = re.compile(
    r'<AmendStatus\s+status\s*=\s*"D"\s*>',
    re.IGNORECASE,
)


def extract_independent_from_bulk_xml(claim_xml_text: str) -> list[str]:
    """
    BULK 파일의 XML 텍스트에서 독립 청구항만 추출.

    입력 예시:
        <claim num="1"><claim-text>아우터 케이스;<br></br>...</claim-text></claim>
        <claim num="2"><claim-text>제 1 항에 있어서,...</claim-text></claim>
        <claim num="13"><AmendStatus status="D">삭제</AmendStatus></claim>

    Args:
        claim_xml_text: BULK 파일의 청구항 필드 원본 텍스트

    Returns:
        ["청구항 1: 본문...", "청구항 9: 본문...", ...]
        (독립항만, 종속항과 "삭제" 제외)
    """
    if not claim_xml_text or not isinstance(claim_xml_text, str):
        return []

    result: list[str] = []

    # <claim num="N">...</claim> 패턴 매칭
    for match in _BULK_CLAIM_PATTERN.finditer(claim_xml_text):
        number = int(match.group(1))
        inner = match.group(2)

        # 삭제 청구항인지 확인 (본문에 <AmendStatus status="D"> 있으면)
        if _AMEND_DELETED_PATTERN.search(inner):
            continue

        # HTML 엔티티 처리
        text = _clean_html_entities(inner)

        # XML 태그 제거 (<claim-text>, <br></br> 등)
        text = _strip_xml_tags(text)

        # 공백 정리
        body = _normalize_whitespace(text)

        if not body:
            continue

        # 종속항 판단
        if _is_dependent_claim(body):
            continue

        # 통일된 형식으로 저장
        result.append(_format_claim(number, body))

    return result


# ============================================================
# KIPRIS API 응답 처리
# ============================================================

def extract_independent_from_plain_list(claims_list: list[str]) -> list[str]:
    """
    KIPRIS API 개별 조회 응답의 청구항 리스트에서
    독립 청구항만 추출.

    입력 예시:
        ["1. 음향기기에서 출력되는...",
         "2. 제1 항에 있어서, ...",
         "3. 삭제"]

    Args:
        claims_list: KIPRIS <claim> 태그별 텍스트 리스트

    Returns:
        ["청구항 1: 본문...", ...]
    """
    if not claims_list:
        return []

    result: list[str] = []

    for claim in claims_list:
        if not claim:
            continue

        # HTML 엔티티 처리 + 공백 정리
        text = _clean_html_entities(claim)
        text = _normalize_whitespace(text)

        if not text:
            continue

        # 청구항 번호 추출 (본문 시작에서)
        number = _extract_claim_number_from_text(text)

        # 번호 prefix 제거
        body = _remove_claim_prefix(text)

        # "삭제" 청구항 체크 (prefix 제거 후 본문이 "삭제"인 경우)
        if re.fullmatch(r"삭\s*제[.]?\s*", body):
            continue

        if not body:
            continue

        # 종속항 판단 (prefix 제거된 본문에서)
        if _is_dependent_claim(body):
            continue

        if number is None:
            logger.warning(
                f"[claims_extractor] 청구항 번호 추출 실패, 스킵: {text[:80]}..."
            )
            continue

        result.append(_format_claim(number, body))

    return result