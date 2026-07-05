"""
============================================================
BULK TXT XML 태그 제거 (BULK 전용)
============================================================
KIPRIS BULK TXT 파일의 초록 필드에는 <br>, <p> 같은 HTML 태그가
포함되어 있어 정제 전 제거 필요.

KIPRIS API 응답은 ElementTree로 파싱하면 이미 평문이 추출되므로
이 모듈은 사용하지 않음. BULK 적재 스크립트에서만 사용.
============================================================
"""

import re


# XML/HTML 태그 패턴
_XML_TAG_PATTERN = re.compile(r"<[^>]+>")

# <br>, <br/>, <BR/> 등을 공백으로 치환 (줄바꿈 → 단어 구분)
_NEWLINE_TAG_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)


def strip_xml_tags(text: str) -> str:
    """
    XML/HTML 태그 제거

    <br> 태그는 공백으로 치환 (줄바꿈 자리에 단어 구분이 필요).
    나머지 태그는 모두 제거.

    Args:
        text: 태그가 포함된 원본 텍스트 (None 또는 빈 문자열 안전 처리)

    Returns:
        태그가 제거된 평문 텍스트
    """
    if not text:
        return ""

    # 1. <br> 태그는 공백으로 치환
    text = _NEWLINE_TAG_PATTERN.sub(" ", text)

    # 2. 나머지 태그 모두 제거
    text = _XML_TAG_PATTERN.sub("", text)

    return text.strip()