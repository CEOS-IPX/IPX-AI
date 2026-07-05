"""
============================================================
전처리 모듈
============================================================
KIPRIS 특허 데이터의 텍스트 정제와 청구항 추출을 담당.

주요 함수:
  abstract_cleaner.clean_text()                       - 공통 텍스트 정제
  claims_extractor.extract_independent_from_bulk_xml()    - BULK XML용 독립항 추출
  claims_extractor.extract_independent_from_plain_list()  - API 평문용 독립항 추출
  bulk_xml_stripper.strip_xml_tags()                  - BULK 전용 XML 제거
============================================================
"""

from app.preprocessing.abstract_cleaner import clean_text
from app.preprocessing.claims_extractor import (
    extract_independent_from_bulk_xml,
    extract_independent_from_plain_list,
)
from app.preprocessing.bulk_xml_stripper import strip_xml_tags

__all__ = [
    "clean_text",
    "extract_independent_from_bulk_xml",
    "extract_independent_from_plain_list",
    "strip_xml_tags",
]