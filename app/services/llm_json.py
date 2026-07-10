"""
============================================================
LLM 응답 JSON 파싱 유틸
============================================================
Claude는 "JSON만 출력, 코드블록 금지" 프롬프트 지시를 무시하고
```json ... ``` 코드펜스로 감싸서 응답한다.
json.loads 호출 전에 이 펜스를 제거해 파싱 실패를 방지한다.
============================================================
"""

import re

_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    """응답 텍스트 양끝의 마크다운 코드펜스(```json ... ```)를 제거"""
    return _CODE_FENCE_PATTERN.sub("", text.strip()).strip()
