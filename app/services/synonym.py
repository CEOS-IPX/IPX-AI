"""
============================================================
동의어 사전 서비스
============================================================
동의어 사전 파일(synonyms_patent.txt)을 파싱해서 키워드를 확장한다.
============================================================
"""


import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class SynonymExpander:
    """동의어 사전 기반 키워드 확장기"""

    def __init__(self, dict_path: str):
        """
        Args:
            dict_path: synonyms_patent.txt 파일 경로
        """
        self.dict_path = dict_path
        self._groups: list[set[str]] = []
        self._word_to_group_index: dict[str, set[int]] = {}
        self._load()

    def _load(self) -> None:
        """사전 파일을 파싱해서 메모리에 로드"""
        path = Path(self.dict_path)
        if not path.exists():
            logger.warning(f"동의어 사전 파일을 찾을 수 없습니다: {path}")
            return

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # 파일 규칙: 빈 줄 또는 주석 건너뜀
                if not line or line.startswith("#"):
                    continue

                # 파일 규칙: 쉼표로 분리, 정규화
                words = {w.strip().lower() for w in line.split(",")}
                words = {w for w in words if w}  # 빈 문자열 제거

                if len(words) < 2:
                    continue  # 동의어 그룹은 최소 2개

                group_idx = len(self._groups)
                self._groups.append(words)

                # 역방향 인덱스 구축 (단어 → 그룹 인덱스 집합)
                for word in words:
                    self._word_to_group_index.setdefault(word, set()).add(group_idx)

        logger.info(f"[동의어 사전] {len(self._groups)}개 그룹 로드 완료")

    def expand(self, keywords: Iterable[str]) -> list[str]:
        """
        키워드 리스트를 받아 동의어/관련어를 추가한 확장 리스트 반환.

        - 원본 키워드는 항상 포함
        - 각 키워드가 속한 동의어 그룹의 모든 단어 추가
        - 중복 제거, 입력 순서 보존

        Args:
            keywords: 원본 키워드 리스트

        Returns:
            확장된 키워드 리스트
        """
        seen: set[str] = set()
        result: list[str] = []

        for kw in keywords:
            kw_norm = kw.strip().lower()
            if not kw_norm:
                continue

            # 원본 키워드 추가
            if kw_norm not in seen:
                seen.add(kw_norm)
                result.append(kw.strip())

            # 동의어 그룹의 다른 단어들 추가
            group_indices = self._word_to_group_index.get(kw_norm, set())
            for idx in group_indices:
                for synonym in self._groups[idx]:
                    if synonym not in seen:
                        seen.add(synonym)
                        result.append(synonym)

        return result

    def stats(self) -> dict:
        """로드된 사전 통계"""
        return {
            "groups": len(self._groups),
            "unique_words": len(self._word_to_group_index),
        }


# ============================================================
# 싱글톤 인스턴스: 서버 시작 시 1회 로드
# ============================================================

from app.config import settings

synonym_expander = SynonymExpander(settings.synonyms_file_path)