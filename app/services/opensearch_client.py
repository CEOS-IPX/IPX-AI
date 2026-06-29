"""
============================================================
OpenSearch 검색 서비스
============================================================
BM25 + nori 형태소 분석 + 동의어 확장을 활용한 키워드 검색.

쿼리 구성 (boost 값은 점차 튜닝해갈 예정!!)
  - 제목: 원본 쿼리 (사용자 의도의 핵심이므로 가장 강하게)
  - 제목: 키워드 텍스트, boost 3.0
  - 초록: 가상 초록(HyDE), boost 1.5 (동의어 확장 자동)
  - 청구항: 키워드 텍스트, boost 1.0 (동의어 확장 자동)
  - IPC: prefix 매칭, boost 2.0 (LLM 추정값이므로 IPC의 앞부분만 매칭되도록)
  - legal_status: terms 필터

응답:
  - application_number와 score만 받음 (RRF 병합용)
============================================================
"""

import logging
import asyncio
from typing import Optional

from opensearchpy import OpenSearch

from app.config import settings
from app.services.types import PatentScore

logger = logging.getLogger(__name__)


class OpenSearchService:
    """OpenSearch 비동기 검색 클라이언트"""

    def __init__(self):
        self._client: Optional[OpenSearch] = None

    @property
    def client(self) -> OpenSearch:
        if self._client is None:
            self._client = OpenSearch(
                hosts=[{
                    "host": settings.opensearch_host,
                    "port": settings.opensearch_port,
                }],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                ssl_show_warn=False,
                timeout=30,
                max_retries=2,
                retry_on_timeout=True,
            )
        return self._client

    async def search(
        self,
        expanded_keywords: list[str],
        hypothetical_abstract: str,
        user_query: str = None,
        ipc_codes: Optional[list[str]] = None,
        legal_status: Optional[list[str]] = None,
        size: int = 30,
    ) -> list[PatentScore]:
        """
        OpenSearch에서 키워드 기반 검색을 수행한다.

        Args:
            expanded_keywords: 동의어 확장이 완료된 키워드 리스트
            hypothetical_abstract: Claude HyDE 가상 초록
            user_query: 사용자 입력 자연어
            ipc_codes: LLM이 추정한 IPC 코드
            legal_status: 사용자가 선택한 등록상태
            size: 반환할 결과 개수

        Returns:
            점수 내림차순의 PatentScore 리스트
        """
        query = self._build_query(
            user_query=user_query,
            expanded_keywords=expanded_keywords,
            hypothetical_abstract=hypothetical_abstract,
            ipc_codes=ipc_codes,
            legal_status=legal_status,
            size=size,
        )

        try:
            response = await asyncio.to_thread(
                self.client.search,
                index=settings.opensearch_index,
                body=query,
            )
        except Exception:
            logger.exception("[OpenSearch] 검색 실패")
            return []

        results: list[PatentScore] = []
        for rank, hit in enumerate(response["hits"]["hits"], start=1):
            results.append(PatentScore(
                application_number=hit["_id"],
                score=float(hit["_score"]),
                source="opensearch",
                rank=rank,
            ))

        total = response["hits"]["total"]["value"]
        logger.info(f"[OpenSearch] 검색 완료: 반환 {len(results)}건 / 전체 매칭 {total}건")
        return results

    def _build_query(
        self,
        user_query: str,
        expanded_keywords: list[str],
        hypothetical_abstract: str,
        ipc_codes: Optional[list[str]] = None,
        legal_status: Optional[list[str]] = None,
        size: int = 30,
    ) -> dict:
        """OpenSearch Query 구조 생성"""

        keyword_text = " ".join(expanded_keywords) if expanded_keywords else ""

        # ===== 텍스트 매칭 (should 절) =====
        should_clauses: list[dict] = []

        # 원본 쿼리: 사용자 의도의 핵심 신호 (가장 강하게)
        if user_query:
            should_clauses.append({
                "match": {
                    "title": {
                        "query": user_query,
                        "boost": 6.0,
                    }
                }
            })

        # 제목 매칭 (높은 가중치)
        if keyword_text:
            should_clauses.append({
                "match": {
                    "title": {
                        "query": keyword_text,
                        "boost": 3.0,
                    }
                }
            })

        # 초록 매칭 (가상 초록 사용 + 동의어 자동 확장)
        if hypothetical_abstract:
            should_clauses.append({
                "match": {
                    "abstract_clean": {
                        "query": hypothetical_abstract,
                        "boost": 1.5,
                    }
                }
            })

        # 청구항 매칭
        if keyword_text:
            should_clauses.append({
                "match": {
                    "claims_independent": {
                        "query": keyword_text,
                        "boost": 1.0,
                    }
                }
            })

        # ===== IPC 코드 매칭 =====
        # LLM 추정 IPC는 100% 정확하지 않으므로 should + boost로 가산점 부여
        # 메인 그룹 prefix만 추출: "H01M 10/052" -> "H01M"
        if ipc_codes:
            ipc_prefixes = {
                code.split(" ")[0].strip()
                for code in ipc_codes
                if code and code.strip()
            }
            for prefix in ipc_prefixes:
                should_clauses.append({
                    "prefix": {
                        "ipc_codes": {
                            "value": prefix,
                            "boost": 2.0,
                        }
                    }
                })

        # ===== 정확 일치 필터 =====
        filter_clauses: list[dict] = []

        # 등록상태: 사용자가 명시적으로 선택한 경우이므로 필터링
        if legal_status:
            filter_clauses.append({
                "terms": {"legal_status": legal_status}
            })

        # ===== 최종 쿼리 구성 =====
        query: dict = {
            "size": size,
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 2,
                    "filter": filter_clauses,
                }
            },
            # RRF 병합 단계에서는 ID와 점수만 필요
            "_source": ["application_number"],
        }

        return query

    async def close(self) -> None:
        """클라이언트 정리 (서버 종료 시 호출)"""
        if self._client is not None:
            self._client.close()
            self._client = None


# ============================================================
# 싱글톤 인스턴스
# ============================================================
opensearch_service = OpenSearchService()