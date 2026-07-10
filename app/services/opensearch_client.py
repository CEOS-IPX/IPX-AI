"""
============================================================
OpenSearch 검색 서비스
============================================================
BM25 + nori 형태소 분석 + 동의어 확장을 활용한 키워드 검색.

기능:
  - search(): 키워드 검색 (본문 함께 반환)
  - get_by_application_number(): 출원번호 단건 조회
  - index_document(): 문서 삽입 (자동 적재용)

Boost 구조:
  [매우 강한 신호 - 변리사 직접 입력]
  title → title 필드:                     boost 6.0
  title → abstract_clean 필드:            boost 4.0
  trusted_ipc → ipc_codes (prefix):       boost 4.0

  [강한 신호 - description 자연어 매칭]
  description → abstract_clean:           boost 3.0
  description → claims_independent:       boost 2.0

  [보조 신호 - LLM 추정/확장]
  expanded_keywords → title:              boost 2.5
  hypothetical_abstract → abstract_clean: boost 1.5
  expanded_keywords → claims_independent: boost 1.5
  estimated_ipc → ipc_codes (prefix):     boost 1.5
============================================================
"""

import logging
import asyncio
from typing import Optional, List

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError
from pydantic import BaseModel, Field

from app.config import settings
from app.services.types import PatentScore

logger = logging.getLogger(__name__)


# ============================================================
# 검색 결과 모델
# ============================================================

class PatentSource(BaseModel):
    """OpenSearch에서 가져온 특허 본문 정보"""
    application_number: str = Field(description="출원번호")
    title: Optional[str] = Field(default=None, description="발명의 명칭")
    abstract: Optional[str] = Field(default=None, description="초록 (전처리됨)")
    claims_independent: List[str] = Field(default_factory=list, description="독립 청구항 (전처리됨)")
    ipc_codes: list[str] = Field(default_factory=list, description="IPC 분류 코드")
    cpc_codes: list[str] = Field(default_factory=list, description="CPC 분류 코드")
    applicant_name: Optional[str] = Field(default=None, description="출원인명")
    inventor_name: Optional[str] = Field(default=None, description="발명자명")
    application_date: Optional[str] = Field(default=None, description="출원일자")
    open_date: Optional[str] = Field(default=None, description="공개일자")
    open_number: Optional[str] = Field(default=None, description="공개번호")
    registration_date: Optional[str] = Field(default=None, description="등록일자")
    registration_number: Optional[str] = Field(default=None, description="등록번호")
    legal_status: Optional[str] = Field(default=None, description="등록상태")


class OpenSearchResult(BaseModel):
    scores: list[PatentScore] = Field(description="RRF 병합용 점수 리스트")
    sources: dict[str, PatentSource] = Field(description="출원번호 → 본문 매핑")


# ============================================================
# OpenSearch 서비스
# ============================================================

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

    # ============================================================
    # 키워드 검색
    # ============================================================

    async def search(
            self,
            title: str,
            description: str,
            expanded_keywords: list[str],
            hypothetical_abstract: str,
            trusted_ipc: Optional[list[str]] = None,
            estimated_ipc: Optional[list[str]] = None,
            size: int = 30,
    ) -> OpenSearchResult:
        """OpenSearch에서 키워드 기반 검색 수행 (본문 함께 반환)"""
        query = self._build_query(
            title=title,
            description=description,
            expanded_keywords=expanded_keywords,
            hypothetical_abstract=hypothetical_abstract,
            trusted_ipc=trusted_ipc or [],
            estimated_ipc=estimated_ipc or [],
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
            return OpenSearchResult(scores=[], sources={})

        scores: list[PatentScore] = []
        sources: dict[str, PatentSource] = {}

        for rank, hit in enumerate(response["hits"]["hits"], start=1):
            app_num = hit["_id"]
            source_data = hit.get("_source", {})

            scores.append(PatentScore(
                application_number=app_num,
                score=float(hit["_score"]),
                source="opensearch",
                rank=rank,
            ))
            sources[app_num] = self._parse_source(app_num, source_data)

        total = response["hits"]["total"]["value"]
        logger.info(f"[OpenSearch] 검색 완료: 반환 {len(scores)}건 / 전체 매칭 {total}건")
        return OpenSearchResult(scores=scores, sources=sources)

    # ============================================================
    # 단건 조회
    # ============================================================

    async def get_by_application_number(
            self, application_number: str
    ) -> Optional[PatentSource]:
        """출원번호로 단건 조회"""
        try:
            response = await asyncio.to_thread(
                self.client.get,
                index=settings.opensearch_index,
                id=application_number,
            )
        except NotFoundError:
            return None
        except Exception:
            logger.exception(f"[OpenSearch] 단건 조회 실패: {application_number}")
            return None

        if not response.get("found"):
            return None

        return self._parse_source(application_number, response["_source"])

    # ============================================================
    # 문서 INSERT (자동 적재용)
    # ============================================================

    async def index_document(
            self,
            application_number: str,
            document: dict,
    ) -> bool:
        """
        문서를 OpenSearch에 INSERT (자동 적재용).

        Returns:
            True: 성공, False: 실패
        """
        try:
            await asyncio.to_thread(
                self.client.index,
                index=settings.opensearch_index,
                id=application_number,
                body=document,
                refresh=True,
            )
            logger.info(f"[OpenSearch] 문서 적재 완료: {application_number}")
            return True
        except Exception:
            logger.exception(f"[OpenSearch] 문서 적재 실패: {application_number}")
            return False

    # ============================================================
    # 문서 DELETE (롤백용)
    # ============================================================

    async def delete_document(self, application_number: str) -> bool:
        """
        문서를 OpenSearch에서 삭제 (자동 적재 롤백용).

        Returns:
            True: 삭제 성공 (또는 애초에 없음)
            False: 삭제 실패
        """
        try:
            await asyncio.to_thread(
                self.client.delete,
                index=settings.opensearch_index,
                id=application_number,
                refresh=True,
            )
            logger.info(f"[OpenSearch] 문서 삭제 완료: {application_number}")
            return True
        except NotFoundError:
            # 이미 없으면 성공으로 간주
            logger.info(f"[OpenSearch] 삭제 대상 문서 없음 (이미 삭제됨): {application_number}")
            return True
        except Exception:
            logger.exception(f"[OpenSearch] 문서 삭제 실패: {application_number}")
            return False

    # ============================================================
    # 내부 헬퍼
    # ============================================================

    def _parse_source(self, application_number: str, source_data: dict) -> PatentSource:
        """OpenSearch _source dict → PatentSource"""
        return PatentSource(
            application_number=application_number,
            title=source_data.get("title"),
            abstract=source_data.get("abstract_clean"),
            claims_independent=source_data.get("claims_independent") or [],
            ipc_codes=source_data.get("ipc_codes") or [],
            cpc_codes=source_data.get("cpc_codes") or [],
            applicant_name=source_data.get("applicant_name"),
            inventor_name=source_data.get("inventor_name"),
            application_date=source_data.get("application_date"),
            open_date=source_data.get("open_date"),
            open_number=source_data.get("open_number"),
            registration_date=source_data.get("registration_date"),
            registration_number=source_data.get("registration_number"),
            legal_status=source_data.get("legal_status"),
        )

    def _build_query(
            self,
            title: str,
            description: str,
            expanded_keywords: list[str],
            hypothetical_abstract: str,
            trusted_ipc: list[str],
            estimated_ipc: list[str],
            size: int,
    ) -> dict:
        """OpenSearch Query 구조 생성"""
        keyword_text = " ".join(expanded_keywords) if expanded_keywords else ""
        should_clauses: list[dict] = []

        if title:
            should_clauses.append({"match": {"title": {"query": title, "boost": 6.0}}})
            should_clauses.append({"match": {"abstract_clean": {"query": title, "boost": 4.0}}})

        for prefix in self._extract_ipc_prefixes(trusted_ipc):
            should_clauses.append({"prefix": {"ipc_codes": {"value": prefix, "boost": 4.0}}})

        if description:
            should_clauses.append({"match": {"abstract_clean": {"query": description, "boost": 3.0}}})
            should_clauses.append({"match": {"claims_independent": {"query": description, "boost": 2.0}}})

        if keyword_text:
            should_clauses.append({"match": {"title": {"query": keyword_text, "boost": 2.5}}})

        if hypothetical_abstract:
            should_clauses.append({"match": {"abstract_clean": {"query": hypothetical_abstract, "boost": 1.5}}})

        if keyword_text:
            should_clauses.append({"match": {"claims_independent": {"query": keyword_text, "boost": 1.5}}})

        for prefix in self._extract_ipc_prefixes(estimated_ipc):
            should_clauses.append({"prefix": {"ipc_codes": {"value": prefix, "boost": 1.5}}})

        return {
            "size": size,
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": "30%",
                }
            },
            "_source": [
                "application_number", "title", "abstract_clean", "claims_independent",
                "ipc_codes", "cpc_codes", "applicant_name", "inventor_name",
                "application_date", "open_date", "open_number",
                "registration_date", "registration_number", "legal_status",
            ],
        }

    @staticmethod
    def _extract_ipc_prefixes(ipc_codes: list[str]) -> set[str]:
        """'H01M 10/052' → 'H01M' prefix 추출"""
        return {
            code.split(" ")[0].strip()
            for code in ipc_codes
            if code and code.strip()
        }

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# ============================================================
# 싱글톤 인스턴스
# ============================================================
opensearch_service = OpenSearchService()