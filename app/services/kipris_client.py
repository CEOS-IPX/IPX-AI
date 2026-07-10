"""
============================================================
KIPRIS API 클라이언트
============================================================
KIPRIS Plus의 두 가지 API를 사용:

1. 상세 조회 (getBibliographyDetailInfoSearch)
   - Base URL: http://plus.kipris.or.kr/kipo-api/kipi
   - 인증: ServiceKey 파라미터
   - 사용자 lookup 및 증분 배치의 상세 조회에서 사용
   - 청구항 포함

2. 출원일자별 검색 (applicationDateSearchInfo)
   - Base URL: http://plus.kipris.or.kr/openapi/rest
   - 인증: accessKey 파라미터
   - 증분 배치의 사전 필터링에서 사용
   - 청구항 없음 (서지 정보 + 초록 + IPC만)
============================================================
"""

import logging
import xml.etree.ElementTree as ET
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.preprocessing import (
    clean_text,
    extract_independent_from_plain_list,
)

logger = logging.getLogger(__name__)


# ============================================================
# 응답 모델: KIPRIS 조회 결과
# ============================================================

class KiprisPatentDetail(BaseModel):
    """
    KIPRIS에서 가져온 특허 상세 정보 (getBibliographyDetailInfoSearch)
    BULK 적재 필드와 동일한 구조
      - CPC: API로 불가능 (제외)
      - 이미지경로: OpenSearch에 안 넣음 (제외)
    """

    # 출원/공개/등록 정보
    application_date: Optional[str] = Field(default=None, description="출원일자 (예: '2005.06.10')")
    application_number: str = Field(description="출원번호 (예: '10-2005-0050026')")
    open_date: Optional[str] = Field(default=None, description="공개일자")
    open_number: Optional[str] = Field(default=None, description="공개번호")
    register_date: Optional[str] = Field(default=None, description="등록일자")
    register_number: Optional[str] = Field(default=None, description="등록번호")
    register_status: Optional[str] = Field(default=None, description="등록상태 (예: '등록', '소멸')")

    # 발명 내용
    invention_title: Optional[str] = Field(default=None, description="발명의 명칭")
    abstract: Optional[str] = Field(default=None, description="초록 (전처리됨)")
    claims_independent: list[str] = Field(
        default_factory=list,
        description="독립 청구항 리스트 (전처리됨, BULK 적재와 동일한 'claims_independent' 스키마)"
    )
    ipc_codes: list[str] = Field(default_factory=list, description="IPC 분류 코드 목록")

    # 관계자 (한글명, 여러 명일 경우 쉼표로 join)
    applicants: Optional[str] = Field(default=None, description="출원인명 (예: '엘지전자 주식회사, 삼성전자 주식회사')")
    inventors: Optional[str] = Field(default=None, description="발명자명")


class KiprisPatentSummary(BaseModel):
    """
    KIPRIS 검색 API 응답 항목 (applicationDateSearchInfo)
    증분 배치의 IPC 사전 필터링용. 상세 정보는 어차피 상세 조회에서 다시 가져오므로
    필터링에 꼭 필요한 최소 필드만 유지.
    """

    application_number: str
    ipc_codes: list[str] = Field(default_factory=list)


# ============================================================
# KIPRIS 서비스
# ============================================================

class KiprisService:
    """KIPRIS Plus API 클라이언트"""

    DETAIL_SEARCH_PATH = "/patUtiModInfoSearchSevice/getBibliographyDetailInfoSearch"
    APPLICATION_DATE_SEARCH_PATH = "/patUtiModInfoSearchSevice/applicationDateSearchInfo"

    def __init__(self):
        self._detail_client: Optional[httpx.AsyncClient] = None    # kipo-api/kipi
        self._search_client: Optional[httpx.AsyncClient] = None    # openapi/rest

    @property
    def detail_client(self) -> httpx.AsyncClient:
        """상세 조회용 클라이언트 (kipo-api/kipi base)"""
        if self._detail_client is None:
            self._detail_client = httpx.AsyncClient(timeout=15.0)
        return self._detail_client

    @property
    def search_client(self) -> httpx.AsyncClient:
        """검색용 클라이언트 (openapi/rest base)"""
        if self._search_client is None:
            self._search_client = httpx.AsyncClient(timeout=30.0)
        return self._search_client

    # ============================================================
    # 1. 출원번호 단건 상세 조회
    # ============================================================

    async def fetch_by_application_number(
        self, application_number: str
    ) -> Optional[KiprisPatentDetail]:
        """
        출원번호로 특허 상세 정보를 조회한다.

        Args:
            application_number: 출원번호 (다양한 형식 허용)
              예: '1020050050026', '10-2005-0050026', 'KR10-2005-0050026'

        Returns:
            KiprisPatentDetail 또는 None (조회 실패 시)
        """
        normalized = self._normalize_application_number(application_number)

        params = {
            "ServiceKey": settings.kipris_api_key,
            "applicationNumber": normalized,
        }

        url = f"{settings.kipris_kipo_base_url}{self.DETAIL_SEARCH_PATH}"

        try:
            response = await self.detail_client.get(url, params=params)
            response.raise_for_status()
            xml_text = response.text
        except httpx.HTTPError:
            logger.exception(f"[KIPRIS] 상세 조회 실패: {application_number}")
            return None
        except Exception:
            logger.exception(f"[KIPRIS] 상세 조회 중 오류: {application_number}")
            return None

        result = self._parse_detail_response(xml_text)
        if result:
            logger.info(f"[KIPRIS] 상세 조회 성공: {application_number}")
        else:
            logger.warning(f"[KIPRIS] 상세 조회 결과 없음: {application_number}")

        return result

    # ============================================================
    # 2. 출원일자별 검색 (증분 배치용)
    # ============================================================

    async def search_by_application_date(
        self,
        application_date: str,   # "20260705"
        docs_start: int = 1,
        docs_count: int = 500,   # 최대 500
    ) -> tuple[list[KiprisPatentSummary], int]:
        """
        특정 출원일자의 특허 목록 조회.

        Args:
            application_date: 출원일자 (YYYYMMDD)
            docs_start: 페이지 시작 번호 (1부터)
            docs_count: 페이지당 건수 (기본 500)

        Returns:
            (특허 목록, 전체 건수) 튜플
        """
        params = {
            "applicationDate": application_date,
            "docsStart": docs_start,
            "docsCount": docs_count,
            "patent": "true",
            "utility": "true",
            "sortSpec": "AD",
            "descSort": "false",
            "accessKey": settings.kipris_api_key,
        }

        url = f"{settings.kipris_openapi_base_url}{self.APPLICATION_DATE_SEARCH_PATH}"

        try:
            response = await self.search_client.get(url, params=params)
            response.raise_for_status()
            xml_text = response.text
        except httpx.HTTPError:
            logger.exception(
                f"[KIPRIS] 검색 API 호출 실패: date={application_date}, start={docs_start}"
            )
            return [], 0

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.exception(f"[KIPRIS] 검색 응답 XML 파싱 실패: date={application_date}")
            return [], 0

        # 전체 건수
        total_count_str = self._get_text(root, ".//TotalSearchCount") or "0"
        try:
            total_count = int(total_count_str)
        except ValueError:
            total_count = 0

        # 특허 목록 파싱
        summaries: list[KiprisPatentSummary] = []
        for item in root.findall(".//PatentUtilityInfo"):
            summary = self._parse_summary(item)
            if summary:
                summaries.append(summary)

        logger.info(
            f"[KIPRIS] 검색 완료: date={application_date}, "
            f"start={docs_start}, 반환={len(summaries)}, 전체={total_count}"
        )
        return summaries, total_count

    # ============================================================
    # XML 파싱: 상세 조회
    # ============================================================

    def _parse_detail_response(self, xml_text: str) -> Optional[KiprisPatentDetail]:
        """상세 조회 API 응답 XML → KiprisPatentDetail"""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.exception("[KIPRIS] 응답 XML 파싱 실패")
            return None

        # ===== 응답 헤더 체크 =====
        success_yn = self._get_text(root, ".//successYN")
        if success_yn != "Y":
            result_msg = self._get_text(root, ".//resultMsg") or "알 수 없음"
            logger.warning(f"[KIPRIS] API 응답 실패: {result_msg}")
            return None

        # ===== item 노드 추출 =====
        item = root.find(".//body/item")
        if item is None:
            return None

        # ===== 서지정보 =====
        biblio = item.find(".//biblioSummaryInfo")
        if biblio is None:
            logger.warning("[KIPRIS] biblioSummaryInfo 누락")
            return None

        application_number = self._get_text(biblio, "applicationNumber")
        if not application_number:
            logger.warning("[KIPRIS] applicationNumber 누락")
            return None

        # ===== IPC 코드 =====
        ipc_codes: list[str] = []
        for ipc_info in item.findall(".//ipcInfo"):
            ipc_num = self._get_text(ipc_info, "ipcNumber")
            if ipc_num:
                ipc_codes.append(ipc_num)

        # ===== 초록 (전처리 적용) =====
        abstract_raw = self._get_text(item, ".//abstractInfo/astrtCont")
        abstract = clean_text(abstract_raw) if abstract_raw else None
        abstract = abstract if abstract else None  # 빈 문자열 → None

        # ===== 독립 청구항 추출 + 전처리 =====
        all_claims: list[str] = []
        for claim_info in item.findall(".//claimInfo"):
            claim_text = self._get_text(claim_info, "claim")
            if claim_text:
                all_claims.append(claim_text)

        independent_list = extract_independent_from_plain_list(all_claims)
        # BULK 적재와 동일하게 리스트 형태 유지, 항목별 전처리
        claims_independent = [clean_text(c) for c in independent_list if c] if independent_list else []

        # ===== 출원인 (한글명, 쉼표로 join) =====
        applicant_names: list[str] = []
        for applicant in item.findall(".//applicantInfo"):
            name = self._get_text(applicant, "name")
            if name:
                applicant_names.append(name)
        applicants = ", ".join(applicant_names) if applicant_names else None

        # ===== 발명자 (한글명, 쉼표로 join) =====
        inventor_names: list[str] = []
        for inventor in item.findall(".//inventorInfo"):
            name = self._get_text(inventor, "name")
            if name:
                inventor_names.append(name)
        inventors = ", ".join(inventor_names) if inventor_names else None

        return KiprisPatentDetail(
            application_date=self._get_text(biblio, "applicationDate"),
            application_number=application_number,
            open_date=self._get_text(biblio, "openDate"),
            open_number=self._get_text(biblio, "openNumber"),
            register_date=self._get_text(biblio, "registerDate"),
            register_number=self._get_text(biblio, "registerNumber"),
            register_status=self._get_text(biblio, "registerStatus"),
            invention_title=self._get_text(biblio, "inventionTitle"),
            abstract=abstract,
            claims_independent=claims_independent,
            ipc_codes=ipc_codes,
            applicants=applicants,
            inventors=inventors,
        )

    # ============================================================
    # XML 파싱: 검색 API
    # ============================================================

    def _parse_summary(self, item: ET.Element) -> Optional[KiprisPatentSummary]:
        """
        검색 응답의 PatentUtilityInfo → KiprisPatentSummary
        IPC 필터링에 필요한 최소 필드만 추출 (application_number, ipc_codes).
        """
        try:
            application_number = self._get_text(item, "ApplicationNumber")
            if not application_number:
                return None

            # IPC 코드는 파이프 구분자
            ipc_raw = self._get_text(item, "InternationalpatentclassificationNumber") or ""
            ipc_codes = [
                code.strip()
                for code in ipc_raw.split("|")
                if code.strip()
            ]

            return KiprisPatentSummary(
                application_number=application_number,
                ipc_codes=ipc_codes,
            )
        except Exception:
            logger.exception("[KIPRIS] 검색 결과 항목 파싱 실패")
            return None

    # ============================================================
    # 유틸리티
    # ============================================================

    @staticmethod
    def _get_text(element: ET.Element, path: str) -> Optional[str]:
        """
        XML 요소에서 경로의 텍스트 추출 (없거나 빈 문자열이면 None)

        KIPRIS 응답은 빈 값을 공백 한 칸 ' '으로 채우는 경우가 있어
        strip 후 빈 문자열도 None으로 처리한다.
        """
        node = element.find(path)
        if node is None or node.text is None:
            return None
        text = node.text.strip()
        return text if text else None

    @staticmethod
    def _normalize_application_number(application_number: str) -> str:
        """
        출원번호 형식 정규화

        변리사가 다양한 형식으로 입력 가능:
          - '1020050050026'
          - '10-2005-0050026'
          - 'KR10-2005-0050026'

        모두 '1020050050026' 형식으로 정규화 (API 호출용)
        """
        result = application_number.upper().strip()
        result = result.replace("KR", "").replace("-", "").replace(" ", "")
        return result

    # ============================================================
    # 리소스 정리
    # ============================================================

    async def close(self) -> None:
        """HTTP 클라이언트 정리 (서버 종료 시 호출)"""
        if self._detail_client is not None:
            await self._detail_client.aclose()
            self._detail_client = None
        if self._search_client is not None:
            await self._search_client.aclose()
            self._search_client = None


# ============================================================
# 싱글톤 인스턴스
# ============================================================
kipris_service = KiprisService()