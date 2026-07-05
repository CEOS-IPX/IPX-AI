"""
============================================================
자동 적재 서비스
============================================================
KIPRIS API에서 가져온 특허 데이터를 우리 DB에 자동 적재한다.
- 임베딩 생성 (BGE-M3)
- OpenSearch INSERT
- pgvector INSERT

정합성 보장:
  - OpenSearch 성공 + pgvector 실패 → OpenSearch 롤백
  - 두 저장소 모두 성공한 경우에만 True 반환

사용처:
  - "출원번호로 불러오기" 시 OpenSearch에 없는 특허를
    KIPRIS에서 가져와서 자동 적재
============================================================
"""

import logging

from app.services.kipris_client import KiprisPatentDetail
from app.services.embedding import embedding_service
from app.services.opensearch_client import opensearch_service
from app.services.pgvector_client import pgvector_service

logger = logging.getLogger(__name__)


async def ingest_from_kipris(kipris_detail: KiprisPatentDetail) -> bool:
    """
    KIPRIS에서 가져온 특허를 우리 DB에 적재한다.

    흐름:
      1. 임베딩 생성
      2. OpenSearch INSERT
      3. pgvector INSERT
      두 저장소 모두 성공해야 True 반환.
      pgvector 실패 시 OpenSearch에서 롤백.

    Args:
        kipris_detail: KIPRIS API 조회 결과

    Returns:
        True: 두 저장소 모두 적재 성공
        False: 하나라도 실패 (정합성 유지된 상태)
    """
    app_num = kipris_detail.application_number

    # ===== 1. 임베딩 생성 =====
    embedding_text = _build_embedding_text(kipris_detail)
    if not embedding_text.strip():
        logger.warning(f"[AutoIngest] 임베딩 텍스트 비어있음: {app_num}")
        return False

    try:
        embedding = embedding_service.embed(embedding_text)
    except Exception:
        logger.exception(f"[AutoIngest] 임베딩 생성 실패: {app_num}")
        return False

    # ===== 2. OpenSearch INSERT =====
    document = _build_opensearch_document(kipris_detail)
    os_success = await opensearch_service.index_document(
        application_number=app_num,
        document=document,
    )
    if not os_success:
        logger.warning(f"[AutoIngest] OpenSearch 적재 실패: {app_num}")
        return False

    # ===== 3. pgvector INSERT =====
    pg_success = await pgvector_service.insert(
        application_number=app_num,
        embedding=embedding,
        ipc_codes=kipris_detail.ipc_codes,
    )
    if not pg_success:
        logger.warning(
            f"[AutoIngest] pgvector 적재 실패, OpenSearch 롤백 시도: {app_num}"
        )
        # 롤백: OpenSearch에서도 삭제하여 정합성 유지
        rollback_success = await opensearch_service.delete_document(app_num)
        if not rollback_success:
            # 롤백까지 실패한 경우 - 매우 드문 케이스
            # 운영자가 수동 처리해야 하는 상태
            logger.error(
                f"[AutoIngest] 롤백 실패, 정합성 깨진 상태: {app_num}. "
                f"OpenSearch에는 문서 있으나 pgvector에는 없음. 수동 처리 필요."
            )
        return False

    logger.info(f"[AutoIngest] 적재 완료: {app_num}")
    return True


# ============================================================
# 내부 헬퍼
# ============================================================

def _build_embedding_text(detail: KiprisPatentDetail) -> str:
    """
    임베딩 생성용 텍스트 조립.
    BULK 적재 스크립트와 동일한 형식: "제목. 초록"
    """
    title = detail.invention_title or ""
    abstract = detail.abstract or ""
    return f"{title}. {abstract}".strip(". ").strip()


def _build_opensearch_document(detail: KiprisPatentDetail) -> dict:
    """
    KiprisPatentDetail → OpenSearch 문서 형식.
    BULK 적재와 동일한 필드 구조 유지 (CPC는 KIPRIS에서 못 가져오므로 빈 배열).
    """
    return {
        "application_number": detail.application_number,
        "title": detail.invention_title,
        "abstract_clean": detail.abstract,
        "claims_independent": detail.claims_independent,
        "ipc_codes": detail.ipc_codes,
        "cpc_codes": [],
        "applicant_name": detail.applicants,
        "inventor_name": detail.inventors,
        "application_date": detail.application_date,
        "open_date": detail.open_date,
        "open_number": detail.open_number,
        "registration_date": detail.register_date,
        "registration_number": detail.register_number,
        "legal_status": detail.register_status,
    }