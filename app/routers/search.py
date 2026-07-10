"""
============================================================
검색 API 라우터 (전체 파이프라인 통합)
============================================================
Spring 서버로부터 검색 요청을 받아 검색 파이프라인을 실행하고
변리사용 상세 페이지에 필요한 모든 정보를 반환한다.

엔드포인트:
  POST /search                        - 검색 실행
  POST /search/{id}/cancel            - 중단 요청
  GET  /search/{id}/status            - 진행 상태 조회
  GET  /search/{id}/result            - 결과 조회
  POST /search/{id}/add-manual        - 탐색 후 수동 특허 추가

시나리오:
  A. 순수 검색: required_application_numbers 비어있음
     - 결과: 정확히 result_count건
  B. 탐색 전 수동 추가: required_application_numbers에 번호 지정
     - 결과: 정확히 (result_count + len(required))건
     - 검색 결과 result_count건 (required 제외) + required 특허 전체
  C. 탐색 후 수동 추가: /add-manual 엔드포인트
     - 기존 결과 + 추가된 특허들

파이프라인:
  Step 0 (3%):   required 특허 사전 확인 + KIPRIS fallback + 자동 적재
  Step 1 (5%):   LLM 의도 해석 (Gemini)
  Step 2 (7%):   동의어 확장
  Step 3 (10%):  IPC 통합
  Step 4 (25%):  HyDE 가상 초록 (Claude)
  Step 5 (30%):  BGE-M3 임베딩
  Step 6 (50%):  병렬 검색 (OpenSearch + pgvector)
  Step 7 (55%):  RRF 병합
  Step 7.5 (58%): required 특허 결과 포함 확인 (없으면 꼴지로 삽입)
  Step 8 (60%):  본문 데이터 준비
  Step 9 (95%):  병렬 LLM 추출
  Step 10 (100%): 응답 조립
============================================================
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.intent import interpret_intent, IntentResult
from app.services.synonym import synonym_expander
from app.services.hyde import generate_hypothetical_abstract
from app.services.embedding import embedding_service
from app.services.opensearch_client import opensearch_service, PatentSource
from app.services.pgvector_client import pgvector_service
from app.services.kipris_client import kipris_service
from app.services.ranking import merge_with_rrf, MergedPatent, RRF_K
from app.services.summary import summarize_batch, PatentSummary
from app.services.auto_ingest import ingest_from_kipris
from app.services.progress_tracker import (
    progress_tracker,
    SearchCancelledException,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


# required 특허가 검색 결과에 없을 때 부여할 rank
FALLBACK_RANK = 100

# RRF 병합 시 여유 버퍼 (required 제외한 상위 뽑기용)
BUFFER_SIZE = 10


# ============================================================
# 요청/응답 모델
# ============================================================

class SearchRequest(BaseModel):
    """Spring → Python 검색 요청 (탐색)"""

    search_id: str = Field(description="Spring이 생성한 작업 ID")
    title: str = Field(description="발명의 명칭")
    description: str = Field(description="발명의 핵심 기술 설명")
    technical_field: Optional[str] = Field(default=None, description="기술 분야")
    user_input_ipc: Optional[list[str]] = Field(default=None, description="변리사 직접 입력 IPC")
    result_count: int = Field(default=10, ge=1, le=30, description="결과 개수")
    required_application_numbers: list[str] = Field(
        default_factory=list,
        description="반드시 결과에 포함되어야 할 출원번호 (탐색 전 수동 추가)"
    )


class SearchContext(BaseModel):
    """검색 컨텍스트 (사용자 발명 정보) — add-manual에서 사용"""
    title: str = Field(description="사용자 발명 명칭")
    description: str = Field(description="사용자 발명 설명")
    user_keywords: list[str] = Field(default_factory=list, description="사용자 발명 키워드")


class PatentResult(BaseModel):
    """검색 결과 1건"""
    application_number: str
    title: Optional[str] = None
    applicant_name: Optional[str] = None
    application_date: Optional[str] = None
    registration_date: Optional[str] = None
    registration_number: Optional[str] = None
    legal_status: Optional[str] = None
    ipc_codes: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    purpose: Optional[str] = None
    features: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    reason: Optional[str] = None
    relevance: str
    rrf_score: float
    sources: list[str] = Field(default_factory=list)


class AddManualRequest(BaseModel):
    """탐색 후 수동 추가 요청 (Spring → Python)

    Spring이 사전에 다음을 처리한 상태로 요청해야 함:
      - cases 테이블에서 context 조회
      - prior_arts에서 existing_results 조회
      - 중복 특허 필터링 (application_numbers는 기존 결과와 중복 없음)
    """
    application_numbers: list[str] = Field(description="추가할 출원번호 리스트")
    context: SearchContext = Field(description="원본 검색의 사용자 발명 정보")
    existing_results: list[PatentResult] = Field(
        default_factory=list,
        description="기존 검색 결과 (병합 정렬용)"
    )


class SearchDebugInfo(BaseModel):
    expanded_keywords: list[str]
    trusted_ipc: list[str]
    estimated_ipc: list[str]
    hypothetical_abstract: str
    embedding_dim: int
    opensearch_count: int
    pgvector_count: int
    merged_unique: int


class SearchResponse(BaseModel):
    search_id: Optional[str] = None    # add-manual 응답에선 없을 수 있음
    is_valid: bool
    reason_invalid: Optional[str] = None
    intent: Optional[IntentResult] = None
    results: list[PatentResult] = Field(default_factory=list)
    debug: Optional[SearchDebugInfo] = None


class CancelResponse(BaseModel):
    search_id: str
    cancelled: bool


# ============================================================
# 유틸
# ============================================================

def _to_relevance(rank: int, total: int) -> str:
    """검색 결과 내 rank를 변리사용 등급으로 변환"""
    if total <= 0:
        return "낮음"
    percentile = rank / total
    if percentile <= 0.2:
        return "매우 높음"
    elif percentile <= 0.5:
        return "높음"
    elif percentile <= 0.8:
        return "보통"
    else:
        return "낮음"


def _calculate_forced_rrf(fallback_rank: int = FALLBACK_RANK, k: int = RRF_K) -> float:
    """required 특허가 두 소스 모두에 없을 때의 RRF 점수"""
    return 1.0 / (k + fallback_rank) + 1.0 / (k + fallback_rank)


# ============================================================
# 검색 엔드포인트
# ============================================================

@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """검색 파이프라인 진입점"""

    logger.info(
        f"[검색 요청] search_id={request.search_id}, title='{request.title}', "
        f"required={request.required_application_numbers}, "
        f"result_count={request.result_count}"
    )

    await progress_tracker.start(request.search_id)

    try:
        return await _execute_search(request)

    except SearchCancelledException:
        logger.info(f"[검색] 중단 처리 완료: {request.search_id}")
        return SearchResponse(
            search_id=request.search_id,
            is_valid=False,
            reason_invalid="검색이 사용자에 의해 중단되었습니다.",
        )

    except HTTPException:
        await progress_tracker.mark_failed(request.search_id, "검색 중 오류 발생")
        raise

    except Exception as e:
        logger.exception(f"[검색] 예상치 못한 오류: {request.search_id}")
        await progress_tracker.mark_failed(request.search_id, str(e))
        raise HTTPException(status_code=500, detail="검색 중 오류가 발생했습니다.")


async def _execute_search(request: SearchRequest) -> SearchResponse:
    """검색 파이프라인 본체"""

    # ===== Step 0: required 특허 사전 확인 + 자동 적재 (3%) =====
    if request.required_application_numbers:
        await progress_tracker.check_cancelled(request.search_id)
        await progress_tracker.update(request.search_id, "지정된 특허 확인 중", 3)
        await _ensure_required_exists(request.required_application_numbers)

    # ===== Step 1: LLM 의도 해석 (5%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "검색 의도 분석 중", 5)

    try:
        intent = await interpret_intent(
            title=request.title,
            description=request.description,
            technical_field=request.technical_field,
        )
    except ValueError as e:
        logger.warning(f"의도 해석 실패: {e}")
        raise HTTPException(status_code=502, detail="검색 의도 해석에 실패했습니다.")

    if not intent.is_valid:
        await progress_tracker.mark_completed(
            request.search_id,
            {
                "search_id": request.search_id,
                "is_valid": False,
                "reason_invalid": intent.reason_invalid,
                "intent": intent.model_dump(),
                "results": [],
            },
        )
        return SearchResponse(
            search_id=request.search_id,
            is_valid=False,
            reason_invalid=intent.reason_invalid,
            intent=intent,
        )

    logger.info(f"[의도 해석 완료] keywords={intent.keywords}, ipc={intent.ipc_codes}")

    # ===== Step 2: 동의어 확장 (7%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "키워드 확장 중", 7)
    expanded_keywords = synonym_expander.expand(intent.keywords)
    logger.info(f"[동의어 확장] {len(intent.keywords)}개 → {len(expanded_keywords)}개")

    # ===== Step 3: IPC 통합 (10%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "분류 코드 정리 중", 10)
    trusted_ipc = request.user_input_ipc or []
    estimated_ipc = [
        ipc for ipc in (intent.ipc_codes or [])
        if ipc not in trusted_ipc
    ]

    # ===== Step 4: HyDE 가상 초록 (25%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "유사 특허 모델 구성 중", 25)

    try:
        hypothetical_abstract = await generate_hypothetical_abstract(
            title=request.title,
            description=request.description,
            technical_field=request.technical_field,
            keywords=expanded_keywords,
            ipc_codes=trusted_ipc + estimated_ipc,
        )
    except Exception:
        logger.exception("HyDE 가상 초록 생성 실패")
        raise HTTPException(status_code=502, detail="가상 초록 생성 실패")

    # ===== Step 5: 임베딩 (30%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "의미 벡터 생성 중", 30)

    try:
        query_vector = embedding_service.embed(hypothetical_abstract)
    except Exception:
        logger.exception("임베딩 생성 실패")
        raise HTTPException(status_code=500, detail="임베딩 생성 실패")

    # ===== Step 6: 병렬 검색 (50%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "특허 데이터베이스 검색 중", 50)

    candidate_size = max(request.result_count * 3, 30)

    opensearch_task = opensearch_service.search(
        title=request.title,
        description=request.description,
        expanded_keywords=expanded_keywords,
        hypothetical_abstract=hypothetical_abstract,
        trusted_ipc=trusted_ipc,
        estimated_ipc=estimated_ipc,
        size=candidate_size,
    )
    pgvector_task = pgvector_service.search(
        query_vector=query_vector,
        trusted_ipc=trusted_ipc,
        estimated_ipc=estimated_ipc,
        size=candidate_size,
    )

    opensearch_result, pgvector_results = await asyncio.gather(
        opensearch_task, pgvector_task,
    )

    # ===== Step 7: RRF 병합 (55%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "검색 결과 통합 순위 계산 중", 55)

    required = request.required_application_numbers
    buffer_top_n = request.result_count + len(required) + BUFFER_SIZE

    merged_full: list[MergedPatent] = merge_with_rrf(
        opensearch_results=opensearch_result.scores,
        pgvector_results=pgvector_results,
        top_n=buffer_top_n,
    )

    # ===== Step 7.5: 정확한 개수 확보 (58%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "결과 개수 정리 중", 58)

    if required:
        merged = _apply_required_with_exact_count(
            merged_full=merged_full,
            required=required,
            result_count=request.result_count,
        )
    else:
        merged = merged_full[:request.result_count]

    if not merged:
        logger.warning("[검색] 병합 결과 없음")
        empty_response = SearchResponse(
            search_id=request.search_id,
            is_valid=True,
            intent=intent,
            results=[],
            debug=SearchDebugInfo(
                expanded_keywords=expanded_keywords,
                trusted_ipc=trusted_ipc,
                estimated_ipc=estimated_ipc,
                hypothetical_abstract=hypothetical_abstract,
                embedding_dim=len(query_vector),
                opensearch_count=len(opensearch_result.scores),
                pgvector_count=len(pgvector_results),
                merged_unique=0,
            ),
        )
        await progress_tracker.mark_completed(request.search_id, empty_response.model_dump())
        return empty_response

    # ===== Step 8: 본문 데이터 준비 (60%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "검색 결과 정리 중", 60)

    missing_sources = [
        m.application_number for m in merged
        if m.application_number not in opensearch_result.sources
    ]
    for num in missing_sources:
        src = await opensearch_service.get_by_application_number(num)
        if src:
            opensearch_result.sources[num] = src

    patent_data_for_summary: list[dict] = []
    for m in merged:
        src = opensearch_result.sources.get(m.application_number)
        patent_data_for_summary.append({
            "title": src.title if src else "",
            "abstract": src.abstract if src else "",
            "claims_independent": "\n".join(src.claims_independent) if src else "",
        })

    # ===== Step 9: 병렬 LLM 추출 (95%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "각 특허의 핵심 정보 추출 중", 95)

    summaries: list[Optional[PatentSummary]] = await summarize_batch(
        patent_data=patent_data_for_summary,
        user_title=request.title,
        user_description=request.description,
        user_keywords=intent.keywords,
    )

    # ===== Step 10: 응답 조립 (100%) =====
    await progress_tracker.check_cancelled(request.search_id)
    await progress_tracker.update(request.search_id, "최종 결과 준비 중", 99)

    results = _assemble_results(
        merged=merged,
        summaries=summaries,
        opensearch_sources=opensearch_result.sources,
        required_set=set(required),
    )

    response = SearchResponse(
        search_id=request.search_id,
        is_valid=True,
        intent=intent,
        results=results,
        debug=SearchDebugInfo(
            expanded_keywords=expanded_keywords,
            trusted_ipc=trusted_ipc,
            estimated_ipc=estimated_ipc,
            hypothetical_abstract=hypothetical_abstract,
            embedding_dim=len(query_vector),
            opensearch_count=len(opensearch_result.scores),
            pgvector_count=len(pgvector_results),
            merged_unique=len(merged),
        ),
    )

    await progress_tracker.mark_completed(request.search_id)
    return response


# ============================================================
# Step 0: required 특허 사전 확인 + 자동 적재
# ============================================================

async def _ensure_required_exists(application_numbers: list[str]) -> None:
    """required 특허들이 OpenSearch에 존재하는지 확인 → 없으면 KIPRIS 조회 → 자동 적재"""
    for num in application_numbers:
        exists = await opensearch_service.get_by_application_number(num)
        if exists:
            continue

        logger.info(f"[Step 0] OpenSearch에 없음, KIPRIS 조회: {num}")
        kipris_detail = await kipris_service.fetch_by_application_number(num)
        if not kipris_detail:
            logger.warning(f"[Step 0] KIPRIS에도 없음: {num}")
            continue

        success = await ingest_from_kipris(kipris_detail)
        if not success:
            logger.warning(f"[Step 0] 자동 적재 실패: {num}")


# ============================================================
# Step 7.5: 정확한 개수 확보
# ============================================================

def _apply_required_with_exact_count(
    merged_full: list[MergedPatent],
    required: list[str],
    result_count: int,
) -> list[MergedPatent]:
    """
    검색 결과 result_count건 (required 제외) + required 특허 전체.
    총 (result_count + len(required))건 반환.
    """
    required_set = set(required)

    non_required = [m for m in merged_full if m.application_number not in required_set]
    top_search_results = non_required[:result_count]

    natural_required = [m for m in merged_full if m.application_number in required_set]
    natural_required_nums = {m.application_number for m in natural_required}

    missing_required_nums = [num for num in required if num not in natural_required_nums]
    forced_rrf = _calculate_forced_rrf()
    forced_required = [
        MergedPatent(
            application_number=num,
            rrf_score=forced_rrf - (0.00001 * i),
            rank=0,
            sources=["manual"],
            source_ranks={"manual": FALLBACK_RANK},
        )
        for i, num in enumerate(missing_required_nums)
    ]

    final_merged = top_search_results + natural_required + forced_required
    final_merged.sort(key=lambda m: m.rrf_score, reverse=True)

    for new_rank, m in enumerate(final_merged, start=1):
        m.rank = new_rank

    return final_merged


# ============================================================
# 응답 조립
# ============================================================

def _assemble_results(
    merged: list[MergedPatent],
    summaries: list[Optional[PatentSummary]],
    opensearch_sources: dict[str, PatentSource],
    required_set: set[str],
) -> list[PatentResult]:
    """MergedPatent + PatentSummary + PatentSource → PatentResult 조립"""

    total_merged = len(merged)
    results: list[PatentResult] = []

    for m, summary in zip(merged, summaries):
        src = opensearch_sources.get(m.application_number)
        relevance = _to_relevance(m.rank, total_merged)

        result_sources = list(m.sources)
        if m.application_number in required_set and "manual" not in result_sources:
            result_sources.append("manual")

        results.append(PatentResult(
            application_number=m.application_number,
            title=src.title if src else None,
            applicant_name=src.applicant_name if src else None,
            application_date=src.application_date if src else None,
            registration_date=src.registration_date if src else None,
            registration_number=src.registration_number if src else None,
            legal_status=src.legal_status if src else None,
            ipc_codes=src.ipc_codes if src else [],
            summary=summary.summary if summary else None,
            purpose=summary.purpose if summary else None,
            features=summary.features if summary else [],
            keywords=summary.keywords if summary else [],
            reason=summary.reason if summary else None,
            relevance=relevance,
            rrf_score=m.rrf_score,
            sources=result_sources,
        ))

    return results


# ============================================================
# 탐색 후 수동 추가 엔드포인트
# ============================================================

@router.post("/add-manual", response_model=SearchResponse)
async def add_manual_patents(request: AddManualRequest) -> SearchResponse:
    """
    탐색 후 특허 수동 추가.

    Spring이 사전에 처리해야 할 것:
      - cases에서 context 조회
      - prior_arts에서 existing_results 조회
      - 중복 특허 필터링 (application_numbers는 기존 결과와 중복 없음)

    Python 처리:
      1. 자동 적재 확인 (OpenSearch에 없으면 KIPRIS로 가져와 적재)
      2. 서지 정보 조회
      3. LLM 정보 추출 (summary, purpose, features, keywords, reason)
      4. 새 특허들의 forced_rrf 부여
      5. 기존 결과 + 새 결과 병합 정렬
      6. 최종 응답 반환
    """
    # ===== 중복 확인 (Spring이 필터링했지만 만약을 대비) =====
    existing_nums = {r.application_number for r in request.existing_results}
    duplicates = [num for num in request.application_numbers if num in existing_nums]
    if duplicates:
        logger.warning(
            f"[AddManual] Spring이 필터링하지 못한 중복 발견 (스킵): {duplicates}"
        )

    new_numbers = [
        num for num in request.application_numbers
        if num not in existing_nums
    ]

    if not new_numbers:
        logger.info("[AddManual] 처리할 새 특허 없음")
        return SearchResponse(
            is_valid=True,
            results=request.existing_results,
        )

    # ===== 1. 자동 적재 확인 =====
    await _ensure_required_exists(new_numbers)

    # ===== 2. 서지 정보 조회 =====
    new_sources: dict[str, PatentSource] = {}
    for num in new_numbers:
        src = await opensearch_service.get_by_application_number(num)
        if src:
            new_sources[num] = src
        else:
            logger.warning(f"[AddManual] 서지 정보 없음, 스킵: {num}")

    if not new_sources:
        logger.warning("[AddManual] 모든 특허의 서지 정보 조회 실패")
        return SearchResponse(
            is_valid=True,
            results=request.existing_results,
        )

    # ===== 3. LLM 정보 추출 =====
    patent_data = [
        {
            "title": src.title or "",
            "abstract": src.abstract or "",
            "claims_independent": "\n".join(src.claims_independent) if src.claims_independent else "",
        }
        for src in new_sources.values()
    ]

    summaries = await summarize_batch(
        patent_data=patent_data,
        user_title=request.context.title,
        user_description=request.context.description,
        user_keywords=request.context.user_keywords,
    )

    # ===== 4. 새 PatentResult 조립 =====
    forced_rrf = _calculate_forced_rrf()
    new_results: list[PatentResult] = []

    for i, (num, src, summary) in enumerate(zip(new_sources.keys(), new_sources.values(), summaries)):
        new_results.append(PatentResult(
            application_number=num,
            title=src.title,
            applicant_name=src.applicant_name,
            application_date=src.application_date,
            registration_date=src.registration_date,
            registration_number=src.registration_number,
            legal_status=src.legal_status,
            ipc_codes=src.ipc_codes,
            summary=summary.summary if summary else None,
            purpose=summary.purpose if summary else None,
            features=summary.features if summary else [],
            keywords=summary.keywords if summary else [],
            reason=summary.reason if summary else None,
            relevance="낮음",
            rrf_score=forced_rrf - (0.00001 * i),
            sources=["manual"],
        ))

    # ===== 5. 기존 결과 + 새 결과 병합 정렬 =====
    all_results: list[PatentResult] = list(request.existing_results) + new_results
    all_results.sort(key=lambda r: r.rrf_score, reverse=True)

    # relevance 재계산 (최종 순위 기준)
    total = len(all_results)
    for new_rank, r in enumerate(all_results, start=1):
        r.relevance = _to_relevance(new_rank, total)

    logger.info(
        f"[AddManual] 완료: 기존 {len(request.existing_results)}건 + "
        f"새 {len(new_results)}건 = 총 {len(all_results)}건"
    )

    return SearchResponse(
        is_valid=True,
        results=all_results,
    )


# ============================================================
# 중단/상태/결과 조회
# ============================================================

@router.post("/{search_id}/cancel", response_model=CancelResponse)
async def cancel_search(search_id: str) -> CancelResponse:
    """진행 중인 검색 중단 요청"""
    success = await progress_tracker.cancel(search_id)
    return CancelResponse(search_id=search_id, cancelled=success)


@router.get("/{search_id}/status")
async def get_search_status(search_id: str) -> dict:
    """검색 진행 상태 조회"""
    status = await progress_tracker.get_status(search_id)
    if status is None:
        raise HTTPException(status_code=404, detail="존재하지 않는 검색입니다.")
    return status