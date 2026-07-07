"""
============================================================
증분 특허 적재 서비스
============================================================
주 1회 실행되는 배치. 지난 7일간 신규 등록된 특허 중
우리 도메인 IPC(G06N, G06T)에 해당하는 것들을 자동 적재.

흐름:
  1. 이전 실패 목록 로드 (재시도 대상)
  2. 지난 7일 각 날짜별로 KIPRIS 검색 API 호출
  3. IPC 필터링 (target_ipc_prefixes)
  4. 이미 DB에 있는 특허 제외
  5. 실패 목록 + 새 대상 병합
  6. 각 특허 상세 조회 + 자동 적재 (rate limiting)
  7. 실패 목록 갱신 (성공 제거, 이번에도 실패한 것 유지)
  8. 3회 이상 실패한 것은 영구 제거

실패 로그 파일 형식 (탭 구분):
  {application_number}\t{retry_count}
============================================================
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from app.config import settings
from app.services.auto_ingest import ingest_from_kipris
from app.services.kipris_client import kipris_service
from app.services.opensearch_client import opensearch_service

logger = logging.getLogger(__name__)


# ============================================================
# 실행 결과
# ============================================================

@dataclass
class IngestionReport:
    """배치 실행 결과"""
    total_candidates: int = 0        # 조회 대상 총 수
    already_exists: int = 0          # 이미 DB에 있어서 스킵
    detail_fetch_failed: int = 0     # 상세 조회 실패
    ingestion_failed: int = 0        # 적재 실패
    success: int = 0                 # 적재 성공
    permanently_dropped: int = 0     # 3회 이상 실패로 영구 제거

    def summary(self) -> str:
        return (
            f"성공 {self.success}, "
            f"이미 존재 {self.already_exists}, "
            f"상세 조회 실패 {self.detail_fetch_failed}, "
            f"적재 실패 {self.ingestion_failed}, "
            f"영구 제거 {self.permanently_dropped}, "
            f"총 대상 {self.total_candidates}"
        )


# ============================================================
# 실패 목록 관리
# ============================================================

def _load_failure_log() -> dict[str, int]:
    """
    실패 로그 파일에서 재시도 대상 로드.

    Returns:
        {출원번호: 재시도 횟수}
    """
    path = Path(settings.ingestion_failure_log_path)
    if not path.exists():
        return {}

    failures: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    logger.warning(f"[Ingestion] 실패 로그 형식 오류: {line}")
                    continue
                app_num, retry_str = parts
                try:
                    failures[app_num] = int(retry_str)
                except ValueError:
                    logger.warning(f"[Ingestion] 재시도 횟수 파싱 실패: {line}")
    except Exception:
        logger.exception("[Ingestion] 실패 로그 로드 실패")
        return {}

    logger.info(f"[Ingestion] 실패 로그 로드: {len(failures)}건")
    return failures


def _save_failure_log(failures: dict[str, int]) -> None:
    """실패 목록을 파일에 저장 (덮어쓰기)"""
    path = Path(settings.ingestion_failure_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as f:
            for app_num, retry_count in failures.items():
                f.write(f"{app_num}\t{retry_count}\n")
    except Exception:
        logger.exception("[Ingestion] 실패 로그 저장 실패")


# ============================================================
# IPC 매칭 확인
# ============================================================

def _matches_target_ipc(ipc_codes: list[str], target_prefixes: list[str]) -> bool:
    """특허의 IPC 코드 중 하나라도 target prefix로 시작하면 True"""
    for code in ipc_codes:
        code_upper = code.upper().strip()
        for prefix in target_prefixes:
            if code_upper.startswith(prefix.upper()):
                return True
    return False


# ============================================================
# 검색 API 페이지네이션 조회
# ============================================================

async def _fetch_all_summaries_for_date(date_str: str) -> list:
    """
    특정 날짜의 모든 특허 목록 조회 (페이지네이션).

    Returns:
        KiprisPatentSummary 리스트
    """
    all_summaries = []
    docs_start = 1
    docs_count = 500

    while True:
        summaries, total_count = await kipris_service.search_by_application_date(
            application_date=date_str,
            docs_start=docs_start,
            docs_count=docs_count,
        )

        if not summaries:
            break

        all_summaries.extend(summaries)

        # 마지막 페이지 도달
        if docs_start + docs_count > total_count:
            break

        docs_start += docs_count

        # Rate limiting
        await asyncio.sleep(1.0 / settings.kipris_rate_limit_per_second)

    return all_summaries


# ============================================================
# 배치 진입점
# ============================================================

async def run_weekly_ingestion(
    end_date: date = None,
    lookback_days: int = None,
) -> IngestionReport:
    """
    지난 lookback_days일 동안의 신규 특허를 우리 DB에 적재.

    Args:
        end_date: 조회 종료일 (기본: 오늘)
        lookback_days: 조회 기간 (기본: config의 ingestion_lookback_days)

    Returns:
        실행 결과 리포트
    """
    if end_date is None:
        end_date = date.today()
    if lookback_days is None:
        lookback_days = settings.ingestion_lookback_days

    logger.info(
        f"[Ingestion] 배치 시작: end_date={end_date}, lookback={lookback_days}일"
    )

    report = IngestionReport()

    # ===== 1. 이전 실패 목록 로드 =====
    failures = _load_failure_log()

    # ===== 2. 지난 N일 동안의 특허 출원번호 수집 =====
    # 검색 API에서 IPC 필터링에 필요한 정보만 이미 얻음.
    # 이후 상세 조회로 나머지 정보를 다시 가져올 것이므로
    # candidates는 출원번호 set으로 충분 (dict/summary 저장 불필요).
    candidates: set[str] = set()

    for i in range(lookback_days):
        target_date = end_date - timedelta(days=i + 1)
        date_str = target_date.strftime("%Y%m%d")

        try:
            summaries = await _fetch_all_summaries_for_date(date_str)
        except Exception:
            logger.exception(f"[Ingestion] {date_str} 검색 실패, 다음 날짜로 진행")
            continue

        # IPC 필터링
        filtered_count = 0
        for summary in summaries:
            if _matches_target_ipc(summary.ipc_codes, settings.target_ipc_prefixes):
                candidates.add(summary.application_number)
                filtered_count += 1

        logger.info(
            f"[Ingestion] {date_str}: 전체 {len(summaries)}건 → "
            f"IPC 필터 후 {filtered_count}건"
        )

    # ===== 3. 실패 목록 병합 (중복은 set이 자동 제거) =====
    candidates.update(failures.keys())

    report.total_candidates = len(candidates)
    logger.info(f"[Ingestion] 전체 대상: {report.total_candidates}건")

    # ===== 4. 각 특허 처리 =====
    new_failures: dict[str, int] = {}

    for app_num in candidates:
        # 이미 DB에 있으면 스킵
        existing = await opensearch_service.get_by_application_number(app_num)
        if existing:
            report.already_exists += 1
            # 실패 목록에 있었으면 제거 (이미 어딘가에서 적재됨)
            failures.pop(app_num, None)
            continue

        # 상세 조회
        try:
            detail = await kipris_service.fetch_by_application_number(app_num)
        except Exception:
            logger.exception(f"[Ingestion] 상세 조회 예외: {app_num}")
            detail = None

        if not detail:
            report.detail_fetch_failed += 1
            _increment_retry(new_failures, failures, app_num, report)
            await _rate_limit_sleep()
            continue

        # 자동 적재
        try:
            success = await ingest_from_kipris(detail)
        except Exception:
            logger.exception(f"[Ingestion] 적재 예외: {app_num}")
            success = False

        if success:
            report.success += 1
            # 성공했으니 실패 목록에서 제거
            failures.pop(app_num, None)
        else:
            report.ingestion_failed += 1
            _increment_retry(new_failures, failures, app_num, report)

        # Rate limiting
        await _rate_limit_sleep()

    # ===== 5. 실패 목록 저장 =====
    # 기존 failures 중 성공한 것은 이미 pop 됨, 아직 남은 것 유지
    # new_failures로 이번 실패도 병합
    for app_num, count in new_failures.items():
        failures[app_num] = count

    _save_failure_log(failures)

    logger.info(f"[Ingestion] 배치 완료: {report.summary()}")
    return report


# ============================================================
# 내부 유틸리티
# ============================================================

def _increment_retry(
    new_failures: dict[str, int],
    old_failures: dict[str, int],
    app_num: str,
    report: IngestionReport,
) -> None:
    """
    실패 카운트 증가.
    max_retry 초과 시 영구 제거 (실패 목록에 저장 안 함).
    """
    previous_count = old_failures.get(app_num, 0)
    new_count = previous_count + 1

    if new_count >= settings.ingestion_max_retry:
        report.permanently_dropped += 1
        logger.warning(
            f"[Ingestion] 영구 제거 ({new_count}회 실패): {app_num}"
        )
        # old_failures에서도 제거해서 저장 안 되도록
        old_failures.pop(app_num, None)
    else:
        new_failures[app_num] = new_count


async def _rate_limit_sleep() -> None:
    """KIPRIS API 초당 호출 제한 대응"""
    await asyncio.sleep(1.0 / settings.kipris_rate_limit_per_second)


# ============================================================
# 수동 실행용 (CLI)
# ============================================================

async def main():
    """수동 실행: python -m app.services.incremental_ingest"""
    from app.services.embedding import embedding_service
    embedding_service.load()

    report = await run_weekly_ingestion()
    print(f"\n=== 배치 결과 ===\n{report.summary()}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())