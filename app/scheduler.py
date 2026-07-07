"""
============================================================
스케줄러 (APScheduler)
============================================================
주기적으로 실행되는 배치 작업 관리.

현재 등록된 작업:
  - weekly_ingestion: 매주 월요일 04:00 KST
    지난 7일간의 신규 특허 (G06N, G06T) 자동 적재
============================================================
"""

import logging
from datetime import date
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from app.services.incremental_ingest import run_weekly_ingestion

logger = logging.getLogger(__name__)

# 싱글톤 스케줄러 (main.py에서 시작/종료 관리)
_scheduler: Optional[AsyncIOScheduler] = None


# ============================================================
# 스케줄 대상 작업
# ============================================================

async def _scheduled_weekly_ingestion() -> None:
    """매주 실행되는 증분 배치 래퍼"""
    logger.info("[Scheduler] 주간 배치 시작")
    try:
        report = await run_weekly_ingestion(end_date=date.today())
        logger.info(f"[Scheduler] 주간 배치 완료: {report.summary()}")
    except Exception:
        logger.exception("[Scheduler] 주간 배치 예외")


# ============================================================
# 스케줄러 시작/종료
# ============================================================

def start_scheduler() -> None:
    """스케줄러 시작 (main.py의 lifespan에서 호출)"""
    global _scheduler

    if _scheduler is not None:
        logger.warning("[Scheduler] 이미 실행 중")
        return

    _scheduler = AsyncIOScheduler(timezone="Asia/Seoul")

    # 주간 증분 배치 등록
    _scheduler.add_job(
        _scheduled_weekly_ingestion,
        CronTrigger(
            day_of_week=settings.ingestion_cron_day_of_week,
            hour=settings.ingestion_cron_hour,
            minute=settings.ingestion_cron_minute,
        ),
        id="weekly_ingestion",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        f"[Scheduler] 시작됨 - 주간 배치: "
        f"매주 {settings.ingestion_cron_day_of_week} "
        f"{settings.ingestion_cron_hour:02d}:{settings.ingestion_cron_minute:02d} KST"
    )


def shutdown_scheduler() -> None:
    """스케줄러 종료 (main.py의 lifespan에서 호출)"""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("[Scheduler] 종료됨")