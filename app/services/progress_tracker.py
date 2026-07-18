"""
============================================================
검색 진행률 추적 서비스 (Redis 기반)
============================================================
Redis 키 구조:
  search:{case_id}                → 진행 상태 (Hash)
    - status: pending/in_progress/completed/cancelled/invalid_input/no_results/failed
    - step: 현재 단계 설명
    - progress: 진행률 (0~100)
    - started_at, updated_at
    - error (failed 시)

TTL: 1시간
============================================================
"""

import json
import logging
from datetime import datetime, UTC
from typing import Optional

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


def _progress_key(case_id: str) -> str:
    return f"search:{case_id}"

TTL_SECONDS = 3600


class SearchCancelledException(Exception):
    """검색이 사용자 요청으로 중단됨"""
    pass


class ProgressTracker:
    """Redis 기반 진행률/중단/결과 관리 (싱글톤)"""

    def __init__(self):
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                decode_responses=True,
            )
        return self._client

    # ============================================================
    # 검색 시작
    # ============================================================

    async def start(self, case_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self.client.hset(
            _progress_key(case_id),
            mapping={
                "status": "in_progress",
                "step": "검색 준비 중",
                "progress": "0",
                "started_at": now,
                "updated_at": now,
                "reason_invalid": "",
                "error": "",
            },
        )
        await self.client.expire(_progress_key(case_id), TTL_SECONDS)
        logger.info(f"[Progress] 검색 시작: {case_id}")

    # ============================================================
    # 진행률 업데이트
    # ============================================================

    async def update(self, case_id: str, step: str, progress: int) -> None:
        await self.client.hset(
            _progress_key(case_id),
            mapping={
                "step": step,
                "progress": str(progress),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        logger.debug(f"[Progress] {case_id}: {step} ({progress}%)")

    # ============================================================
    # 중단 확인
    # ============================================================

    async def check_cancelled(self, case_id: str) -> None:
        """
        현재 Redis 상태를 조회하여 cancelled면 예외 발생
        각 파이프라인 단계 시작 전에 호출한다

        - Redis 조회 실패 시: 로그만 남기고 진행 (다음 체크포인트에서 재확인)
        - 세션 없음 (None): 로그만 남기고 진행
        - cancelled 감지: SearchCancelledException 발생
        - 그 외 상태: 정상 진행
        """
        try:
            status = await self.client.hget(_progress_key(case_id), "status")
        except Exception as e:
            logger.warning(
                f"[Progress] check_cancelled Redis 조회 실패 (스킵): "
                f"case_id={case_id}, error={e}"
            )
            return

        if status is None:
            logger.warning(f"[Progress] 세션 없음 (스킵): case_id={case_id}")
            return

        if status == "cancelled":
            logger.info(f"[Progress] 취소 감지: case_id={case_id}")
            raise SearchCancelledException(
                f"Search {case_id} was cancelled by user"
            )

    # ============================================================
    # 예외 상황
    # ============================================================

    async def mark_invalid_input(self, case_id: str, reason: str) -> None:
        """사용자 입력이 부적절한 경우"""
        await self.client.hset(
            _progress_key(case_id),
            mapping={
                "status": "invalid_input",
                "step": "사용자 입력 부적절 (의도 해석 실패)",
                "progress": "5",
                "updated_at": datetime.now(UTC).isoformat(),
                "reason_invalid": reason,
            },
        )
        logger.info(f"[Progress] 의도 해석 실패: {case_id}")

    async def mark_no_results(self, case_id: str) -> None:
        """정상 완료했으나 결과 0건"""
        await self.client.hset(
            _progress_key(case_id),
            mapping={
                "status": "no_results",
                "step": "정상 완료 (결과 0건)",
                "progress": "100",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        logger.info(f"[Progress] 정상 완료 (결과 0건): {case_id}")

    # ============================================================
    # 실패 처리
    # ============================================================

    async def mark_failed(self, case_id: str, error: str) -> None:
        """시스템 오류"""
        await self.client.hset(
            _progress_key(case_id),
            mapping={
                "status": "failed",
                "step": "실패",
                "updated_at": datetime.now(UTC).isoformat(),
                "error": error,
            },
        )
        logger.warning(f"[Progress] 검색 실패: {case_id} - {error}")

    # ============================================================
    # 리소스 정리
    # ============================================================

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ============================================================
# 싱글톤 인스턴스
# ============================================================
progress_tracker = ProgressTracker()