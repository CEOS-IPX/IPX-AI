"""
============================================================
검색 진행률 추적 서비스 (Redis 기반)
============================================================
Redis 키 구조:
  search:{search_id}                → 진행 상태 (Hash)
    - status: pending/in_progress/completed/cancelled/failed
    - step: 현재 단계 설명
    - progress: 진행률 (0~100)
    - started_at, updated_at
    - error (failed 시)
  search:{search_id}:cancelled      → 중단 신호
  search:{search_id}:result         → 완료된 검색 결과 JSON
  search:{search_id}:context        → 원본 검색 컨텍스트 JSON
                                       (탐색 후 수동 추가 지원용)

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


def _progress_key(search_id: str) -> str:
    return f"search:{search_id}"

def _cancel_key(search_id: str) -> str:
    return f"search:{search_id}:cancelled"

def _result_key(search_id: str) -> str:
    return f"search:{search_id}:result"

def _context_key(search_id: str) -> str:
    return f"search:{search_id}:context"


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

    async def start(self, search_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self.client.hset(
            _progress_key(search_id),
            mapping={
                "status": "in_progress",
                "step": "검색 준비 중",
                "progress": "0",
                "started_at": now,
                "updated_at": now,
            },
        )
        await self.client.expire(_progress_key(search_id), TTL_SECONDS)
        logger.info(f"[Progress] 검색 시작: {search_id}")

    # ============================================================
    # 진행률 업데이트
    # ============================================================

    async def update(self, search_id: str, step: str, progress: int) -> None:
        await self.client.hset(
            _progress_key(search_id),
            mapping={
                "step": step,
                "progress": str(progress),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        logger.debug(f"[Progress] {search_id}: {step} ({progress}%)")

    # ============================================================
    # 중단 확인
    # ============================================================

    async def check_cancelled(self, search_id: str) -> None:
        if await self.client.exists(_cancel_key(search_id)):
            await self.client.hset(
                _progress_key(search_id),
                mapping={
                    "status": "cancelled",
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            logger.info(f"[Progress] 검색 중단: {search_id}")
            raise SearchCancelledException(f"Search {search_id} cancelled by user")

    # ============================================================
    # 중단 요청
    # ============================================================

    async def cancel(self, search_id: str) -> bool:
        status = await self.client.hget(_progress_key(search_id), "status")
        if status not in ("in_progress", "pending"):
            logger.warning(f"[Progress] 중단 불가 (상태={status}): {search_id}")
            return False

        await self.client.set(_cancel_key(search_id), "1", ex=300)
        logger.info(f"[Progress] 중단 요청: {search_id}")
        return True

    # ============================================================
    # 완료 처리
    # ============================================================

    async def mark_completed(self, search_id: str, result: dict) -> None:
        await self.client.hset(
            _progress_key(search_id),
            mapping={
                "status": "completed",
                "step": "완료",
                "progress": "100",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        await self.client.set(
            _result_key(search_id),
            json.dumps(result, ensure_ascii=False),
            ex=TTL_SECONDS,
        )
        logger.info(f"[Progress] 검색 완료: {search_id}")

    # ============================================================
    # 실패 처리
    # ============================================================

    async def mark_failed(self, search_id: str, error: str) -> None:
        await self.client.hset(
            _progress_key(search_id),
            mapping={
                "status": "failed",
                "step": "실패",
                "error": error,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        logger.warning(f"[Progress] 검색 실패: {search_id} - {error}")

    # ============================================================
    # 검색 컨텍스트 저장/조회 (탐색 후 수동 추가용)
    # ============================================================

    async def save_context(self, search_id: str, context: dict) -> None:
        """
        원본 검색 컨텍스트 저장.
        탐색 후 수동 추가 시 사용자 발명 정보를 재사용하기 위함.

        context 예시:
          {
            "title": "...",
            "description": "...",
            "technical_field": "...",
            "user_keywords": [...]   # LLM 추출 원본
          }
        """
        await self.client.set(
            _context_key(search_id),
            json.dumps(context, ensure_ascii=False),
            ex=TTL_SECONDS,
        )

    async def get_context(self, search_id: str) -> Optional[dict]:
        """저장된 검색 컨텍스트 조회"""
        data = await self.client.get(_context_key(search_id))
        if not data:
            return None
        return json.loads(data)

    # ============================================================
    # 상태/결과 조회
    # ============================================================

    async def get_status(self, search_id: str) -> Optional[dict]:
        data = await self.client.hgetall(_progress_key(search_id))
        if not data:
            return None
        return {
            "search_id": search_id,
            "status": data.get("status"),
            "step": data.get("step"),
            "progress": int(data.get("progress", "0")),
            "started_at": data.get("started_at"),
            "updated_at": data.get("updated_at"),
            "error": data.get("error"),
        }

    async def get_result(self, search_id: str) -> Optional[dict]:
        result_json = await self.client.get(_result_key(search_id))
        if not result_json:
            return None
        return json.loads(result_json)

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