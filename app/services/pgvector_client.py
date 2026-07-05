"""
============================================================
pgvector 벡터 검색 서비스
============================================================
BGE-M3 임베딩 벡터(1024차원)와 patent_vectors.embedding의
코사인 유사도를 계산하여 상위 N건을 반환한다.

기능:
  - search(): 벡터 유사도 검색
  - insert(): 단건 삽입 (자동 적재용)
============================================================
"""

import logging
import asyncio
from typing import Optional

from psycopg2 import pool

from app.config import settings
from app.services.types import PatentScore

logger = logging.getLogger(__name__)


class PgvectorService:
    """pgvector 벡터 검색 클라이언트 (연결 풀 사용)"""

    def __init__(self):
        self._pool: Optional[pool.SimpleConnectionPool] = None

    @property
    def connection_pool(self) -> pool.SimpleConnectionPool:
        if self._pool is None:
            self._pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                host=settings.postgres_host,
                port=settings.postgres_port,
                dbname=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
            )
        return self._pool

    # ============================================================
    # 벡터 검색
    # ============================================================

    async def search(
        self,
        query_vector: list[float],
        trusted_ipc: Optional[list[str]] = None,
        estimated_ipc: Optional[list[str]] = None,
        size: int = 30,
    ) -> list[PatentScore]:
        """pgvector에서 코사인 유사도 기반 벡터 검색"""
        try:
            results = await asyncio.to_thread(
                self._search_sync,
                query_vector,
                trusted_ipc or [],
                estimated_ipc or [],
                size,
            )
        except Exception:
            logger.exception("[pgvector] 검색 실패")
            return []

        logger.info(f"[pgvector] 검색 완료: {len(results)}건")
        return results

    def _search_sync(
        self,
        query_vector: list[float],
        trusted_ipc: list[str],
        estimated_ipc: list[str],
        size: int,
    ) -> list[PatentScore]:
        """실제 SQL 실행"""
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        all_ipc_prefixes = self._extract_ipc_prefixes(trusted_ipc + estimated_ipc)

        if all_ipc_prefixes:
            sql = """
                SELECT
                    application_number,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM patent_vectors
                WHERE EXISTS (
                    SELECT 1 FROM unnest(ipc_codes) AS code
                    WHERE code LIKE ANY(%s)
                )
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            like_patterns = [f"{prefix}%" for prefix in all_ipc_prefixes]
            params = (vector_str, like_patterns, vector_str, size)
        else:
            sql = """
                SELECT
                    application_number,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM patent_vectors
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params = (vector_str, vector_str, size)

        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            self.connection_pool.putconn(conn)

        results: list[PatentScore] = []
        for rank, (app_num, similarity) in enumerate(rows, start=1):
            results.append(PatentScore(
                application_number=app_num,
                score=float(similarity),
                source="pgvector",
                rank=rank,
            ))

        return results

    # ============================================================
    # 단건 INSERT (자동 적재용)
    # ============================================================

    async def insert(
        self,
        application_number: str,
        embedding: list[float],
        ipc_codes: list[str],
    ) -> bool:
        """
        patent_vectors 테이블에 단건 삽입 (자동 적재용).

        이미 존재하는 경우 UPDATE (ON CONFLICT).

        Args:
            application_number: 출원번호
            embedding: BGE-M3 임베딩 (1024차원)
            ipc_codes: IPC 코드 리스트

        Returns:
            True: 성공, False: 실패
        """
        try:
            await asyncio.to_thread(
                self._insert_sync,
                application_number,
                embedding,
                ipc_codes,
            )
            logger.info(f"[pgvector] 적재 완료: {application_number}")
            return True
        except Exception:
            logger.exception(f"[pgvector] 적재 실패: {application_number}")
            return False

    def _insert_sync(
        self,
        application_number: str,
        embedding: list[float],
        ipc_codes: list[str],
    ) -> None:
        """실제 INSERT SQL"""
        vector_str = "[" + ",".join(map(str, embedding)) + "]"

        sql = """
            INSERT INTO patent_vectors
                (application_number, embedding, ipc_codes, updated_at)
            VALUES
                (%s, %s::vector, %s, NOW())
            ON CONFLICT (application_number) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                ipc_codes = EXCLUDED.ipc_codes,
                updated_at = NOW();
        """

        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (application_number, vector_str, ipc_codes))
            conn.commit()
        finally:
            self.connection_pool.putconn(conn)

    # ============================================================
    # 유틸리티
    # ============================================================

    @staticmethod
    def _extract_ipc_prefixes(ipc_codes: list[str]) -> set[str]:
        """'H01M 10/052' → 'H01M' prefix 추출"""
        return {
            code.split(" ")[0].strip()
            for code in ipc_codes
            if code and code.strip()
        }

    async def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None


# ============================================================
# 싱글톤 인스턴스
# ============================================================
pgvector_service = PgvectorService()