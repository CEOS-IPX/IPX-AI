"""
============================================================
FastAPI 진입점
============================================================
서버 시작 시 BGE-M3 모델을 로드하고, APScheduler를 시작한다.
============================================================
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import search, components, novelty, inventive_step
from app.services.embedding import embedding_service
from app.services.opensearch_client import opensearch_service
from app.services.pgvector_client import pgvector_service
from app.services.kipris_client import kipris_service
from app.services.progress_tracker import progress_tracker
from app.scheduler import start_scheduler, shutdown_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 작업"""
    # === Startup ===
    logger.info("[Startup] BGE-M3 모델 로드 시작")
    embedding_service.load()

    logger.info("[Startup] 스케줄러 시작")
    start_scheduler()

    logger.info("[Startup] 초기화 완료, 요청 처리 준비됨")

    yield

    # === Shutdown ===
    logger.info("[Shutdown] 리소스 정리 시작")
    shutdown_scheduler()
    await opensearch_service.close()
    await pgvector_service.close()
    await kipris_service.close()
    await progress_tracker.close()
    logger.info("[Shutdown] 서버 종료")


app = FastAPI(
    title="IPX-AI Search Server",
    lifespan=lifespan,
)

app.include_router(search.router)
app.include_router(components.router)
app.include_router(novelty.router)
app.include_router(inventive_step.router)


@app.get("/health")
def health():
    return {
        "status": "UP",
        "embedding_loaded": embedding_service.is_loaded,
        "embedding_device": embedding_service.device,
    }