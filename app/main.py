"""
============================================================
FastAPI 진입점
============================================================
"""

import logging
from fastapi import FastAPI
from app.routers import search

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="IPX-AI Search Server")

# 라우터 등록
app.include_router(search.router)


@app.get("/health")
def health():
    return {"status": "UP"}