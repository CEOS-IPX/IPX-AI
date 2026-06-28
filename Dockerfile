# ===== Build stage =====
FROM python:3.11-slim AS builder

WORKDIR /app

# 빌드 시 필요한 시스템 패키지 (gcc 등, ML 패키지 컴파일에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ===== Runtime stage =====
FROM python:3.11-slim

# 런타임 시스템 패키지 (libgomp1: torch/numpy OpenMP 런타임)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 시간대 한국
ENV TZ=Asia/Seoul

# non-root 사용자 생성
RUN groupadd -r appgroup && useradd -r -g appgroup -s /bin/false appuser

WORKDIR /app

# 빌드 스테이지에서 설치한 Python 패키지 복사
COPY --from=builder /install /usr/local

# Python 환경 변수
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# 캐시 디렉토리 생성 및 권한
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appgroup /app/.cache

# 소스 코드 복사
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8000

# 헬스체크 (Spring 측과 동일한 패턴)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uvicorn 단일 worker (BGE-M3 모델 메모리 부담 고려)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers"]