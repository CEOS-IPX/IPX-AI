"""
============================================================
BGE-M3 임베딩 서비스
============================================================
BGE-M3 모델로 텍스트를 1024차원 벡터로 변환한다.

서버 시작 시 1회 로드해서 메모리에 유지하여
검색 요청마다 새로 로드하지 않고 같은 인스턴스를 재사용한다.
============================================================
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingService:
    """BGE-M3 임베딩 모델 래퍼 (싱글톤)"""

    def __init__(self):
        self._model = None
        self._device: Optional[str] = None

    def load(self) -> "EmbeddingService":
        """
        모델을 메모리에 로드한다.
        최초 호출 시 ~2GB 다운로드 + 로드 (수 분 소요 가능).
        이후엔 캐시에서 빠르게 로드.
        """
        if self._model is not None:
            logger.info("[임베딩] 이미 로드됨, 재사용")
            return self

        from sentence_transformers import SentenceTransformer
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[임베딩] BGE-M3 로딩 중... (device={self._device})")

        self._model = SentenceTransformer("BAAI/bge-m3", device=self._device)

        logger.info("[임베딩] BGE-M3 로드 완료")
        return self

    def embed(self, text: str) -> list[float]:
        """
        텍스트 1건을 1024차원 벡터로 변환.

        Args:
            text: 임베딩할 텍스트

        Returns:
            1024차원 float 리스트 (L2 정규화됨 -> 코사인 검색에 바로 사용 가능)

        Raises:
            RuntimeError: 모델이 로드되지 않은 상태에서 호출 시
        """
        if self._model is None:
            raise RuntimeError("EmbeddingService.load()를 먼저 호출해야 합니다.")

        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다.")

        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vec.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        여러 텍스트를 배치로 임베딩 (증분 적재 등에 사용).

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 한 번에 처리할 배치 크기

        Returns:
            각 텍스트의 1024차원 벡터 리스트
        """
        if self._model is None:
            raise RuntimeError("EmbeddingService.load()를 먼저 호출해야 합니다.")

        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vecs.tolist()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> Optional[str]:
        return self._device


# ============================================================
# 싱글톤 인스턴스
# ============================================================
# main.py의 lifespan에서 load() 호출
embedding_service = EmbeddingService()