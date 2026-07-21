"""
Checkpoint 파일을 이용한 특허 데이터 적재 스크립트.

Colab에서 임베딩까지 완료된 checkpoint JSON 파일을 이용하여
OpenSearch와 pgvector에 직접 적재.

사용법:
    python scripts/ingest_from_checkpoint.py /tmp/checkpoint_batch_1.json

배치별로 순차 실행:
    python scripts/ingest_from_checkpoint.py /tmp/checkpoint_batch_1.json
    python scripts/ingest_from_checkpoint.py /tmp/checkpoint_batch_2.json
    python scripts/ingest_from_checkpoint.py /tmp/checkpoint_batch_3.json
"""

import sys
import os
from pathlib import Path
import pandas as pd

# 도커 컨테이너 내 실행 시 상위 디렉터리를 모듈 탐색 경로에 추가 (Import Error 방지)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 기존 initial_ingestion.py의 함수들 재사용
from scripts.initial_ingestion import (
    create_opensearch_client,
    bulk_index_opensearch,
    create_pgvector_connection,
    bulk_insert_pgvector,
    insert_sync_history,
)


def load_checkpoint_safely(checkpoint_path: str) -> pd.DataFrame:
    """JSON 로드 시 출원번호/등록번호/공개번호의 앞자리 '0' 유실 및 타입 변형 방지"""
    # dtype=False를 부여하여 판다스의 자동 타입 추론 억제
    df = pd.read_json(checkpoint_path, orient="records", dtype=False)

    # 번호 관련 컬럼들을 순수 문자열 형태(str)로 강제 정형화
    str_cols = ["application_number", "registration_number", "open_number"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True)
            df[col] = df[col].apply(lambda x: "" if x in ["None", "nan", "NoneType"] else x)

    print(f"[Load] {len(df)}건 로드 및 데이터 타입 검증 완료")
    return df


def main(checkpoint_path: str):
    """Checkpoint 파일로 OpenSearch + pgvector에 적재."""

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint 파일 없음: {checkpoint_path}")
        sys.exit(1)

    print(f"[Start] Checkpoint 적재: {checkpoint_path}")

    # 1. Checkpoint 안전 로드
    df = load_checkpoint_safely(checkpoint_path)

    # 2. OpenSearch 적재
    print("\n[OpenSearch] 적재 시작...")
    os_client = create_opensearch_client()
    bulk_index_opensearch(os_client, df)

    # 3. pgvector 적재
    print("\n[pgvector] 적재 시작...")
    pg_conn = create_pgvector_connection()
    bulk_insert_pgvector(pg_conn, df)

    # 4. sync_history 이력 기록
    insert_sync_history(pg_conn, len(df))
    pg_conn.close()

    print(f"\n[Complete] {len(df)}건 적재 완료!")
    print("[Note] 모든 배치(batch_1~3) 적재가 완료되면 HNSW 인덱스를 생성해 주세요.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/ingest_from_checkpoint.py <checkpoint_path>")
        sys.exit(1)

    main(sys.argv[1])