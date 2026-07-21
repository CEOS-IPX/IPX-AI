"""
============================================================
KIPRIS BULK 초기 적재 파이프라인 (로컬용)
Colab은 내 로컬 컴퓨터(127.0.0.1) 보지 못하는 관계로 Python 스크립트로 실행 (FastAPI 서버 필요 X)
============================================================
실행 순서
  1. KIPRIS BULK 압축 해제 (TXT 폴더에 6개 파일)
  2. 6개 .txt 파일 파싱 및 출원번호 기준 병합
  3. 텍스트 전처리 (XML 태그/상투어/노이즈 제거)
  4. BGE-M3 임베딩 생성 (Colab GPU)
  5. OpenSearch + pgvector 적재
  6. sync_history 기록
============================================================
"""

# ===== 0. 라이브러리 설치 (Colab 최초 1회) =====
# !pip install sentence-transformers opensearch-py psycopg2-binary pandas

import re
import json
import unicodedata
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import os

from app.preprocessing import (
    clean_text,
    strip_xml_tags,
    extract_independent_from_bulk_xml,
)

# ===== 설정 =====
BULK_DIR = "./data/bulk/TXT"        # 압축 해제된 TXT 폴더 경로
DELIMITER = "¶"                             # KIPRIS BULK 반환 .txt 파일 구분자
ENCODING = "utf-8"                          # 인코딩

# PostgreSQL / pgvector 설정 (도커 환경변수 우선 적용)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")   # 로컬 실행 시, localhost로 변경
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "patent_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "ipx_patent_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "ipx_patent_password")

# OpenSearch 설정 (도커 환경변수 우선 적용)
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "opensearch")    # 로컬 실행 시, localhost로 변경
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "patents")

BATCH_SIZE = 100                            # 임베딩 배치 크기


# ============================================================
# 1. KIPRIS BULK 파일 파싱
# ============================================================

def read_bulk_file(filename: str) -> pd.DataFrame:
    """KIPRIS BULK txt 파일을 pandas DataFrame으로 읽기"""
    filepath = Path(BULK_DIR) / filename
    df = pd.read_csv(
        filepath,
        sep=DELIMITER,
        encoding=ENCODING,
        dtype=str,                          # 모든 컬럼 문자열로 읽기 (출원번호 앞자리 0 보존)
        keep_default_na=False,              # 빈 문자열을 NaN으로 변환하지 않음
        engine="python"                     # 1자 외 구분자는 python 엔진 필요
    )
    df.columns = df.columns.str.strip()
    return df


def parse_bibliographic() -> pd.DataFrame:
    """서지정보 파싱"""
    df = read_bulk_file("Bibliographic.txt")

    # 필요한 컬럼만 추출 및 이름 정리
    result = pd.DataFrame({
        "application_number": df["출원번호"],
        "registration_number": df["등록번호"],
        "open_number": df["공개번호"],
        "application_date": df["출원일자"].apply(parse_date),
        "open_date": df["공개일자"].apply(parse_date),
        "registration_date": df["등록일자"].apply(parse_date),
        "title": df["발명의명칭(국문)"].str.strip(),
        "title_en": df["발명의명칭(영문)"].str.strip(),
        "legal_status": df["등록사항"].str.strip(),         # 공개/등록/소멸/취하/거절
    })
    return result


def parse_abstract() -> pd.DataFrame:
    """초록 파싱 (XML 태그 제거)"""
    df = read_bulk_file("Abstract.txt")
    df["abstract_raw"] = df["초록"].apply(strip_xml_tags)
    return df[["출원번호", "abstract_raw"]].rename(columns={"출원번호": "application_number"})


def parse_claim() -> pd.DataFrame:
    """청구항 파싱 (XML 태그 제거 + 독립 청구항 추출)"""
    df = read_bulk_file("Claim.txt")
    df["claims_independent"] = df["청구항"].apply(extract_independent_from_bulk_xml)
    return df[["출원번호", "claims_independent"]].rename(columns={"출원번호": "application_number"})


def parse_ipc() -> pd.DataFrame:
    """IPC 코드 파싱 (출원번호당 여러 코드 → 리스트)"""
    df = read_bulk_file("IPC.txt")
    df["ipc코드"] = df["ipc코드"].str.strip()
    ipc_grouped = df.groupby("출원번호")["ipc코드"].apply(list).reset_index()
    return ipc_grouped.rename(columns={"출원번호": "application_number", "ipc코드": "ipc_codes"})


def parse_cpc() -> pd.DataFrame:
    """CPC 코드 파싱 (출원번호당 여러 코드 → 리스트)"""
    df = read_bulk_file("CPC.txt")
    df["cpc코드"] = df["cpc코드"].str.strip()
    cpc_grouped = df.groupby("출원번호")["cpc코드"].apply(list).reset_index()
    return cpc_grouped.rename(columns={"출원번호": "application_number", "cpc코드": "cpc_codes"})


def parse_related_person() -> pd.DataFrame:
    """관련인 파싱 (출원인, 발명자만 추출)"""
    df = read_bulk_file("RelatedPerson.txt")
    df["성명"] = df["성명"].str.strip()
    df["관련인구분"] = df["관련인구분"].str.strip()

    # 출원인만 필터링
    applicants = df[df["관련인구분"].str.startswith("출원인")]
    applicants_grouped = applicants.groupby("출원번호")["성명"].apply(
        lambda x: ", ".join(x.dropna())
    ).reset_index()
    applicants_grouped.columns = ["application_number", "applicant_name"]

    # 발명자만 필터링
    inventors = df[df["관련인구분"].str.startswith("발명자")]
    inventors_grouped = inventors.groupby("출원번호")["성명"].apply(
        lambda x: ", ".join(x.dropna())
    ).reset_index()
    inventors_grouped.columns = ["application_number", "inventor_name"]

    return applicants_grouped.merge(inventors_grouped, on="application_number", how="outer")


# ============================================================
# 2. 텍스트 전처리
# ============================================================


def parse_date(date_str: str) -> str | None:
    """날짜 문자열 정규화 → 'YYYY-MM-DD'"""
    if not date_str or pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    if not date_str:
        return None
    # 'YYYYMMDD' → 'YYYY-MM-DD'
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    # 이미 'YYYY-MM-DD' 형식이면 그대로
    try:
        datetime.strptime(date_str[:10], "%Y-%m-%d")
        return date_str[:10]
    except ValueError:
        return None


# ============================================================
# 3. 전체 병합
# ============================================================

def merge_all_files() -> pd.DataFrame:
    """6개 파일을 출원번호 기준으로 병합"""
    print("[병합] Bibliographic.txt 읽기")
    biblio = parse_bibliographic()
    print(f"  - {len(biblio)}건")

    print("[병합] Abstract.txt 읽기")
    abstract = parse_abstract()
    print(f"  - {len(abstract)}건")

    print("[병합] Claim.txt 읽기")
    claim = parse_claim()
    print(f"  - {len(claim)}건")

    print("[병합] IPC.txt 읽기")
    ipc = parse_ipc()
    print(f"  - {len(ipc)}건")

    print("[병합] CPC.txt 읽기")
    cpc = parse_cpc()
    print(f"  - {len(cpc)}건")

    print("[병합] RelatedPerson.txt 읽기")
    related = parse_related_person()
    print(f"  - {len(related)}건")

    # 서지정보를 기준으로 left join (서지에 없는 출원번호는 무시)
    merged = (
        biblio
        .merge(abstract, on="application_number", how="left")
        .merge(claim, on="application_number", how="left")
        .merge(ipc, on="application_number", how="left")
        .merge(cpc, on="application_number", how="left")
        .merge(related, on="application_number", how="left")
    )

    # 리스트 컬럼 NaN → 빈 리스트
    merged["ipc_codes"] = merged["ipc_codes"].apply(lambda x: x if isinstance(x, list) else [])
    merged["cpc_codes"] = merged["cpc_codes"].apply(lambda x: x if isinstance(x, list) else [])

    print(f"[병합 완료] 총 {len(merged)}건")
    return merged


# ============================================================
# 4. 전처리 적용
# ============================================================

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """전체 DataFrame에 전처리 적용"""
    print("[전처리] 초록/청구항 정제")

    df["abstract_clean"] = df["abstract_raw"].apply(clean_text)
    # df["claims_independent"] = df["claims_independent"].apply(clean_text)
    df["claims_independent"] = df["claims_independent"].apply(
        lambda claims: [clean_text(claim) for claim in claims] if isinstance(claims, list) else []
    )

    # 임베딩 원본: "제목. 전처리된 초록"
    df["embedding_text"] = df.apply(
        lambda row: f"{row['title']}. {row['abstract_clean']}",
        axis=1
    )

    # 초록이 비어있는 행 제외
    before = len(df)
    df = df[df["abstract_clean"].str.len() > 0].reset_index(drop=True)
    print(f"[전처리 완료] 유효 특허: {len(df)}건 (제외: {before - len(df)}건)")

    return df


# ============================================================
# 5. BGE-M3 임베딩
# ============================================================

def load_embedding_model():
    """BGE-M3 모델 로드"""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[임베딩] 디바이스: {device}")

    model = SentenceTransformer("BAAI/bge-m3", device=device)
    print("[임베딩] BGE-M3 로드 완료")
    return model


def generate_embeddings(model, df: pd.DataFrame) -> pd.DataFrame:
    """배치 단위로 임베딩 생성"""
    texts = df["embedding_text"].tolist()
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.extend(embeddings.tolist())

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  임베딩 진행: {min(i + BATCH_SIZE, len(texts))} / {len(texts)}")

    df["embedding"] = all_embeddings
    print(f"[임베딩 완료] {len(df)}건")
    return df


# ============================================================
# 6. OpenSearch 적재
# ============================================================

def create_opensearch_client():
    from opensearchpy import OpenSearch
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,     # 추가: 60초로 늘림
        max_retries=3,  # 추가: 자동 재시도
    )
    print(f"[OpenSearch] 연결: {client.info()['version']['number']}")
    return client


def bulk_index_opensearch(client, df: pd.DataFrame):
    """OpenSearch Bulk 적재"""
    from opensearchpy.helpers import bulk

    def clean_value(v):
        """NaN, None, 빈 공백을 None으로 정규화"""
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def generate_actions():
        for _, row in df.iterrows():
            doc = {
                "_index": OPENSEARCH_INDEX,
                "_id": str(row["application_number"]),
                "application_number": str(row["application_number"]),
                "registration_number": clean_value(row.get("registration_number")),
                "open_number": clean_value(row.get("open_number")),
                "application_date": clean_value(row.get("application_date")),
                "open_date": clean_value(row.get("open_date")),
                "registration_date": clean_value(row.get("registration_date")),
                "title": row["title"],
                "abstract_clean": row["abstract_clean"],
                "claims_independent": clean_value(row.get("claims_independent")),
                "applicant_name": clean_value(row.get("applicant_name")) or "",
                "inventor_name": clean_value(row.get("inventor_name")) or "",
                "ipc_codes": row.get("ipc_codes", []),
                "cpc_codes": row.get("cpc_codes", []),
                "legal_status": row.get("legal_status", ""),
                "collected_date": date.today().isoformat(),
            }
            # None 값 제거 (OpenSearch에 NaN 안 들어가게)
            doc = {k: v for k, v in doc.items() if v is not None or k == "_index" or k == "_id"}
            yield doc

    success, errors = bulk(client, generate_actions(), chunk_size=500)
    print(f"[OpenSearch] 적재 완료: 성공 {success}건, 실패 {len(errors)}건")
    if errors:
        print(f"  에러 샘플: {errors[:3]}")


# ============================================================
# 7. pgvector 적재
# ============================================================

def create_pgvector_connection():
    import psycopg2
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    print("[pgvector] PostgreSQL 연결 완료")
    return conn


def bulk_insert_pgvector(conn, df: pd.DataFrame):
    """pgvector Bulk 적재"""
    import psycopg2.extras

    cursor = conn.cursor()
    values = []

    for _, row in df.iterrows():
        values.append((
            row["application_number"],
            row["embedding"],
            row.get("ipc_codes", []),
        ))

    psycopg2.extras.execute_batch(
        cursor,
        """
        INSERT INTO patent_vectors
            (application_number, embedding, ipc_codes)
        VALUES
            (%s, %s::vector, %s)
        ON CONFLICT (application_number) DO NOTHING
        """,
        values,
        page_size=500,
    )
    conn.commit()
    cursor.close()
    print(f"[pgvector] 적재 완료: {len(values)}건")


def create_hnsw_index(conn):
    """HNSW 벡터 인덱스 생성 (적재 완료 후)"""
    cursor = conn.cursor()
    print("[pgvector] HNSW 인덱스 생성 중...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pv_hnsw ON patent_vectors
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200);
    """)
    conn.commit()
    cursor.close()
    print("[pgvector] HNSW 인덱스 생성 완료")


# ============================================================
# 8. sync_history 기록
# ============================================================

def insert_sync_history(conn, count: int):
    """수집 이력 저장"""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sync_history (job_type, query_date_to, records_added, status)
        VALUES (%s, %s, %s, %s)
        """,
        ("initial", date.today(), count, "success")
    )
    conn.commit()
    cursor.close()
    print(f"[sync_history] 초기 적재 이력 기록: {count}건")


# ============================================================
# 9. 중간 저장 (세션 끊김 대비)
# ============================================================

def save_checkpoint(df: pd.DataFrame, filename: str = "checkpoint.json"):
    """임베딩 결과를 파일로 저장"""
    df.to_json(filename, orient="records", force_ascii=False)
    print(f"[체크포인트] {len(df)}건 저장 → {filename}")


def load_checkpoint(filename: str = "checkpoint.json") -> pd.DataFrame:
    """저장된 체크포인트 복원"""
    df = pd.read_json(filename, orient="records")
    print(f"[체크포인트] {len(df)}건 복원 ← {filename}")
    return df


# ============================================================
# 10. 메인 실행
# ============================================================

def run_initial_ingestion():
    """초기 적재 전체 파이프라인 실행"""

    print("\n" + "=" * 60)
    print("STEP 1: BULK 파일 파싱 및 병합")
    print("=" * 60)
    df = merge_all_files()

    print("\n" + "=" * 60)
    print("STEP 2: 텍스트 전처리")
    print("=" * 60)
    df = preprocess_dataframe(df)

    print("\n" + "=" * 60)
    print("STEP 3: BGE-M3 임베딩 생성")
    print("=" * 60)
    model = load_embedding_model()
    df = generate_embeddings(model, df)

    # 중간 저장
    save_checkpoint(df, "checkpoint_embedded.json")

    print("\n" + "=" * 60)
    print("STEP 4: OpenSearch 적재")
    print("=" * 60)
    os_client = create_opensearch_client()
    bulk_index_opensearch(os_client, df)

    print("\n" + "=" * 60)
    print("STEP 5: pgvector 적재")
    print("=" * 60)
    pg_conn = create_pgvector_connection()
    bulk_insert_pgvector(pg_conn, df)
    create_hnsw_index(pg_conn)
    insert_sync_history(pg_conn, len(df))
    pg_conn.close()

    print("\n" + "=" * 60)
    print(f"[완료] 총 {len(df)}건 적재 완료")
    print("=" * 60)


# ===== 실행 방식 택 =====
# 1. Colab에서 아래 한 줄 실행
# run_initial_ingestion()

# 2. Python 스크립트에서 아래 실행
if __name__ == "__main__":
    run_initial_ingestion()