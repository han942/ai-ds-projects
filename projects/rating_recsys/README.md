# Rating Recommender System v2

이 프로젝트는 기존의 리뷰 기반 평점 예측 실험을 넘어, 실제 데이터베이스를
사용하는 2-stage 식당 추천 시스템으로 확장한다.

- 전체 계획: [RECOMMENDER_V2_PLAN.md](./RECOMMENDER_V2_PLAN.md)
- 이전 버전: [legacy/v1_rating_prediction/](./legacy/v1_rating_prediction/)

현재 단계에서는 v2의 Supabase PostgreSQL ingestion 기반을 구현하고 있다.
기존의 노트북, 수집 데이터, 모델 checkpoint 및 예측 결과는 삭제하지 않고
legacy 폴더에 그대로 보존하였다.

## Supabase PostgreSQL ingestion

현재 5개 legacy CSV를 Supabase PostgreSQL의 private `recsys` schema로 적재하는
첫 번째 파이프라인이 구현되어 있다. 원본 사용자명은 저장하지 않고 고정된
salt를 사용한 SHA-256 pseudonym으로 변환한다. 파일 hash와 review content
hash를 사용하므로 같은 파일을 다시 실행해도 리뷰가 중복 삽입되지 않는다.

### 1. 환경 준비

```bash
cd projects/rating_recsys
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Supabase Dashboard의 `Connect`에서 연결 문자열을 복사하여 `.env`의
`DATABASE_URL`에 설정한다. Migration에는 direct connection이 가장 적합하지만
로컬 네트워크가 IPv4 only라면 port 5432의 Session pooler를 사용할 수 있다.
비밀번호의 예약 문자는 URL encoding하고 `sslmode=require`를 유지한다.

`USER_HASH_SALT`는 한 번 생성한 긴 random 문자열로 설정하고 이후 변경하지
않는다. Salt가 바뀌면 같은 source 사용자가 다른 사용자로 생성된다.

### 2. DB 연결 전 검증

```bash
rating-recsys-ingest --dry-run
```

Dry-run은 Supabase에 연결하거나 데이터를 쓰지 않고 5개 파일의 schema,
필수값, 날짜 parsing, 중복 review hash와 entity 수를 검사한다.

### 3. Migration과 적재

```bash
rating-recsys-migrate
rating-recsys-ingest --skip-migrations
```

또는 migration을 포함해 한 번에 실행한다.

```bash
rating-recsys-ingest
```

적재 결과는 파일별 전체·신규·중복·거부 행 수를 JSON으로 출력한다. 같은
명령을 다시 실행하면 이미 성공한 동일 file hash는 `skipped` 상태가 된다.

검증 query는 [queries/verify_ingestion.sql](./queries/verify_ingestion.sql)에
있다.

### 주요 파일

| 경로 | 역할 |
|---|---|
| `migrations/001_initial_ingestion.sql` | `crawl_runs`, `restaurants`, `app_users`, `reviews` schema |
| `src/rating_recsys/ingestion/transform.py` | CSV 정규화, 날짜 parsing, pseudonym 및 dedup hash |
| `src/rating_recsys/ingestion/loader.py` | PostgreSQL COPY와 set-based upsert |
| `src/rating_recsys/ingestion/cli.py` | dry-run 및 전체 ingestion command |
| `tests/test_transform.py` | 변환 규칙과 5개 CSV smoke test |
