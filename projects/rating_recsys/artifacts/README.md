# 실험 파일 보관

Git에 올릴 결과 문서는 [카테고리별 overview](../reports/README.md)다. 현재 reference baseline의 실행 리포트 하나는 모델 정의의 근거로 유지한다. 실행별 비교 리포트·과거 리포트·생성 파일은 로컬 보관본으로 옮겼다.

## 현재 보존 범위 (2026-10-07 정리 후)

사용자 요청으로 임베딩 벡터 캐시와 설치 모델은 E5만 남겼다. `e5_review_embedding_cache.sqlite`의 전체 59,455개 입력, `local_models/`의 E5 모델, E5 토크나이저를 보존했다. 원본 `snapshots/`, 현재 baseline/E5 LTR run·검증 JSON·prepared 후보, MLflow와 Git 복구 백업도 유지한다.

Liquid SQLite 캐시·로컬 모델·토크나이저와 `embeddings.zip`, 임시 `e5_smoke.sqlite`는 삭제했다. V1 전용 환경·FastText 캐시·notebook·모델·예측 결과와 `legacy/v2_experiments.zip`도 삭제했다. V1 CSV 5개는 현재 ingestion의 기본 입력이므로 보존한다. 현재 프로젝트 `.venv`와 `.env`는 삭제하지 않았다. 별도 삭제본 ZIP은 만들지 않는다.

아래의 과거 보관 inventory는 당시 상태의 기록이다. 삭제한 파일을 지금도 보존하거나 ZIP에서 복원할 수 있다는 뜻은 아니다. Liquid 실험의 요약 수치는 `reports/review_embeddings.md`에 남지만 벡터 캐시는 로컬에서 복원할 수 없다.

| 보관 위치 | 내용 |
|---|---|
| `archive/20261006_report_cleanup/bm25.zip` | BM25·전처리·Kiwi 벤치마크의 후보·색인·수치·조건·검증·소스 |
| 같은 폴더의 `review_aspects.zip` | 추출 입력·응답·벡터 CSV·검증·비용 |
| 같은 폴더의 `baseline_features.zip` | 지역·피처·C1/LightGCN 진단 모델과 수치 |
| 같은 폴더의 `shrinkage.zip` | 이전 평점 보정 비교 모델·추천·수치 |
| `archive/20261006_category_overviews/detailed_reports.zip` | 실행별 상세 리포트·사용자별 원문 근거·정리 전 문서 |

생성 파일 212개와 문서 보관본은 실제로 압축 해제해 크기·SHA-256 일치를 확인한 뒤 원래 파일을 정리했다. 각 보관 폴더의 `inventory.json`에 원본 경로·해시·검증 결과가 있다. 첫 정리의 inventory는 당시 보관 상태의 기록이며, 이후 리포트 이동은 두 번째 inventory에 기록했다. API 키와 `.env`는 포함하지 않았다.

## 바로 읽는 자료와 재사용 자료

`reports/`는 최신 카테고리별 overview다. 미실행 Jev는 목록에만 표시하고 실험 후 문서를 만든다. 새 실행별 리포트와 learning curve는 기본적으로 Git에서 제외한다.

`runs/`의 baseline 모델·metrics·query·추천, `snapshots/`, `prepared/`, E5 SQLite 캐시·로컬 모델·토크나이저, MLflow는 이후 실행에서 재사용하므로 유지했다. 이번 정리에서는 모델 학습이나 API 호출을 수행하지 않았다.

## 상세 리포트 확인

명령은 `projects/rating_recsys`에서 실행한다. 과거 리포트는 임시 폴더에 풀어서 읽는다. ZIP에는 정리 전 README·PLAN도 포함되어 있다.

```bash
python -m zipfile -l artifacts/archive/20261006_category_overviews/detailed_reports.zip
python -m zipfile -e artifacts/archive/20261006_category_overviews/detailed_reports.zip /tmp/rating_recsys_report_history
```

이전 BM25 리포트는 임시 폴더의 `artifacts/comparisons/bm25/<run_id>/report.md`, 사용자별 원문 근거는 `artifacts/comparisons/review_aspects/20261006_top15_nemotron3_v3/user_profiles.md`에서 확인한다.

## 생성 파일 복원

과거 실행 폴더를 참조하는 재계산·검증은 해당 생성 파일을 먼저 복원한다. 예를 들어 리뷰 속성 결과를 API 없이 재계산하려면:

```bash
python -m zipfile -e artifacts/archive/20261006_report_cleanup/review_aspects.zip .
.venv/bin/python -m rating_recsys.experiments.review_aspect_audit \
  --output artifacts/comparisons/review_aspects/20261006_top15_nemotron3_v3 \
  --adaptive-batch-size 20 --concurrency 2 --analyze-only
.venv/bin/python -m rating_recsys.experiments.review_aspect_validation \
  artifacts/comparisons/review_aspects/20261006_top15_nemotron3_v3
```

BM25 전처리의 과거 후보 참조에는 `bm25.zip`을 같은 방식으로 복원한다. `embeddings.zip`은 사용자 요청으로 삭제했으므로 이 방법으로 Liquid 벡터·후보를 복원할 수 없다. 필요한 상세 원문 리포트는 위 임시 폴더에서 읽는다. MLflow의 과거 artifact 경로는 해당 보관 파일이 남아 있는 경우에만 복원할 수 있다.

Git에는 overview·안내·현재 baseline 리포트를 남긴다. ZIP·원본 데이터·캐시·API 응답·개인별 원문 근거는 로컬 전용이므로 다른 환경에서 상세 자료가 필요하면 보관본도 함께 옮긴다. 미채택 실험은 overview에 조건과 결론만 남기고, 복원 가치가 있는 원본은 압축 보관한다.
