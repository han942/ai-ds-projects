# artifacts/

실험 입력과 출력이 쌓이는 로컬 폴더다. Git에는 이 파일과 각 run의 `report.md`(후보 비교
실험은 `learning_curve.png`도)만 올라가고, 나머지는 이 컴퓨터에만 있다(`.gitignore` 참고).

## 구조

```text
artifacts/
├── README.md                 이 파일
├── snapshots/                고정된 모델링 입력
│   ├── <snapshot_id 16자>.jsonl          interaction (리뷰 본문 없음)
│   └── <snapshot_id 16자>.reviews.jsonl  review_id별 리뷰 본문 (DeepCoNN만 읽음)
├── runs/                     실험 1회 = 폴더 1개
│   └── <run_id>/
│       ├── report.md         결과 보고서 (git에 올라감)
│       ├── manifest.json     조건, 구간, 누수 경계, 코드·환경, 소요 시간
│       ├── metrics.json      validation/test 단계별 지표, 선택 과정, 학습 데이터 요약
│       ├── queries_{validation,test}.jsonl          사용자별 이력과 window 정답
│       ├── recommendations_{validation,test}.jsonl  사용자별 Top-K
│       ├── model.txt         최종 LambdaRank (LightGBM 텍스트 형식)
│       └── source.diff       미커밋 상태로 실행했을 때만 생성
├── comparisons/              후보 모델 비교 실험 (Stage 1만, MLflow 기록 없음)
│   └── <model>/<run_id>/     lightgcn(당시 기준선 C3와 비교, C5를 고른 run), deepconn(C5와 비교)
│       ├── report.md         결과 보고서 (git에 올라감)
│       ├── learning_curve.png  grid별 학습 loss·validation Recall (git에 올라감)
│       ├── manifest.json     실행 명령, 조건, grid, 구간, 누수 경계, 코드·환경 (리뷰 본문을 읽는 모델은 본문 파일 hash 포함)
│       ├── metrics.json      epoch별 학습 곡선, validation/test 후보 지표, bootstrap
│       ├── candidates_test.jsonl  사용자별 새 후보·선택 결합·C5 후보
│       └── source.diff       미커밋 상태로 실행했을 때만 생성
├── mlflow.db                 MLflow tracking DB (지표·파라미터만, 파일 복사 없음)
├── ingestion_sources/        DB에 적재한 크롤링 원본의 고정 사본과 적재 결과
└── archive/primary/          보관한 이전 평가 방식의 결과
    ├── runs/                 primary leave-last-two-out run 폴더
    ├── comparisons/          지역 제거·LightGCN 비교 결과
    └── mlruns/               당시 MLflow가 복사한 파일
```

## 이름 규칙

- `snapshot_id`: interaction 행을 정렬해 계산한 SHA-256. 같은 데이터면 항상 같은
  ID와 같은 파일이 나온다. `manifest.json`의 `snapshot.path`가 이 파일을 가리킨다.
- `run_id`: `<UTC 시각>-<snapshot_id 앞 8자>`. 이름만 봐도 어떤 데이터로 돌렸는지
  알 수 있다.

## 생성되는 곳

| 경로 | 만드는 명령 |
|---|---|
| `snapshots/`, `runs/<run_id>/` | `rating-recsys-experiment` |
| `comparisons/<model>/<run_id>/` | `rating-recsys-compare <model>` (`lightgcn`, `deepconn`). 2026-09-30 이전 run은 모델별 `python -m rating_recsys.experiments.<model>_cli`로 만들었다 |
| `snapshots/*.reviews.jsonl` | `rating-recsys-compare deepconn` (본문 파일이 없으면 DB에서 읽기 전용으로 한 번 만든다) |
| `mlflow.db` | `rating-recsys-experiment` (`--no-mlflow`이면 생략) |
| `ingestion_sources/` | `python -m rating_recsys.ingestion.import_crawler` |
| `archive/primary/` | 현재 코드에서는 만들지 않음. [보관 안내](../analysis/archive/primary/README.md) |

## 보는 방법

- 보고서: `runs/<run_id>/report.md`
- Streamlit: `rating-recsys-dashboard` → run 선택 → 보고서 · 지표 · 사용자별 추천 ·
  식당별 노출 탭
- MLflow: `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db` →
  experiment `rating-recsys`. Run의 `report` tag가 로컬 보고서 경로다. 이전 기록은
  `archive · primary leave-last-two-out` experiment에 있다.

## 정리 기준

- 결과를 인용하지 않을 run은 폴더째 지워도 된다. MLflow 기록은 UI에서 run을 삭제한 뒤
  `MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow.db mlflow gc
  --backend-store-uri sqlite:///artifacts/mlflow.db`로 비운다.
- 보고서가 가리키는 snapshot과 `.reviews.jsonl`은 지우지 않는다. 재현에 필요하다.
  리뷰 본문 파일에는 원문이 그대로 있으므로 git·외부로 올리지 않는다.
- `ingestion_sources/`는 DB 적재 출처의 증빙이므로 지우지 않는다.
