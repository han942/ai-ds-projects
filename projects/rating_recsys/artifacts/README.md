# artifacts/

실험 입력과 출력이 쌓이는 로컬 폴더다. Git에는 이 파일과 각 run의 `report.md`만
올라가고, 나머지는 이 컴퓨터에만 있다(`.gitignore` 참고).

## 구조

```text
artifacts/
├── README.md                 이 파일
├── snapshots/                고정된 모델링 입력
│   └── <snapshot_id 16자>.jsonl
├── runs/                     실험 1회 = 폴더 1개
│   └── <run_id>/
│       ├── report.md         결과 보고서 (git에 올라감)
│       ├── manifest.json     조건, 구간, 누수 경계, 코드·환경, 소요 시간
│       ├── metrics.json      validation/test 단계별 지표, 선택 과정, 학습 데이터 요약
│       ├── queries_{validation,test}.jsonl          사용자별 이력과 window 정답
│       ├── recommendations_{validation,test}.jsonl  사용자별 Top-K
│       ├── model.txt         최종 LambdaRank (LightGBM 텍스트 형식)
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
- 보고서가 가리키는 snapshot은 지우지 않는다. 재현에 필요하다.
- `ingestion_sources/`는 DB 적재 출처의 증빙이므로 지우지 않는다.
