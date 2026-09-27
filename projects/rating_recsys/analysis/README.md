# analysis/

실험 코드가 아닌 기록을 모아 둔다. 현재 모델의 결과는 여기 있지 않고 각 run의
보고서(`artifacts/runs/<run_id>/report.md`)에 있다.

```text
analysis/
├── README.md
├── data/                        데이터 수집·적재·규모 점검
│   ├── 2026-09-26_crawler_import.md
│   └── 2026-09-27_data_snapshot.md
└── archive/
    └── primary/                 보관한 이전 평가 방식(leave-last-two-out)
        ├── README.md            프로토콜, 시도별 결과 요약, 재현 방법
        └── 2026-09-2x_*.md      시도별 결과 문서
```

## 어디에 무엇을 쓰나

| 내용 | 위치 |
|---|---|
| 모델 실험 결과 (현재 평가 방식) | `artifacts/runs/<run_id>/report.md`. 자동 생성되고, 6절 해석만 직접 채운다 |
| 데이터 수집·품질·규모 점검 | `analysis/data/<YYYY-MM-DD>_<주제>.md` |
| 더 이상 쓰지 않는 평가 방식의 기록 | `analysis/archive/<방식>/` |

## 문서 작성 기준

1. 파일 이름은 `<YYYY-MM-DD>_<주제>.md`로 한다.
2. 첫 부분에 사용한 snapshot ID, run ID(해당 시), 조회 시점을 적는다.
3. 수치 옆에 모수(사용자 수, query 수)를 적고, 비교에는 가능하면 신뢰구간을 붙인다.
4. 다른 snapshot의 수치와 비교할 때는 데이터가 다르다는 점을 명시한다.
5. 끝에 재실행 명령과 원본 artifact 경로를 적는다.
6. 일회성 노트북이나 스크립트는 결론을 문서로 옮긴 뒤 지운다.
