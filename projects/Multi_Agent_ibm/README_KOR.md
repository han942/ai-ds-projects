# [Multi-Agent] AskIntern: IBM 인턴 업무 지원 어시스턴트

AskIntern은 **IBM watsonx Orchestrate**를 기반으로 만든 멀티 에이전트 업무 지원 어시스턴트입니다. 인턴들이 업무 중 반복적으로 묻는 점심 식당, IBM 제품 및 개발 도구, 회의·세미나 노트 관련 질문을 하나의 웹 채팅 화면에서 처리하는 것을 목표로 합니다.

- [English](./README.md)
- 데모: [`demo_askintern.mp4`](./demo_askintern.mp4)
- 발표 자료: [`AskIntern_presentation.pdf`](./AskIntern_presentation.pdf)

---

## 1. 프로젝트 개요

- 목표: IBM 인턴을 위한 신뢰할 수 있는 멀티 에이전트 업무 지원 시스템 구축
- 플랫폼: IBM watsonx Orchestrate, IBM Code Engine
- 인터페이스: 웹 채팅
- 메인 오케스트레이터: `askintern_supervisor`
- 하위 에이전트: `lunch_agent`, `ibm_specs_agent`, `notes_qa_agent`
- 상태: 이 폴더의 발표 자료와 데모 영상으로 정리된 프로토타입/데모 구현

AskIntern은 사용자의 질문을 적절한 전문 에이전트로 라우팅한 뒤, 결과를 하나의 답변으로 제공합니다. 검색, 외부 도구, 워크플로 실행, 접근 제어, 응답 가드레일을 결합해 엔터프라이즈 환경에서 필요한 안정성과 통제 가능성을 검토했습니다.

---

## 2. 시스템 아키텍처

```text
웹 채팅
   |
   v
askintern_supervisor  (watsonx Orchestrate)
   |------------------------|-------------------------|
   v                        v                         v
lunch_agent            ibm_specs_agent          notes_qa_agent
   |                        |                         |
Google API              MCP 서버                    RAG
   |                        |                         |
   +------------------------+-------------------------+
                            v
                         Astra DB
```

### 에이전트별 역할

| 에이전트 | 역할 | 주요 연동 |
|---|---|---|
| `lunch_agent` | 실시간 주변 식당 검색, 내부 리뷰 검색, 부적절한 선택지 제외 | Google API, Astra DB, workflow, API, RAG |
| `ibm_specs_agent` | IBM 제품, watsonx Orchestrate, ADK/CLI 사용법 질의 응답 | Astra DB hybrid search, MCP client, `wxo-docs` MCP, RAG |
| `notes_qa_agent` | 회의·세미나 노트 검색 및 요약, 접근·개인정보 정책 적용 | Astra DB, RAG, RBAC, PII masking |

---

## 3. 주요 사용 시나리오

### 점심 식당 추천

`lunch_agent`는 “IFC 근처에서 오늘 점심으로 무엇을 먹을까?”와 같은 질문을 다음 방식으로 처리합니다.

- Google API를 통한 실시간 식당 검색
- Astra DB에 저장된 내부 식당 리뷰 검색
- 음식 종류 선택 또는 자유 입력을 통한 사용자 선호 수집
- 영업 여부 등 적합하지 않은 선택지 제외
- IBM 인턴 추천과 Google 추천을 구분한 최종 답변 제공

### IBM 제품 및 개발 도구 질문

`ibm_specs_agent`는 “watsonx Orchestrate가 무엇인가?”, “`agents import` 옵션은 무엇인가?”와 같은 질문에 답합니다. 제품 지식에는 RAG를 사용하고, 외부 문서와 최신 개발자 정보에는 MCP 연결을 사용합니다.

발표 자료에서는 출처 추적, 한국어 검색 품질, 제품 질문과 ADK/CLI 질문의 분리를 주요 설계 목표로 다룹니다. 기본 벡터 검색에서 Astra DB의 벡터·lexical hybrid search로 확장하고, 문서 제목 prefix와 한국어 기준 Granite embedding을 활용하는 전략도 제시합니다.

### 회의·세미나 노트 질의응답

`notes_qa_agent`는 회의 노트와 세미나 노트에서 관련 내용을 찾아 요약합니다. 보안 설계에는 다음 요소가 포함됩니다.

- RBAC: 익명 `Guest`는 notes agent에 접근할 수 없고, `role=intern` 인증 사용자는 접근 가능
- PII masking: 답변 반환 전 이름과 이메일 주소 마스킹
- supervisor 단계에서 욕설과 prompt injection 차단
- 색인된 노트에 근거가 없을 때 안전한 보류/불가 답변 제공

---

## 4. 엔터프라이즈 수준의 설계

단순한 에이전트 데모를 넘어 실제 운영에서 필요한 문제를 검토했습니다.

### 신뢰성과 오류 처리

- tool, flow, agent 단계의 실패를 구분
- `error_kind`, `retriable`, 사용자에게 보여줄 `note` 등 구조화된 실패 정보 반환
- 실패를 정상 답변으로 숨기지 않고 retry 및 failure branch로 처리
- 정상, no-match, 수집 실패 결과를 하나의 flow output으로 통합

### 데이터 일관성과 동시성

- 식당, 날짜, 이름을 기반으로 결정적 문서 ID 생성
- 동일 노트가 반복 처리되어도 중복이 생기지 않도록 upsert 방식 사용
- 동시 쓰기 상황에서 delete/insert 순서와 제한된 retry 로직 적용
- 발표 자료의 테스트에서는 문서 유실률이 51%에서 0%로, 쓰기 성공 건수가 30/48에서 48/48로 개선된 결과를 제시

### 자동화

회의 노트 적재는 다음과 같은 자동화 파이프라인으로 설계했습니다.

```text
Raw 회의 노트
   -> 전처리
   -> Astra DB 적재
   -> 검증
   -> 검색 가능한 노트
```

문서화된 GitHub Actions 흐름은 새 raw 파일을 감지하고, 전처리와 검증을 수행한 뒤 실패하면 파이프라인을 중단합니다. 새 raw 파일이 없으면 작업을 건너뜁니다.

### Observability

watsonx native observation과 Langfuse의 session 중심 tracing을 비교했습니다. 라우팅, plugin, tool 호출, prompt, token, session, tag 정보를 확인할 수 있도록 구성했으며, native export만으로 충분하지 않았던 테스트 흐름을 보완하기 위해 AgentOps REST polling과 Langfuse push를 함께 검토했습니다.

---

## 5. 평가

평가 흐름은 다음과 같습니다.

```text
Case 정의 -> watsonx Orchestrate 실행 -> JSON 결과·trace -> Feedback -> 반복 개선
```

평가 지표는 세 그룹으로 구성했습니다.

- Agent 지표: journey success, routing accuracy, total steps, LLM steps, 응답 시간, keyword match, semantic match, text match
- Tool 지표: total calls, expected/correct calls, missed calls, relevant calls, bad parameters, recall, precision, match success
- RAG custom rubric: faithfulness, factual correctness, answer relevance, context recall, citation accuracy, abstain accuracy

### 발표 자료에 기록된 평가 결과

- `ibm_specs_agent`: 7개 case 평가, 6개 통과 및 1개 실패
- Agent 수준 점수는 대부분 `1.00`, citation accuracy는 `0.86`으로 기록
- 테스트한 case에서 예상 tool 호출과 실제 호출이 일치
- 2회 실행 재현성 확인에서 보고된 지표의 변동 없음
- 실패한 `case07_related_product_trap`은 keyword 기반 평가의 한계와 더 강한 검증의 필요성을 드러냄

위 결과는 발표 자료에 담긴 프로토타입 평가 스냅샷이며, 운영 환경의 종합 벤치마크는 아닙니다. 발표 자료는 judge 기반 평가가 tool 누락을 놓칠 수 있고, 반복 판정에서 결과가 달라질 수 있으며, 원시 JSON이 복잡하면 검토가 어려워진다는 점도 지적합니다. 이를 보완하기 위해 IBM Bob 기반 재현성 확인, 독립적인 지표 검토, case별 Pass/Fail 근거를 요약하는 `SUMMARY.md` 방식을 제안합니다.

---

## 6. 폴더 구성

| 파일 | 설명 |
|---|---|
| `AskIntern_presentation.pdf` | 아키텍처, 에이전트 흐름, 엔지니어링 의사결정, 관측성, 평가 내용을 담은 발표 자료 |
| `demo_askintern.mp4` | AskIntern 사용 경험을 보여주는 데모 영상 |
| `README.md` | 영문 프로젝트 문서 |
| `README_KOR.md` | 한글 프로젝트 문서 |

현재 폴더에는 실행 가능한 전체 소스 코드가 아니라 발표 자료와 데모 산출물이 포함되어 있습니다. 따라서 배포 명령어와 환경 변수 설정은 제공하지 않습니다.

---

## 7. 핵심 정리

- 멀티 에이전트 라우팅으로 반복적인 인턴 업무 질문을 하나의 채팅 경험으로 통합했습니다.
- RAG, MCP, 외부 API를 모든 에이전트에 공통 노출하지 않고 역할에 맞게 분리했습니다.
- RBAC, PII masking, prompt injection 방어, retry, trace 수집을 핵심 요구사항으로 다뤘습니다.
- 엔터프라이즈 수준의 에이전트 개발에는 성공적인 데모 대화뿐 아니라 end-to-end 평가와 재현 가능한 피드백 루프가 필요합니다.
