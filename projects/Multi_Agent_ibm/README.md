# [Multi-Agent] AskIntern: IBM Intern Workspace Assistant

AskIntern is a multi-agent workspace assistant built with **IBM watsonx Orchestrate**. It is designed to answer the recurring questions IBM interns ask during the workday, such as where to have lunch, how IBM products work, and what was discussed in a meeting or seminar.

- [Korean](./README_KOR.md)
- Demo: [`demo_askintern.mp4`](./demo_askintern.mp4)
- Presentation: [`AskIntern_presentation.pdf`](./AskIntern_presentation.pdf)

---

## 1. Project Overview

- Goal: Build an end-to-end, trustworthy multi-agent assistant for IBM interns.
- Platform: IBM watsonx Orchestrate on IBM Code Engine
- Interface: Web chat
- Main orchestrator: `askintern_supervisor`
- Sub-agents: `lunch_agent`, `ibm_specs_agent`, `notes_qa_agent`
- Status: Prototype/demo implementation documented through the presentation and demo artifacts in this folder

AskIntern routes each user request to the appropriate specialist agent and combines the result into a single response. The system uses retrieval, external tools, workflow execution, access control, and response guardrails to make the assistant useful in an enterprise setting.

---

## 2. System Architecture

```text
Web Chat
   |
   v
askintern_supervisor  (watsonx Orchestrate)
   |------------------------|-------------------------|
   v                        v                         v
lunch_agent            ibm_specs_agent          notes_qa_agent
   |                        |                         |
Google API              MCP Server                 RAG
   |                        |                         |
   +------------------------+-------------------------+
                            v
                         Astra DB
```

### Agent responsibilities

| Agent | Responsibility | Main integrations |
|---|---|---|
| `lunch_agent` | Find nearby restaurants in real time, search internal review data, and filter unsuitable options | Google API, Astra DB, workflow, API, RAG |
| `ibm_specs_agent` | Answer questions about IBM products, watsonx Orchestrate, and ADK/CLI usage | Astra DB hybrid search, MCP client, `wxo-docs` MCP, RAG |
| `notes_qa_agent` | Retrieve and summarize meeting/seminar notes while enforcing access and privacy rules | Astra DB, RAG, RBAC, PII masking |

---

## 3. Main Use Cases

### Lunch recommendations

`lunch_agent` handles requests such as “What should I eat for lunch near IFC?” by combining:

- Real-time restaurant search through Google API
- Internal restaurant review lookup through Astra DB
- User preference collection, including cuisine type and free-form input
- Filtering for restaurants that are not currently available or suitable
- A final response that separates internal IBM recommendations from Google recommendations

### IBM product and developer questions

`ibm_specs_agent` answers questions such as “What is watsonx Orchestrate?” or “What does the `agents import` option do?”. The agent uses RAG for product knowledge and an MCP connection for external documentation and current developer information.

The design emphasizes source traceability, Korean search quality, and a clear distinction between a product lookup and an ADK/CLI lookup. The presentation describes a move from basic vector retrieval to Astra DB hybrid search using vector and lexical signals, together with title-prefix chunking and Korean-aware Granite embeddings.

### Meeting-note Q&A

`notes_qa_agent` retrieves information from meeting notes and seminar notes. The security design includes:

- RBAC: an anonymous `Guest` cannot access the notes agent, while an authenticated user with `role=intern` can be routed to it
- PII masking: names and email addresses are masked before the answer is returned
- Supervisor guardrails for profanity and prompt injection
- A safe abstention response when the requested information is not present in the indexed notes

---

## 4. Enterprise-Level Engineering

The project explores the operational concerns that become important beyond a toy agent demo.

### Reliability and error handling

- Separates tool, flow, and agent-level failures
- Returns structured failure information such as `error_kind`, `retriable`, and a user-facing note
- Uses retry and failure branches instead of allowing a failure to silently become a normal answer
- Consolidates normal, no-match, and collection-failure outcomes into a single flow output

### Data consistency and concurrency

- Uses deterministic document IDs based on restaurant, date, and name
- Uses upsert-style writes to avoid duplicate records when a note is processed repeatedly
- Handles concurrent writes with delete/insert ordering and bounded retry logic
- The presentation reports an improvement from 51% document loss to 0% in the tested concurrent-write scenario, and from 30/48 to 48/48 successful writes after retry handling

### Automation

Meeting-note ingestion is designed as an automated pipeline:

```text
Raw meeting note
   -> preprocessing
   -> Astra DB ingestion
   -> validation
   -> searchable note
```

The documented GitHub Actions flow detects new raw files, preprocesses them, validates the result, and stops the pipeline when validation fails. If no new raw file is found, the job is skipped.

### Observability

The project compares watsonx-native observations with Langfuse session-oriented tracing. The presentation highlights visibility into routing, plugins, tool calls, prompts, tokens, sessions, and tags. It also documents a two-path approach using AgentOps REST polling and Langfuse push because the native export path was not sufficient for the tested workflow.

---

## 5. Evaluation

The evaluation workflow is structured as:

```text
Case definition -> watsonx Orchestrate execution -> JSON result and traces -> feedback -> iteration
```

The project uses three groups of metrics:

- Agent metrics: journey success, routing accuracy, total steps, LLM steps, response time, keyword match, semantic match, and text match
- Tool metrics: total calls, expected/correct calls, missed calls, relevant calls, bad parameters, recall, precision, and match success
- Custom RAG rubrics: faithfulness, factual correctness, answer relevance, context recall, citation accuracy, and abstain accuracy

### Reported evaluation snapshot

- `ibm_specs_agent`: 7 cases evaluated, 6 passed and 1 failed
- Agent-level scores were mostly `1.00`; citation accuracy was reported as `0.86`
- Tool-call evaluation matched the expected calls in the tested cases
- Two-run reproducibility checks found no metric drift in the reported run
- The failed `case07_related_product_trap` exposed a limitation in keyword-based scoring and the need for stronger evaluation checks

These results are a documented prototype snapshot, not a production benchmark. The presentation also notes that judge-based evaluation can miss tool omissions, vary across repeated judgments, and become difficult to review when raw JSON is complex. The IBM Bob workflow and `SUMMARY.md` concept are proposed to add reproducibility checks, independent metric review, and concise case-level Pass/Fail explanations.

---

## 6. Repository Contents

| File | Description |
|---|---|
| `AskIntern_presentation.pdf` | Project presentation covering architecture, agent flows, engineering decisions, observability, and evaluation |
| `demo_askintern.mp4` | Demonstration video of the AskIntern experience |
| `README.md` | English project documentation |
| `README_KOR.md` | Korean project documentation |

The current folder contains presentation/demo artifacts rather than the full executable source code. Therefore, deployment commands and environment-variable setup are not included here.

---

## 7. Key Takeaways

- Multi-agent routing turns repeated intern questions into a single chat experience.
- RAG, MCP, and external APIs are assigned to the agents that need them instead of being exposed uniformly.
- RBAC, PII masking, prompt-injection defense, retries, and trace collection are treated as core product requirements.
- Enterprise-grade agent development requires end-to-end evaluation and reproducible feedback, not only successful demo conversations.
