# Campus Mate

Codex Community Hackathon — Seoul for Students · [Korean](./README_KOR.md)

[Live Site](https://campusmate.site) · [Demo Video](https://github.com/han942/codex-hackerthon/blob/main/campusmate_demo.mov) · [Event Page](https://codex-community-korea.skysplit.chatgpt.site/en/hackathon/seoul-2026) · [Source Repository](https://github.com/han942/codex-hackerthon)

## About The Project

Students have gaps between classes, but no way to know who else on campus is free at the same time.

Campus Mate reads a class timetable, computes the free periods, and matches two students whose gaps overlap — then suggests somewhere nearby to eat. The goal was to turn a timetable into a matching signal, built in a single day by a team formed on the spot.

### The Event

| | |
|---|---|
| Event | [Codex Community Hackathon — Seoul for Students](https://codex-community-korea.skysplit.chatgpt.site/en/hackathon/seoul-2026) |
| Organizer | Codex Community Korea |
| Date | August 16, 2026 · 09:00–21:00 |

### Built With

| Layer | Choice | Note |
|---|---|---|
| Frontend | React 19, TypeScript, Vite | Mock API mode for backend-free screen checks |
| Backend | Node.js 22, Express 5, TypeScript | `tsx` runtime, `vitest` for tests |
| Database | PostgreSQL 17 | Raw SQL migrations, no ORM |
| Auth | Supabase | External provider only — no self-built tokens |
| AI | OpenAI API + Zod | Structured output, schema-validated |
| Infra | Docker Compose, nginx | TLS proxy in front of the stack |

## Architecture

Campus Mate is a single Docker Compose stack with a deliberately separated React frontend and Express backend. The browser talks to the backend through the versioned `/api/v1` contract; authentication remains with Supabase rather than being reimplemented in the application.

```mermaid
flowchart LR
    Browser[Browser]

    subgraph Frontend[Frontend · React 19 + TypeScript + Vite]
        App[App.tsx<br/>auth state · tabs · screens]
        AuthClient[lib/auth.ts<br/>Supabase email auth · session storage]
        ApiClient[lib/api.ts<br/>API client · Bearer token · mock branch]
        Contracts[lib/contracts.ts<br/>UI/API contract types]
        App --> AuthClient
        App --> ApiClient
        ApiClient --> Contracts
    end

    Edge[Nginx<br/>TLS ingress · SPA serving · /api proxy]
    Supabase[Supabase Auth]

    subgraph Backend[Backend · Node.js + Express + TypeScript]
        Auth[requireAuth<br/>Supabase verifier or demo verifier]
        Routers[HTTP routers<br/>profile · schedule · match · venue<br/>conversation · proposal]
        Domain[Pure domain rules<br/>time.ts · social.ts]
        Port[CoreQueryPort<br/>stable seam between backend areas]
        Store[SocialStore]
        AI[RecommendationAi<br/>optional OpenAI adapter]
        Fallback[Rule ranking + template reasons]
        Auth --> Routers
        Routers --> Domain
        Routers --> Port
        Routers --> Store
        Routers --> AI
        AI -. unavailable or invalid .-> Fallback
    end

    DB[(PostgreSQL 17<br/>raw SQL migrations)]
    OpenAI[OpenAI Responses API]

    Browser --> App
    AuthClient -->|email/password · refresh token| Supabase
    ApiClient -->|/api/v1 + Authorization: Bearer| Edge
    Edge -->|static SPA| App
    Edge -->|API reverse proxy| Auth
    Port --> Store
    Store --> DB
    AI --> OpenAI
    AI -. failure .-> Fallback

    classDef frontend fill:#e8f0fe,stroke:#4a6da7,color:#1f2328
    classDef backend fill:#fdf0e3,stroke:#c98b3a,color:#1f2328
    classDef external fill:#e9f5ec,stroke:#4a8a5f,color:#1f2328
    class App,AuthClient,ApiClient,Contracts frontend
    class Auth,Routers,Domain,Port,Store,AI,Fallback backend
    class Supabase,DB,OpenAI,Edge external
```

### Frontend architecture

The frontend is a Vite-built React SPA. The current implementation keeps screen composition and local UI state in `frontend/src/app/App.tsx` rather than introducing a separate router or state-management library.

| Area | Implementation | Responsibility |
|---|---|---|
| App shell | `src/main.tsx`, `src/app/App.tsx` | Auth loading, login/onboarding guard, tab navigation, screens, proposal refresh |
| Authentication | `src/lib/auth.ts` | Calls Supabase Auth, persists access/refresh tokens in `localStorage` or `sessionStorage`, injects the access token into the API client |
| API boundary | `src/lib/api.ts` | Calls `/api/v1`, adds Bearer headers, normalizes API errors, and switches to mock data when `VITE_USE_MOCK_API=true` |
| Contract types | `src/lib/contracts.ts` | TypeScript representations of the API response and request shapes |
| Static serving | `frontend/nginx.conf` | SPA fallback for client-side paths and `/api/` proxying in the container; Vite proxies `/api` during local development |

The production and mock paths share the same UI contract types. Mock mode replaces the network calls with local in-memory state for schedules, availability, matches, venues, and proposals; it does not change the screens or user flow.

### Backend architecture

The backend is assembled through `createApp()`, which accepts the store, token verifier, `CoreQueryPort`, AI adapter, and clock as dependencies. This keeps HTTP tests independent of PostgreSQL and makes the boundary between the two backend workstreams explicit.

| Layer | Location | Responsibility |
|---|---|---|
| HTTP boundary | `backend/src/http/*.ts` | Authentication-aware routes, Zod input validation, pagination, and consistent error responses |
| Core Time | `profile-routes.ts`, `schedule-routes.ts`, `core-query-service.ts` | Profile, match preferences, schedules, service-window free time, preferred availability, and the public match view |
| Social Flow | `match-routes.ts`, `match-conversation-routes.ts`, `venue-routes.ts`, `proposal-routes.ts` | Candidate matching, natural-language chat, venue ranking, proposal creation/listing, and status transitions |
| Domain rules | `backend/src/domain/time.ts`, `domain/social.ts` | Pure functions for free-time calculation, slot intersection, scoring, venue ranking, conflicts, and state transitions |
| Port and store | `domain/core-query-port.ts`, `core-query-service.ts`, `store.ts`, `postgres-store.ts` | Hides timetable/profile reads behind `CoreQueryPort`; supports an in-memory store for tests and a PostgreSQL store for runtime |
| AI boundary | `ai-recommendation-service.ts` | Structured intent parsing and re-ranking through Zod-validated responses, with a disabled adapter when no API key is configured |

Backend A owns the Core Time data and APIs. Backend B consumes only the following interface instead of reading A's timetable tables directly:

```ts
interface CoreQueryPort {
  getUserMatchView(userId: string): Promise<UserMatchView>;
  listDiscoverableCampusUsers(campusId: string, excludeUserId: string): Promise<UserMatchView[]>;
  getEffectiveSlots(userId: string): Promise<TimeSlot[]>;
}
```

The HTTP contract is split accordingly:

| Backend A · Core Time | Backend B · Social Flow |
|---|---|
| `GET /schools` | `GET /matches` |
| `GET /schools/:schoolId/campuses` | `POST /match-conversations/messages` |
| `GET /profile-options` | `GET /venues/recommendations` |
| `GET/PUT /me/profile` | `POST /meeting-proposals` |
| `GET/PUT /me/match-preferences` | `GET /meeting-proposals` |
| `GET/POST /me/schedules` | `PATCH /meeting-proposals/:proposalId/status` |
| `PATCH/DELETE /me/schedules/:scheduleId` |  |
| `GET /me/free-times` |  |
| `GET/PUT /me/availability` |  |

### Matching and proposal data flow

1. Profile, class schedules, and preferred lunch slots are stored by Core Time. Free time is calculated inside the `11:00–15:00` service window and intersected with preferred availability.
2. Social Flow obtains a restricted `UserMatchView` and effective slots through `CoreQueryPort`. The server first filters by same school/campus, discoverability, activity, common time, and minimum meeting duration.
3. Rule-scored candidates are optionally re-ranked by AI. The server accepts only returned IDs that were already in the filtered candidate set; malformed or unavailable AI responses fall back to rule ranking and template reasons.
4. Venue recommendations use seeded, active venues. The server rule-ranks up to 30 candidates and may ask AI to select up to 3. A custom venue remains available when no recommendation exists.
5. Proposal creation and acceptance both re-check common availability and accepted-proposal conflicts. A user lock and conditional status update prevent two concurrent acceptances from succeeding. Accepted `MeetingProposal` records are displayed as appointments; there is no separate `Appointment` table.

## Getting Started

### Prerequisites

- Node.js 22 or later
- Docker

### Installation

Full stack (frontend + backend + PostgreSQL):

```bash
git clone https://github.com/han942/codex-hackerthon.git
cd codex-hackerthon
cp .env.example .env      # set POSTGRES_PASSWORD, and SUPABASE_* if using real auth
docker compose up -d --build
# frontend  http://localhost:5173
# backend   http://localhost:3000
docker compose down
```

Backend only:

```bash
cd backend
cp .env.example .env
docker compose up -d       # PostgreSQL
npm run migrate
npm run seed
npm start
npm test                   # in-memory store, no database needed
```

Notes:

- `VITE_USE_MOCK_API=true` runs the frontend against mock data with no backend at all.
- Local demo auth uses `Authorization: Bearer demo:user_a`.
- `OPENAI_API_KEY` is optional — without it the AI paths fall back to rule-based ranking.

## Usage

1. **Sign in** and complete the profile — school, campus, interests.
2. **Register your timetable**, then add your preferred lunch times. This step is required: with no registered availability there are no common free periods to compute. The server derives free time inside an `11:00–15:00` window.
3. **Ask in the chat** — "목요일 12시에 한 시간 점심 친구 찾아줘" — or browse the mate list directly. Candidates are filtered to the same campus, overlapping free time, and both sides' minimum meeting duration.
4. **Pick a venue** from the 3 recommendations, ranked by walking distance, budget, and remaining time — or type your own (2–50 characters).
5. **Send the proposal.** The recipient accepts or rejects; accepted proposals appear for both sides under appointments.

A `409` conflict on creation or acceptance is a normal part of the flow — the common free time is re-checked at both points, and the UI re-offers the current slots.

## Development Timeline

```mermaid
flowchart LR
    A["Functional spec<br/>docs/funtiondalspec.md<br/>(single source of truth)"] --> B["API contract<br/>docs/api/*.md"]
    B --> C["Work-split guides<br/>docs/backend/*.md"]
    C --> D["Backend A · Core Time<br/>feat/be-core-time"]
    C --> E["Backend B · Social Flow<br/>feat/be-social-flow"]
    C --> F["Frontend<br/>feat/frontend"]
    D -->|CoreQueryPort| E
    D --> G["Integration<br/>migrate · seed · smoke test"]
    E --> G
    F --> G
    G --> H["Docker Compose + nginx<br/>campusmate.site"]

    classDef spec fill:#e8f0fe,stroke:#4a6da7,color:#1f2328
    classDef dev fill:#fdf0e3,stroke:#c98b3a,color:#1f2328
    classDef ship fill:#e9f5ec,stroke:#4a8a5f,color:#1f2328
    class A,B,C spec
    class D,E,F dev
    class G,H ship
```

> Spec first, contract second, code third — so three agents could work in parallel without stepping on each other

| Time (KST) | Event |
|---|---|
| 09:00 | Event opens; teams formed on site from strangers |
| 14:16 | Initial commit, repo setup |
| 14:33 – 14:49 | Functional spec, API docs, backend work-split guide |
| 14:56 – 15:00 | Backend skeleton, core-time APIs, auth provider, contract tests |
| 15:24 – 15:39 | First frontend screens; match & proposal routes |
| 16:06 – 16:22 | PostgreSQL persistence, demo seed (100 members) |
| 16:33 – 16:37 | Chat-based AI matching API, frontend ↔ server integration |
| 16:51 – 17:26 | Deployment fixes, TLS proxy, demo video |
| 17:30 | Codex Build Logs and presentation submitted |
| 21:00 | Event closes |

Roughly **3 hours 15 minutes** of commits, inside a 12-hour day that also had to cover meeting the team, agreeing on a problem, and preparing the presentation.

### How the parallel work was organized

| Member | Role | Main output |
|---|---|---|
| 신진범 (bumsoft) | Backend A — Core Time | Server skeleton, auth, profile, schedule CRUD, free-time calculation, infra & deploy |
| 한승원 (han942) | Backend B — Social Flow | Match/venue/proposal routes, chat-based AI matching API |
| HangJun | Frontend | React screens, timetable UI, chat integration |
| 박진희 | Spec | Functional specification, presentation |

Four people who had not worked together before needed a way to write code simultaneously without colliding. The answer was to spend the first hour writing documents instead of code:

1. **Fix a single source of truth.** `docs/funtiondalspec.md` defines scope and rules, with an explicit precedence order — spec > API docs > code — and a rule that no P0 feature may be added from the API docs alone.
2. **Turn the spec into an HTTP contract** (`docs/api/`) before any implementation, so frontend and backend could start simultaneously against the same endpoints.
3. **Split the backend into two non-overlapping owners** with a file-ownership table and an explicit "do not touch" list.
4. **Decouple the two halves with an interface.** Backend B never queries A's timetable tables — it goes through `CoreQueryPort`, built against a **fake implementation** until A's real one landed.

   ```ts
   interface CoreQueryPort {
     getUserMatchView(userId: string): Promise<UserMatchView>;
     listDiscoverableCampusUsers(campusId: string, excludeUserId: string): Promise<UserMatchView[]>;
     getEffectiveSlots(userId: string): Promise<TimeSlot[]>;
   }
   ```

5. **Integrate in a fixed order** — A's skeleton merges first, B rebases, the fake port is swapped for the real one, then migrate/seed/test/smoke on a clean clone.

### AI design

AI is used as a **re-ranker inside a rule-filtered candidate set**, never as the source of truth:

- The server first enforces same school/campus, discoverability, common available time, and minimum meeting duration
- Only anonymized candidate IDs and shared-attribute evidence reach the model — no emails, course names, or full timetables
- The model re-ranks at most **5 mates** (from a rule-scored top 50) and **3 venues** (from top 30), and returns IDs the server re-validates
- Every AI response is parsed through a Zod schema; on failure the system falls back to rule-based ranking with templated reasons

The natural-language intent parser follows the same pattern: the model extracts date, time, duration, budget, and atmosphere, and the server does the actual matching.

## Roadmap

Cut from the one-day build on purpose: self-built token/session APIs, block & report, timetable OCR, live venue search, and any activity other than `LUNCH`. Real-time notifications were also left out.

The event page will become a public archive after results are confirmed:

- [ ] Winning teams and selection rationale
- [ ] Project gallery of participating teams
- [ ] Public GitHub and demo links
- [ ] Event recap and verified participation statistics

## Takeaways

- **Writing the contract before the code is what made parallel agent work possible.** Four strangers generating code simultaneously will collide unless ownership and endpoints are decided up front.
- **An interface seam (`CoreQueryPort`) plus a fake implementation removed the blocking dependency** — Backend B was not idle while Backend A built the foundation.
- **Constraining the AI beats trusting it.** Rules filter, AI re-ranks, server re-validates, fallback always exists. The demo works even with the OpenAI key removed.
- **A written "do not build this" list** kept scope from expanding past what a single day allows.
- **The build log is part of the deliverable.** Because judging reviewed *how* Codex was used and recovered from, keeping a clean session log mattered as much as shipping the feature.

## Contact

Seung-Won Han — [@han942](https://github.com/han942)

Source repository: https://github.com/han942/codex-hackerthon

> This folder is a write-up of a hackathon submission, not an open-source project accepting contributions. The source repository declares no license.

## Acknowledgments

- **Organizer** — [Codex Community Korea](https://codex-community-korea.skysplit.chatgpt.site/)
- **Co-hosts** — [투빅스 (ToBigs)](https://www.datamarket.ai.kr/), [가짜연구소 (Pseudo Lab)](https://pseudo-lab.com/), [비타민 (BITAmin)](https://www.bitamin.ai.kr/)
- **Partners** — [OpenAI Codex](https://openai.com/codex/), [AWS](https://aws.amazon.com/), [Runpod](https://www.runpod.io/), Elev8, [DEVOCEAN](https://devocean.sk.com/), Hugging Face KREW, [Endplan](https://endplan.ai/ko)
- Teammates 신진범, HangJun, and 박진희, met on the morning of the event
