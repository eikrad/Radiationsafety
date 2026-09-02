# Frontend

React + TypeScript (Vite) chat UI for the Radiation Safety RAG backend.

See the [root README](../README.md) for the full project overview and [docs/architecture.md](../docs/architecture.md) for how the backend pipeline works.

## Structure

```
src/
  App.tsx                   — top-level app: chat state, model/history handling
  constants.ts              — API base path, model list, model variants, localStorage keys
  storage.ts                — localStorage helpers (API keys, model variant, privacy-mode flag)
  types.ts                  — shared TypeScript types (Message, QueryResponse, ...)
  components/
    QueryForm.tsx            — question input + submit
    ResponseDisplay.tsx      — renders answer, sources, and routing warnings
    ModelSelector.tsx        — LLM provider/model picker
    SettingsModal.tsx        — API key entry (stored in sessionStorage, never persisted) and preferences
    DocumentsPanel.tsx       — document management entry point
    DocumentListSidebar.tsx  — list of ingested document sources
    DocumentUpdatesModal.tsx — check-for-updates / re-ingest UI
e2e/                        — Playwright end-to-end specs (mocked API)
```

API calls go through `API_BASE` (`/api`), proxied to the backend by Vite in dev and by nginx in the Docker build.

## Setup

```bash
npm install
```

## Development

```bash
npm run dev
```

Opens a dev server with hot reload at http://localhost:5173. The backend must be running separately (see root README) — Vite proxies `/api` requests to it.

## Build

```bash
npm run build   # production build to dist/
npm run preview # preview the production build locally
```

## Testing

| Command | What it runs |
|---|---|
| `npm run test` | Unit tests (Vitest) — run once |
| `npm run test:watch` | Unit tests in watch mode |
| `npm run test:e2e` | Playwright end-to-end tests against a **mocked** API (`e2e/`) — run `npx playwright install --with-deps chromium` once first |
| `npm run test:e2e:ui` | Playwright tests with the interactive UI runner |
| `npm run lint` | ESLint |

Unit tests live alongside the components they cover (e.g. `src/components/QueryForm.test.tsx`).

## Notes

- Built with [Vite](https://vite.dev) + [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react) (Babel-based Fast Refresh) and [`typescript-eslint`](https://typescript-eslint.io) for linting/type-checking.
- API keys entered in the Settings modal are kept in `sessionStorage` only — never sent anywhere but the backend, never persisted to disk.
