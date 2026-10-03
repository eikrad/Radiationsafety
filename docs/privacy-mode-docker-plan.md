# Plan: Privacy mode with Docker

**Status:** planned, not started. Prompted by [#106](https://github.com/eikrad/Radiationsafety/issues/106).

## Why

Privacy mode (Ollama) is only documented for a local install without Docker: install Ollama,
pull the models, `uv sync`, run the ingestion. On Windows that means setting up WSL2 by hand.
#106 shared a setup script for Windows 11 + WSL2 + Ollama, but it is tailored to one machine
(an RTX A1000 with 6 GB VRAM, fixed Ollama tuning flags, WSL only).

The Docker setup already runs on Windows, macOS and Linux, but it has no Ollama: the backend
container cannot reach `localhost:11434`, because inside the container that is the container
itself. The goal is to close that gap without a per-machine script.

## Goal

```bash
docker compose up                     # as today: cloud providers, no Ollama, no download
docker compose --profile ollama up    # adds an Ollama container; the models are pulled once
```

- Privacy mode can be added later without reinstalling: start with the profile.
- An Ollama already installed on the host can be used instead of the container.
- Nothing is tailored to one GPU; without a GPU, Ollama runs on the CPU.

## What the code already does

- **Privacy mode is chosen per request** ("Ollama (Local)" in the UI, `model == "ollama"` in
  `api/main.py`). `.env` does not need `LLM_PROVIDER=ollama`; cloud and local answers work
  side by side in one installation.
- **The backend connects to Ollama per request**, not at startup (`graph/llm_factory.py`), so the
  backend does not need a `depends_on` on Ollama.
- **Local embeddings have their own collections** (`radiation-iaea-ollama`,
  `radiation-dk-law-ollama`, `ingestion.py`) and are built by a separate ingestion run with
  `LLM_PROVIDER=ollama`.
- **The error message when Ollama is unreachable** says "Start it with: ollama serve"
  (`_ollama_error_detail` in `api/main.py`), which does not help under Docker.

## Steps

### 1. `docker-compose.yml`: profile `ollama`

- **`ollama`**: pinned `ollama/ollama` image, named volume `ollama-models:/root/.ollama` so the
  models survive restarts, healthcheck via `ollama list`, `restart: unless-stopped`, **no port
  published to the host** (the backend reaches it over the compose network). Apply the same
  hardening as the other services (`no-new-privileges`, `cap_drop` where Ollama still runs).
- **`ollama-pull`**: one-shot service with the same image and `OLLAMA_HOST=ollama:11434`, same
  pattern as `chroma-permissions`. Waits for the `ollama` healthcheck, pulls
  `${OLLAMA_MODEL:-llama3.1:8b}` and `${OLLAMA_EMBED_MODEL:-nomic-embed-text}` (about 5 GB),
  then exits. Later starts finish in seconds because the models are already in the volume.
- **`backend`**: `OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://ollama:11434}`, so without an entry
  in `.env` it points at the container and with one at an external Ollama. Add
  `extra_hosts: ["host.docker.internal:host-gateway"]` so `host.docker.internal` also resolves
  on Linux.

### 2. `docker-compose.gpu.yml` (optional override)

- Only the NVIDIA reservation (`deploy.resources.reservations.devices`) for `ollama`.
- Used as `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile ollama up`,
  or via `COMPOSE_FILE` in `.env`.
- Works on Linux with the NVIDIA Container Toolkit and on Windows with Docker Desktop (WSL2
  backend). Not on macOS: Docker on Mac has no GPU access.
- Low-VRAM tuning (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`,
  `OLLAMA_NUM_PARALLEL=1`, as in #106) as commented hints, not as defaults.
- AMD (ROCm image): mentioned in the docs only.

### 3. `.env.example`

- Commented `COMPOSE_PROFILES=ollama` for people who always want privacy mode.
- The three `OLLAMA_BASE_URL` variants: local run without Docker (`localhost`), Docker with the
  container (default, leave unset), Docker with Ollama on the host (`host.docker.internal`).
- Linux with host Ollama: Ollama must listen on all interfaces (`OLLAMA_HOST=0.0.0.0`).

### 4. Error message (`api/main.py`)

When Ollama is unreachable, `_ollama_error_detail` names the configured `OLLAMA_BASE_URL` and
both remedies (`ollama serve`, or `--profile ollama` under Docker). Test in `tests/test_api.py`.

### 5. README: "Privacy mode with Docker"

- **With the container**: command, about 5 GB on first start, progress via
  `docker compose logs -f ollama-pull`, local index via
  `docker compose run --rm -e LLM_PROVIDER=ollama backend python ingestion.py`, and how long
  that takes on a CPU (to be measured).
- **With an existing Ollama**: `OLLAMA_BASE_URL=http://host.docker.internal:11434`, plus the Linux
  note.
- **GPU**: the override command and the prerequisites per OS.
- **Per platform**: on macOS native Ollama is recommended (GPU via Metal); on Windows with an
  NVIDIA GPU the container is a good fit.

### 6. CI (`docker-integration` job)

`docker compose --profile ollama config --quiet` keeps the profile valid. Pulling models in CI is
out of scope (5 GB per run); starting the Ollama container without models is possible but adds an
image of 1–2 GB per run, so it is left out unless the profile breaks in practice.

## Verification

- `docker compose up` without the profile behaves exactly as before.
- With the profile and **small models** (e.g. `OLLAMA_MODEL=qwen2.5:0.5b`,
  `OLLAMA_EMBED_MODEL=all-minilm`): pull, healthcheck, backend → Ollama, ingestion into the
  `-ollama` collections, one `/api/query` with `model=ollama`.
- The same with the default models once, to measure the CPU ingestion time for the README.
- The GPU override on real NVIDIA hardware: the maintainer's card, and if possible a 6 GB card
  (RTX A1000, @luttegu in #106) to check the low-VRAM hints.
- `uv run pytest tests/ -v` and the pre-commit checks.

## Decisions

- **Default model** stays `llama3.1:8b`, as in the README; smaller models are one `.env` line away.
- **No automatic model download** without the profile: the 5 GB only come when someone asks for
  privacy mode in Docker.

## Out of scope

- A one-click Windows installer (.exe). Bundling the backend (Chroma, LangChain) with PyInstaller,
  code signing to avoid SmartScreen warnings, shipping or building the index, and keeping Ollama
  separate make it a project of its own. Revisit if non-technical users ask for it.
- Shipping a prebuilt index: depends on the licence of the IAEA documents.
