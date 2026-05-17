# AGENTS.md — Radiation Safety RAG

Dieses Dokument ist die zentrale Referenz für alle AI-Agenten (Claude Code, Codex, Cursor, Gemini CLI, etc.), die an diesem Projekt arbeiten oder die Logseq-Wissensbasis nutzen.

---

## 1. Projekt-Überblick

RAG-System zur Abfrage von IAEA- und dänischen Strahlenschutzdokumenten.

- **Backend**: FastAPI + LangGraph-Workflow (`graph/`) + Chroma-Vektordatenbank
- **Embeddings**: immer Gemini (`GOOGLE_API_KEY` erforderlich für Ingestion und Retrieval)
- **LLM für Generierung**: konfigurierbar — `gemini`, `openai`, oder `mistral` via `LLM_PROVIDER`
- **Frontend**: React/TypeScript in `frontend/`
- **Dokumente**: `documents/IAEA/`, `documents/IAEA_other/`, `documents/Bekendtgørelse/`

---

## 2. Codebasis-Karte

```
api/main.py              — FastAPI-Routen, Admin-Auth, Rate-Limiting
graph/graph.py           — LangGraph-Workflow (Knoten, Kanten, Routing)
graph/nodes/             — retrieve, grade_documents, generate, web_search, verify_trusted
graph/chains/            — LLM-Chains (generation, grading, search-query, truncate)
graph/llm_factory.py     — LLM-Provider-Auswahl (Gemini/OpenAI/Mistral)
graph/state.py           — GraphState TypedDict
graph/consts.py          — Knotennamen, env_bool()
ingestion.py             — PDF/XML-Laden, Chunking, Chroma-Befüllung
ingestion_fetch.py       — URL-Fetch-Logik für retsinformation.dk und IAEA
build_document_sources.py — Erstellt document_sources.yaml aus lokalen PDFs
document_updates.py      — Prüft auf neuere Versionen (retsinformation.dk, IAEA)
eval/                    — RAGAS-Evaluierung (run_eval.py, golden.json)
tests/                   — pytest-Suite
frontend/src/App.tsx     — Haupt-UI-Komponente
frontend/src/constants.ts — API-URLs, Konfiguration
```

### Neue Knoten hinzufügen

1. Datei in `graph/nodes/` erstellen, Funktion `(state: GraphState) -> dict` implementieren
2. In `graph/nodes/__init__.py` exportieren
3. In `graph/graph.py` mit `workflow.add_node(NAME, fn)` und Kanten registrieren
4. Konstante in `graph/consts.py` ergänzen

### Neue Chains hinzufügen

1. Datei in `graph/chains/` erstellen, `get_*`-Factory-Funktion implementieren
2. In `graph/chains/__init__.py` exportieren

---

## 3. Entwicklungskonventionen

- **Python**: `uv` für Abhängigkeiten, `uv run pytest tests/ -v` für Tests
- **Frontend**: `npm -C frontend run test`, `npm -C frontend run build`
- **Linting**: pre-commit hooks (`.pre-commit-config.yaml`)
- **Umgebungsvariablen**: Immer `.env.example` aktualisieren wenn neue Variablen hinzukommen
- **Chroma-Collections**: `radiation-iaea` und `radiation-dk-law` — nicht umbenennen ohne Re-Ingestion
- **Admin-Routen**: erfordern `X-Admin-Token`-Header; ohne `ADMIN_TOKEN` → 503

---

## 4. Logseq Second Brain — Schema und Workflows

Dieses Projekt nutzt **Logseq** (via `mcp-logseq`) als kompilierte Wissensbasis für Research zu Strahlenschutz. Das Konzept folgt Karpathys LLM-Wiki-Prinzip: Raw Sources bleiben unverändert, das LLM pflegt verlinkte Seiten in Logseq.

### 4.1 Namespaces

```
Sources/IAEA/          — IAEA-Standards und Safety Guides (GSR, SSG, SSG, TECDOC)
Sources/Danish/        — Dänische Bekendtgørelser (retsinformation.dk)
Sources/Research/      — Wissenschaftliche Paper, externe Studien
Sources/Other/         — Podcasts, Blogs, sonstige Quellen
Concepts/              — Schlüsselbegriffe (Dosimetry, ALARA, Contamination, ...)
Regulations/           — Regulatorische Rahmenwerke und Vergleiche
Index                  — Master-Index aller Seiten mit Kurzbeschreibungen
```

### 4.2 Standard-Properties pro Seite

Jede neue Seite soll diese Properties im Frontmatter haben:

```
source-type::   iaea-standard | iaea-tecdoc | danish-law | paper | book | podcast | other
document-id::   z.B. GSR-3, SSG-46, BEK-2025-138, TECDOC-1380
topic::         dosimetry | transport | medical | research | waste | emergency | regulatory
language::      en | da | de
status::        ingested | reviewed | needs-update | superseded
date::          YYYY-MM-DD (Publikationsdatum wenn bekannt)
url::           (optional, Quell-URL)
```

### 4.3 Ingest-Workflow (neue Quelle)

Wenn eine neue Quelle (PDF, Artikel, Podcast-Notiz) hinzukommt:

1. **`create_page`** — Neue Seite unter passendem Namespace anlegen
   - Titel: `Sources/IAEA/GSR-3` oder `Concepts/ALARA`
   - Properties gemäß 4.2 setzen
   - Inhalt: Zusammenfassung, Schlüsselaussagen, relevante Paragraphen als Blöcke

2. **`query`** — Verwandte Seiten finden:
   ```clojure
   [:find (pull ?p [:block/name])
    :where [?p :block/properties ?props]
           [(get ?props :topic) ?t]
           [(= ?t "dosimetry")]]
   ```

3. **`update_page`** — 3–7 verwandte Seiten mit Back-Links und neuen Querverweisen aktualisieren (append-Modus)

4. **`update_page`** — `Index`-Seite mit Eintrag für die neue Seite ergänzen

### 4.4 Query-Workflow (Kontext für eine Frage)

Bevor du eine komplexe Strahlenschutz-Frage beantwortest:

1. **`find_pages_by_property`** — Nach `topic` oder `document-id` filtern
2. **`query`** — Datalog für präzise Kombinationssuche (z.B. topic=transport AND language=en)
3. **`get_page_content`** — Nur die relevanten Seiten laden (Kontext klein halten)
4. **`get_page_backlinks`** — Verwandte Konzepte via Graph traversieren wenn nötig
5. **`search`** — Fallback für Volltextsuche wenn Properties nicht ausreichen

Ziel: so wenig Seiten wie nötig laden, dann im LLM synthetisieren.

### 4.5 Lint-Routine (periodisch)

Gelegentlich zur Qualitätssicherung:

1. **`query`** — Seiten ohne `source-type`-Property finden (fehlende Metadaten)
2. **`query`** — Seiten mit `status: needs-update` prüfen
3. **`get_page_backlinks`** — Waisen-Seiten identifizieren (keine eingehenden Links)
4. Veraltete Dokumente mit `status: superseded` markieren wenn neuere Version vorhanden

### 4.6 Wichtige Logseq-MCP-Tools (Kurzreferenz)

| Tool | Wann |
|---|---|
| `create_page` | Neue Quelle ingestieren |
| `update_page` | Cross-References ergänzen (append), Inhalt korrigieren (replace) |
| `update_block` | Einzelnen Block via UUID anpassen |
| `find_pages_by_property` | Schnelle Property-Filter (topic, status, document-id) |
| `query` | Komplexe Datalog-Queries |
| `get_page_backlinks` | Welche Seiten verlinken auf X? |
| `get_pages_from_namespace` | Alle Seiten unter Sources/IAEA/ |
| `search` | Volltextsuche als Fallback |

### 4.7 Beispiel-Queries

```clojure
;; Alle IAEA-Standards zum Thema Transport
[:find (pull ?p [:block/name :block/properties])
 :where [?p :block/properties ?props]
        [(get ?props :source-type) "iaea-standard"]
        [(get ?props :topic) "transport"]]

;; Alle Seiten die überprüft werden müssen
[:find (pull ?p [:block/name])
 :where [?p :block/properties ?props]
        [(get ?props :status) "needs-update"]]

;; Alle dänischsprachigen Quellen
[:find (pull ?p [:block/name])
 :where [?p :block/properties ?props]
        [(get ?props :language) "da"]]
```

---

## 5. Dokumente im Projekt (bereits vorhanden)

Diese PDFs liegen lokal und sind bereits in Chroma ingested:

**IAEA Standards:**
GSR-1, GSR-2, GSR-3, GSR-4, GSR-5, GSR-6, GSR-7,
SSG-11, SSG-39, SSG-40, SSG-44, SSG-46, SSG-86, SSG-87,
SSR-6, TECDOC-1380, TECDOC-1638, nuclear_safety_measures (24G)

**Dänische Quellen (Bekendtgørelse):**
BEK-2025-138405, BEK-2025-138505, Brug af åbne radioaktive kilder,
Udarbejdelse af en sikkerhedsvurdering

Diese sollten als erste Seiten in Logseq unter `Sources/IAEA/` bzw. `Sources/Danish/` angelegt werden.
