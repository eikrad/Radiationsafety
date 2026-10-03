# Changelog

All notable changes to this project are documented in this file.

<!-- Versions from 0.5.0 on are written by release-please from Conventional Commit
     messages (see docs/releasing.md). Notes that commits cannot carry,
     such as upgrade steps, are added by hand below the generated entry. -->

## [0.5.0](https://github.com/eikrad/Radiationsafety/compare/v0.4.0...v0.5.0) (2026-10-03)


### ⚠ BREAKING CHANGES

* a setup without LLM_PROVIDER / EMBEDDING_PROVIDER needs SCW_SECRET_KEY and the Scaleway collections (ingestion.py --reembed-from gemini); set both to gemini to keep the previous behaviour.

### Features

* add privacy mode localStorage functions (Phase 2) ([74ffeb7](https://github.com/eikrad/Radiationsafety/commit/74ffeb7b929b8c3004c99bcb6f79ddd4fef280ac))
* add privacy mode toggle UI in Settings (Phase 3) ([d4f6404](https://github.com/eikrad/Radiationsafety/commit/d4f64043b99cedf2a9bdceeef353c5bdd64f2331))
* add Privacy Mode with Ollama for fully local LLM + embeddings ([2d6d9c8](https://github.com/eikrad/Radiationsafety/commit/2d6d9c8664b843acdc83f491d13913f2679b7cce))
* add retry logic for Ollama batch embeddings ([de735b8](https://github.com/eikrad/Radiationsafety/commit/de735b8d562255f9de92be376c35f58796b49a48))
* **api:** report per-provider readiness in /config ([f31ae19](https://github.com/eikrad/Radiationsafety/commit/f31ae190ffaaca2ee2593becbf186e030f3fe100))
* **api:** say why a provider needs SCW_EMBED_MODEL ([869fd5b](https://github.com/eikrad/Radiationsafety/commit/869fd5b2e4243461861c81f837230c24d07e8c50))
* disable non-ollama models in ModelSelector when privacy mode on (Phase 4) ([a0eac3c](https://github.com/eikrad/Radiationsafety/commit/a0eac3ca514641fb86076bc8f8982e3319ab9f12))
* **embeddings:** Scaleway embeddings, chosen separately from the LLM ([71b6e13](https://github.com/eikrad/Radiationsafety/commit/71b6e1355812099a9be10f1b7a43b1093417db3a))
* **embeddings:** Scaleway embeddings, compared with Gemini by retrieval-only eval ([aad1cd8](https://github.com/eikrad/Radiationsafety/commit/aad1cd8870b27815ae0829903cb55834a9adcce5))
* **eval:** --regrade measures the sufficiency grader alone ([71ae09b](https://github.com/eikrad/Radiationsafety/commit/71ae09b232dbd48dd28aafdef330b6873d0c0b97)), closes [#129](https://github.com/eikrad/Radiationsafety/issues/129)
* **eval:** add 11 IAEA questions to the golden set ([c5c54d7](https://github.com/eikrad/Radiationsafety/commit/c5c54d756b163fb7ac15eb488070bc49e7e54c64))
* **eval:** apply the golden set review ([6a60a35](https://github.com/eikrad/Radiationsafety/commit/6a60a35aac9ce1235f1ca3f2aca1bf5aea3d32fe))
* **eval:** backfill history from existing reports ([055b7b8](https://github.com/eikrad/Radiationsafety/commit/055b7b87a4324c81659554199784e789b963b272))
* **eval:** capture git state and prompt fingerprints for each run ([ec84bbd](https://github.com/eikrad/Radiationsafety/commit/ec84bbdbee20b03baf1e8eba52e1c8480696897c))
* **eval:** capture per-node graph outputs for scoring ([b651b4d](https://github.com/eikrad/Radiationsafety/commit/b651b4d27865087296627a140f3abc39df4bda65))
* **eval:** check evidence quotes against the index and pool unjudged chunks ([68747bc](https://github.com/eikrad/Radiationsafety/commit/68747bceb520dfea5d50ff395e6652d26614faf4)), closes [#133](https://github.com/eikrad/Radiationsafety/issues/133)
* **eval:** compare retrieval-only runs by evidence rank in the dashboard ([1df60ab](https://github.com/eikrad/Radiationsafety/commit/1df60ab33daba660217006ce7682ac44567ccb20)), closes [#128](https://github.com/eikrad/Radiationsafety/issues/128)
* **eval:** compute run comparisons for the dashboard ([80481aa](https://github.com/eikrad/Radiationsafety/commit/80481aaf11fb1c8a41a167d92f3eaae38391dae8))
* **eval:** dashboard charts by DemBot's chart rules; BM25 + RRF measured and rejected ([0f2e9f1](https://github.com/eikrad/Radiationsafety/commit/0f2e9f172a67cb9a23d40b9d067dc9985bf77a84))
* **eval:** dashboard for scoring v2 ([a43b153](https://github.com/eikrad/Radiationsafety/commit/a43b153965a1b980f663229c28a5d3f0c278b407))
* **eval:** deterministic evidence metrics and error attribution ([b02e6e0](https://github.com/eikrad/Radiationsafety/commit/b02e6e08eae7fb9f914943821113e8050afd21cd))
* **eval:** draw dashboard trend charts by DemBot's metric-chart rules ([ef3dc04](https://github.com/eikrad/Radiationsafety/commit/ef3dc04e968f67a58c6fd77e7c6fe31b7f122cbb))
* **eval:** golden item format v2 with nuggets and evidence ([72f5d5d](https://github.com/eikrad/Radiationsafety/commit/72f5d5d438569c8a59dbb7abec5c42ef2d69fc46))
* **eval:** golden set v2 and baseline E0 (retrieval plan, phase 2) ([85dbb5a](https://github.com/eikrad/Radiationsafety/commit/85dbb5a2a5c7a99131401604e36bff88aaf30bce))
* **eval:** golden set v2 draft, 15 new questions for review ([624989a](https://github.com/eikrad/Radiationsafety/commit/624989a2111dc224f9c2441af99a5e3b21be83e0))
* **eval:** independent nugget and groundedness judge ([2039ee9](https://github.com/eikrad/Radiationsafety/commit/2039ee9df68014dc8546d77967dac1ae90cfafc0))
* **eval:** judge calibration fixtures and judge_check ([3eed61c](https://github.com/eikrad/Radiationsafety/commit/3eed61cbe5bf20d1128e7cc174ab7cb9aae25157))
* **eval:** k test, answers from the top 3 vs the top 5 chunks per collection ([de6655c](https://github.com/eikrad/Radiationsafety/commit/de6655c178a9e5a034f9e8f617f9398385a94773))
* **eval:** keep the judge verdict in the report ([974d6e7](https://github.com/eikrad/Radiationsafety/commit/974d6e7cae24d15a49b203362c2bd52a8e4fb6d7))
* **eval:** local HTML dashboard for comparing eval runs ([421817b](https://github.com/eikrad/Radiationsafety/commit/421817bd27aec0d2c8134e43449651cd585bb2a8))
* **eval:** majority vote on groundedness ([3dd848d](https://github.com/eikrad/Radiationsafety/commit/3dd848d43b922bb88d8cb256b88e65d69a692be4))
* **eval:** position test, evidence first vs in the middle of the context ([#132](https://github.com/eikrad/Radiationsafety/issues/132)) ([fa9ba82](https://github.com/eikrad/Radiationsafety/commit/fa9ba82e9dd61c559e778010db2c3b674fff9529))
* **eval:** position test, evidence first vs in the middle of the context ([#132](https://github.com/eikrad/Radiationsafety/issues/132)) ([67de37e](https://github.com/eikrad/Radiationsafety/commit/67de37ea47deac6ea6d7420c6c002c0e2af84281))
* **eval:** rank-based retrieval eval with guards (retrieval plan, phase 1) ([bd3e998](https://github.com/eikrad/Radiationsafety/commit/bd3e998617df32373249bf0e8c97990f0a28a7a5))
* **eval:** rank-based retrieval metrics and evidence position ([ec0710d](https://github.com/eikrad/Radiationsafety/commit/ec0710d4e5308949f22b028d8e74d7c33d3fc646)), closes [#128](https://github.com/eikrad/Radiationsafety/issues/128)
* **eval:** record eval runs in a versioned history file ([3bca7f9](https://github.com/eikrad/Radiationsafety/commit/3bca7f95861c2e88cf60fd7f2be00e8ed8f13326))
* **eval:** rescore saved runs without re-running the graph ([2a01059](https://github.com/eikrad/Radiationsafety/commit/2a010598b163b300ace5e1b45919f9b914477d96))
* **eval:** retrieval-only runs for comparing embeddings ([e11014f](https://github.com/eikrad/Radiationsafety/commit/e11014f3fc1d4aad8854acceaea61b3712ccb5b7))
* **eval:** retrieval-only runs rank the evidence 20 chunks deep ([fcd122f](https://github.com/eikrad/Radiationsafety/commit/fcd122f0461f44fc99ac792cb39b8afe2da301ae))
* **eval:** run history and local comparison dashboard ([a17bf10](https://github.com/eikrad/Radiationsafety/commit/a17bf1089f5c8c41fd005c261bf30112becd2c75))
* **eval:** scoring v2 — evidence, nuggets, independent judge ([ad875a3](https://github.com/eikrad/Radiationsafety/commit/ad875a34d0f185156fb5b359ff871b82b6727443))
* **eval:** scoring v2 in run_eval, golden set migrated ([50037ed](https://github.com/eikrad/Radiationsafety/commit/50037eda47b3fb85cbaaa79542ddddafc8d2351c))
* **eval:** sign test on flipped questions in run comparisons ([fb5f32a](https://github.com/eikrad/Radiationsafety/commit/fb5f32a7ff0b777a25dc336a3aa6ffb63c4827ef))
* **eval:** write run metadata and history from run_eval ([4e55b87](https://github.com/eikrad/Radiationsafety/commit/4e55b873ce5a221302a09b66b95ef5566ae1f1bf))
* expose privacy_mode in API QueryResponse (Phase 1) ([ca4bdf3](https://github.com/eikrad/Radiationsafety/commit/ca4bdf389e7984a8889798d8525ff9c92e7fec1f))
* **frontend:** mark the page once /api/config has been applied ([7ef892c](https://github.com/eikrad/Radiationsafety/commit/7ef892c69722db8b1130389b05ecce849b9f4b5a))
* **frontend:** per-provider settings cards with active marker and status ([a395007](https://github.com/eikrad/Radiationsafety/commit/a3950073d6c0ebd053d4a99d92637ba4c4f29eed))
* **frontend:** Scaleway as default provider ([554b6aa](https://github.com/eikrad/Radiationsafety/commit/554b6aa316c9df852ca29271eca8ff1861ba78f5))
* **frontend:** show the question and progress while an answer is on its way ([76c4a69](https://github.com/eikrad/Radiationsafety/commit/76c4a6920a75ab350ffbf2a6932fb26b4343ee7b))
* **frontend:** show the question and progress while an answer is on its way ([093a052](https://github.com/eikrad/Radiationsafety/commit/093a052a2e98960e55d89c1a3166168888529af5))
* **ingestion:** chunk Danish law along its paragraphs, items and annex rows ([e86a572](https://github.com/eikrad/Radiationsafety/commit/e86a57268c808ebdb4a447652b2a2597bad36755))
* **ingestion:** chunk Danish law along its paragraphs, items and annex rows ([5926940](https://github.com/eikrad/Radiationsafety/commit/59269403ada11f53fc3176a834449a98df7bd7ab)), closes [#130](https://github.com/eikrad/Radiationsafety/issues/130)
* **ingestion:** one Chroma collection pair per embedding model ([a9984ca](https://github.com/eikrad/Radiationsafety/commit/a9984ca522d06833665e8ff26d48ddb8fa081c16))
* integrate privacy mode across app + header badge (Phase 5 - Final) ([81a84d9](https://github.com/eikrad/Radiationsafety/commit/81a84d974369629755b5c083d5d72ddc7b5b51c0))
* **llm:** Scaleway Generative APIs as provider ([0aa59e9](https://github.com/eikrad/Radiationsafety/commit/0aa59e9b91cc12470143ffb5803f927ed45e4b23))
* **retrieval:** fuse dense retrieval with BM25 by reciprocal rank, behind HYBRID_RETRIEVAL ([03e9d51](https://github.com/eikrad/Radiationsafety/commit/03e9d51f9679813e49720b1bd7c78850ff281423)), closes [#131](https://github.com/eikrad/Radiationsafety/issues/131)
* **retrieval:** k test, and 5 chunks per collection by default ([1ebf6d2](https://github.com/eikrad/Radiationsafety/commit/1ebf6d2b098af25902ce98d582db8e4c3cc72f6a))
* **retrieval:** retrieve 5 chunks per collection by default ([0e38741](https://github.com/eikrad/Radiationsafety/commit/0e387414006334f313aa0f414454e3e3b3cd0b49))
* Scaleway as default for answers and embeddings ([0692128](https://github.com/eikrad/Radiationsafety/commit/0692128784d9953ac66361aece96c649685b13dc))


### Bug Fixes

* add signal handler for graceful Ctrl+C during ingestion ([b82476d](https://github.com/eikrad/Radiationsafety/commit/b82476d845827e2eb995855eec654c1887945163))
* **api:** explain missing provider configuration instead of a 500 ([1c2b415](https://github.com/eikrad/Radiationsafety/commit/1c2b415ab32250fa587d99c5c6bcf015f3f23cf1))
* auto-switch model to ollama when privacy mode is enabled ([d9067a4](https://github.com/eikrad/Radiationsafety/commit/d9067a4bf1259ae5cc223a4266d2a0d9a0a45285))
* batch Ollama embeddings to prevent connection reset ([3bcd6d5](https://github.com/eikrad/Radiationsafety/commit/3bcd6d50425c9636cb5784b0aedbda4c318d3df3))
* **docker:** install the backend from uv.lock instead of resolving with pip ([9fc7cb3](https://github.com/eikrad/Radiationsafety/commit/9fc7cb384ee70e49671ee21f1fe9d72cf4e6a46e))
* **docker:** install the backend from uv.lock instead of resolving with pip ([93dd4f2](https://github.com/eikrad/Radiationsafety/commit/93dd4f278d6d367559984f299699c2af656f78dc))
* **docker:** use the project's .chroma index instead of an empty volume ([2455d6c](https://github.com/eikrad/Radiationsafety/commit/2455d6c8e6aab076b7ffcb4a260bb364bd9213de))
* **docker:** use the project's .chroma index instead of an empty volume ([f344408](https://github.com/eikrad/Radiationsafety/commit/f3444085bd59640942889396fd9d28c6266eb29e))
* **e2e:** register waitForResponse before click to avoid race condition ([78bb9ff](https://github.com/eikrad/Radiationsafety/commit/78bb9ff3247c98c41b27fd50fefe052c4229bb9b))
* Enter submits query, graceful JSON-parse error handling ([316dd5a](https://github.com/eikrad/Radiationsafety/commit/316dd5a468de24871d314722f73967bca1761507))
* **eval:** correct four judge fixtures that the first calibration run exposed ([9c82636](https://github.com/eikrad/Radiationsafety/commit/9c826361c74b1970eeb07e7d7e77da854d5d2118))
* **eval:** count warnings shown with answers ([50a913f](https://github.com/eikrad/Radiationsafety/commit/50a913f81c2b24979cb4b7133e7781810e7f56b9))
* **eval:** ignore invisible format characters in evidence matching ([8f0d816](https://github.com/eikrad/Radiationsafety/commit/8f0d81676854f2d13c517af508ff0021bb5cad7c))
* **eval:** ingestion's data updates do not block recording a run ([1cd2346](https://github.com/eikrad/Radiationsafety/commit/1cd23466072dda36743abcca1de3ffe8f5234e6b))
* **eval:** load .env explicitly; real Scaleway model ids in examples ([06548e6](https://github.com/eikrad/Radiationsafety/commit/06548e6b6b49eafcfceb5c6aff3dc5f12f5e547c))
* **eval:** runs say which chunks they used and refuse uncommitted code ([8c2484e](https://github.com/eikrad/Radiationsafety/commit/8c2484eb87e80dc783bddda4b90a4e8a29cf7935))
* **eval:** show progress while a full run answers and judges ([e619368](https://github.com/eikrad/Radiationsafety/commit/e619368b6a034b68bcf97f23b6cf615087725bc2))
* **eval:** skip the top-k self-check when retrieving shallower than k ([96d285b](https://github.com/eikrad/Radiationsafety/commit/96d285bc9827a857ff2436f9a0d32472a87ab08d))
* **eval:** use datetime.UTC in the position test (ruff UP017) ([e0df898](https://github.com/eikrad/Radiationsafety/commit/e0df8987d62595b2caab61b588649e21b37931e7))
* filter complex metadata from DoclingLoader for Chroma compatibility ([9bb543c](https://github.com/eikrad/Radiationsafety/commit/9bb543cb8da0dda8be6692d827658f4e85dddd45))
* **frontend:** provider-specific error messages ([0ef2970](https://github.com/eikrad/Radiationsafety/commit/0ef29704e4062048addd708efd23dc3edd35d191))
* **frontend:** readable model selector and settings in light and dark mode ([59df078](https://github.com/eikrad/Radiationsafety/commit/59df0783e2049a1c140054b5a828d4fa3fd266f6))
* **frontend:** remove API keys left in localStorage by earlier versions ([875f8f9](https://github.com/eikrad/Radiationsafety/commit/875f8f9d1b96abc8359e1fbcfefd5ac4277e9404))
* **frontend:** type-check in build and fix the errors it found ([119ca0e](https://github.com/eikrad/Radiationsafety/commit/119ca0e19e195331e10136d1a917d60af638432c))
* give up on stalled provider calls instead of a 504 after 60 s ([ce4f39f](https://github.com/eikrad/Radiationsafety/commit/ce4f39fff2e8478b07fae8f85ced3323e514da62))
* give up on stalled provider calls instead of a 504 after 60 s ([08ea845](https://github.com/eikrad/Radiationsafety/commit/08ea845f05e056e62d7ee660c1f1a67a5b77841a))
* **graph:** ask once more for a missing verdict, then fall back to the cautious one ([ed33339](https://github.com/eikrad/Radiationsafety/commit/ed33339ab28017dae257ca98d9f0a525cc960093))
* **graph:** graders read whole chunks (retrieval plan, phase 3a) ([ad0cf8d](https://github.com/eikrad/Radiationsafety/commit/ad0cf8d09614e814f5412c9b5a4a2611b7ebcf2b))
* **graph:** graders read whole chunks; verify_trusted trusts a passed grading ([2d33f5c](https://github.com/eikrad/Radiationsafety/commit/2d33f5c403c25641889bec77cae303f4553ce7bc))
* **graph:** read a grader's free-text field given as an object or list ([1ffa18b](https://github.com/eikrad/Radiationsafety/commit/1ffa18b7c6ba7ce29ba76a18e5b93e00f3659451))
* **graph:** show the sufficiency grader's example reply as JSON ([d694b62](https://github.com/eikrad/Radiationsafety/commit/d694b6221f5c87f517f992b1850969020f3e3f9e))
* **ingestion:** back up a document only when a new version replaces it ([43ae50c](https://github.com/eikrad/Radiationsafety/commit/43ae50ce58ca35e2079ac211795c53eae4fac396))
* **ingestion:** keep a Danish law file when only its XML layout changed ([8c5da3d](https://github.com/eikrad/Radiationsafety/commit/8c5da3d30042c7caf06eb0bf55ceaa420e2699cc))
* **ingestion:** keep exponents in Danish law text and drop the XML metadata ([e029318](https://github.com/eikrad/Radiationsafety/commit/e029318384876a78fc9b177aa584497a6849ffd8))
* **ingestion:** one copy per Danish law, readable exponents (retrieval plan, step 1) ([58f0df5](https://github.com/eikrad/Radiationsafety/commit/58f0df5dbf50e22854b421cf70bb974136a4a186))
* **ingestion:** one copy per Danish law; rebuild the Danish collection alone ([7e66334](https://github.com/eikrad/Radiationsafety/commit/7e663349b60911b04bc48c5650bb25679a500e62))
* **ingestion:** re-embed into EMBEDDING_PROVIDER, never the answer model's ([65a1ab3](https://github.com/eikrad/Radiationsafety/commit/65a1ab34aafcc2e5aa9957e140da5b7d8b491211))
* **llm:** read Scaleway structured replies written as text ([95c9e1d](https://github.com/eikrad/Radiationsafety/commit/95c9e1d726b6067afaa8317ba3c201771f673e6e))
* **llm:** request structured output from Scaleway via tool calling ([21f2bbc](https://github.com/eikrad/Radiationsafety/commit/21f2bbcb09d10eb631bb5eed1a82497183628c6e))
* narrow Ollama-down detection to avoid false positives ([30afadd](https://github.com/eikrad/Radiationsafety/commit/30afaddbaeda74c1cd81578eca9dd7de384f8789))
* **ollama:** clear error messages for model-not-found vs not-running ([9f91e2a](https://github.com/eikrad/Radiationsafety/commit/9f91e2a7d6f54518210cb520ff92ce47ba5e7e1b))
* Privacy Mode — visibility, model enforcement, Ollama error handling & UX ([7ba1872](https://github.com/eikrad/Radiationsafety/commit/7ba187229aa76775fc68bb514c9657523e40fb6c))
* resolve ESLint errors failing CI on master ([781b7c9](https://github.com/eikrad/Radiationsafety/commit/781b7c9e8b95e800515c46ea4ec5420362e845d5))
* resolve ESLint errors failing CI on master ([a5a4e3b](https://github.com/eikrad/Radiationsafety/commit/a5a4e3b95b222338d52d53d0b2c017c816a208ab))
* use HybridChunker with token-aware splitting for nomic-embed-text ([cfdec4c](https://github.com/eikrad/Radiationsafety/commit/cfdec4cd0efb8e8f565bcd34d562cae8c03ca2e9))


### Security

* bump cryptography 49.0.0 -&gt; 50.0.1 (major version) ([1cc00a0](https://github.com/eikrad/Radiationsafety/commit/1cc00a0a8eb60d99255eadec45040b1a7d4d4aba))


### Reverts

* **retrieval:** remove BM25 + RRF after it lost on every adoption condition ([6a5c5b4](https://github.com/eikrad/Radiationsafety/commit/6a5c5b4d00d564a7e9f310afd389fd9ed3f6f794)), closes [#131](https://github.com/eikrad/Radiationsafety/issues/131)

## 0.5.0 upgrade notes

Read these before upgrading from 0.4.x. The generated list of features and fixes
for 0.5.0 is above.

### Changed
- **⚠️ Breaking: Scaleway is the default provider** for answers (`gemma-4-26b-a4b-it`) and
  retrieval embeddings (`bge-multilingual-gemma2`), EU-hosted, instead of Gemini. On the golden
  set, BGE embeddings found more of the relevant passages than Gemini (evidence recall 0.90 vs
  0.81, pass rate 79 % vs 71 %). A setup without `LLM_PROVIDER` / `EMBEDDING_PROVIDER` now needs
  `SCW_SECRET_KEY` and the Scaleway collections
  (`EMBEDDING_PROVIDER=scaleway uv run python ingestion.py --reembed-from gemini`); set both to
  `gemini` to keep the previous behaviour. Gemini, OpenAI, Mistral and Ollama stay available.
- The UI offers Scaleway first, with its models from the server's `.env`, and a Scaleway key
  field in Settings. The privacy notice no longer says every question goes to Google.

### Privacy and compliance (#94)
- **Fixed a Privacy Mode leak:** with `LLM_PROVIDER=ollama`, the generation-retry path
  could still fall back to Brave web search after two failed retries. It now ends
  instead, and `web_search` refuses to run in Privacy Mode as a second guard.
- LangSmith tracing is off by default; when enabled, the EU endpoint is the documented
  default.
- The header shows a persistent AI disclosure and a not-legal/clinical-advice notice
  (EU AI Act Art. 50(1)); a new Privacy notice names the controller from the optional
  `PRIVACY_CONTROLLER_NAME` / `PRIVACY_CONTROLLER_CONTACT` variables.
- `X-Forwarded-For` is trusted only with `TRUST_PROXY_HEADERS=true`; before, any client
  could spoof it to get around rate limits.
- API keys entered in the browser live in `sessionStorage` instead of `localStorage`.
- Three Danish bekendtgørelser in force since 1 January 2026 (BEK 1386–1388) are staged;
  run `uv run python ingestion.py` to embed them.
- The in-memory rate-limit store drops entries untouched for over an hour instead of
  growing for the life of the process.
  
### Setup checks (#115)
- Settings show per provider whether the server can answer with it and, if not, what is
  missing (e.g. `SCW_MODEL`, `SCW_EMBED_MODEL`, an unbuilt search index). A misconfigured
  provider now returns a 503 with that reason instead of a bare 500.
- **Check your `.env` after upgrading:** with `EMBEDDING_PROVIDER` unset, every cloud
  provider searches with Scaleway embeddings. Set `EMBEDDING_PROVIDER=gemini` to keep
  using an existing Gemini index.

### Security
- **`cryptography` 49.0.0 → 50.0.1 (⚠️ major version bump)** — fixes
  [GHSA-g6cj-pr64-35w5](https://github.com/advisories/GHSA-g6cj-pr64-35w5)
  (PYSEC-2026-3552, CVE-2026-69247): `pkcs7_decrypt_der`/`pkcs7_decrypt_pem`/
  `pkcs7_decrypt_smime` leaked a Bleichenbacher timing/output oracle against
  the recovered content-encryption key (introduced in 44.0.0, fixed in 50.0.0).
  `cryptography` is a transitive dependency (via `google-auth` →
  `google-genai` → `langchain-google-genai`), not pinned directly in
  `pyproject.toml`; bumped via `uv lock --upgrade-package cryptography`.
  Verified with `pip-audit` (CVE no longer reported) and the full test/lint
  suite (177 backend tests, ruff, black, isort — all green; no code changes
  required, `uv.lock` only).

### Notes (routine weekly maintenance, no code changes otherwise)
- `pip-audit` against the resolved environment also flagged `transformers`
  5.8.1 (CVE-2026-9856, path-traversal in `save_pretrained`, fixed in
  5.10.0) and `accelerate`/`chromadb` (no fix version published yet).
  `transformers` could not be bumped: `docling-core`/`docling-ibm-models`
  cap it at `<5.9.0` on `sys_platform == "darwin"`, and `uv.lock` is a
  cross-platform lock, so `uv lock --upgrade-package transformers` cannot
  select 5.10.0 without breaking macOS installs. Left at 5.8.1 pending an
  upstream `docling` release that relaxes the darwin cap; tracked for a
  future maintenance pass rather than forced here.

### Retrieval (#137–#144)
- **Re-ingest the Danish law collection** after upgrading:
  `uv run python ingestion.py --dk-only`. The Retsinformation XML is now chunked along
  its paragraphs, items and annex rows, each chunk headed `law › chapter › §` (#141),
  and a PDF copy of a law that is already read from XML is no longer ingested (#140).
- **`RETRIEVER_K`** (new, default 5; was a fixed 3) sets the chunks retrieved per
  collection. On the golden set, k = 5 fixed the three questions whose evidence ranked
  4–5 and lost nothing elsewhere (full run 32/39 vs 31/39). It sends about two thirds
  more context per call; `RETRIEVER_K=3` restores the previous behaviour.
- The graders read whole chunks instead of their first 420 or 1200 characters (#139),
  so answers built only on local sources no longer get a false "could not be fully
  verified" warning.

## 0.4.0 - 2026-06-11

### Added
- **Privacy Mode**: fully air-gapped operation via Ollama for both LLM generation and embeddings.
  - No cloud API calls, LangSmith tracing, or web search when `LLM_PROVIDER=ollama`
  - Separate Chroma collections (`-ollama` suffix) to preserve cloud indexes
  - Automatic privacy guards: web search and tracing disabled in privacy mode
  - Local embedding models: `nomic-embed-text` (default)
  - Local LLM models: `llama3.1:8b` (default)
  - Configurable via `.env`: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`
- **Test parallelization**: pytest-xdist for parallel test execution (`-n auto` in CI)
- Updated all dependency versions to currently installed stable versions for consistency

### Changed
- CI now runs tests in parallel on available CPU cores for faster feedback
- All main and dev dependencies pinned to stable versions (e.g., chromadb 1.5+, fastapi 0.136+, langchain 1.3+, pytest 9.0+)
- `grade_documents` node now respects `privacy_mode` flag to prevent web search in air-gapped mode

## 0.3.0 - 2026-05-18

### Added
- **Reflexion retry loop** (Shinn et al., NeurIPS 2023): when a generation fails
  grading, the grader now produces a short verbal hint (`missing_info`) describing
  the specific missing fact or document section. This hint is stored as `reflection`
  in graph state and passed to `retrieve_missing` on the next attempt, so the
  retrieval query targets the exact gap rather than blindly re-querying.
- New `GRADE_GENERATION` node (`graph/nodes/grade_generation.py`): extracts the
  LLM grading work from the old combined routing function into a proper LangGraph
  node so it can write `reflection` and `generation_passed_grading` to state.
- New state fields: `reflection: str`, `generation_passed_grading: bool`.
- `reflection` parameter on `invoke_missing_query_chain` — when non-empty, the
  hint is injected into the human prompt turn as focused context.
- Two new test files: `tests/test_grade_generation.py`, `tests/test_reflection.py`.

### Changed
- `GradeGeneration` schema simplified: `grounded: bool` + `answers_question: bool`
  collapsed into `passed: bool` + `missing_info: str`. Both old fields routed
  identically; merging them makes the grader prompt cleaner and less ambiguous.
- `grade_generation_grounded` routing function split into:
  - `GRADE_GENERATION` node (LLM call, writes state)
  - `route_after_grade_generation` (pure function, no LLM call, reads state flags)
- `generate` node now resets `reflection = ""` on every generation attempt so
  stale hints never leak into the next turn of a multi-turn conversation.
- `eval/metrics.py`: `faithfulness` and `answer_relevance` now both read `passed`
  (previously read `grounded` and `answers_question` respectively).
- Test count: 126 → 135.

## 0.2.0 - 2026-04-30

### Added
- Admin-token protection for mutating backend routes with fail-closed behavior.
- In-memory per-client rate limiting for query/admin endpoints with `Retry-After` on `429`.
- Optional Redis rate-limit backend for multi-replica deployments (`RATE_LIMIT_BACKEND=redis`).
- Request correlation via `X-Request-ID`.
- Expanded Prometheus-style metrics:
  - request totals and error totals,
  - duration sum,
  - per-endpoint request/error counters,
  - response status-class counters,
  - web-search attempt counter.
- Pre-commit hook configuration (`black --check`, `isort --check-only`) and CI alignment.

### Changed
- CI now validates pre-commit checks and frontend build before tests complete.
- Docker/Compose runtime hardening:
  - non-root backend runtime,
  - healthchecks and startup dependency on healthy backend,
  - reduced privileges/capabilities in Compose,
  - safer runtime defaults and graceful shutdown tuning.

### Notes
- Mypy gate is intentionally scoped in CI to changed critical files and can be expanded gradually.
