"""Tests for Ollama Privacy Mode.

Spec: when LLM_PROVIDER=ollama, the system runs fully local —
no cloud API calls for LLM, embeddings, tracing, or web search.
Written from the behavioral spec, not the implementation.
"""

import re

import pytest

# ---------------------------------------------------------------------------
# 1. LLM factory: Ollama provider
# ---------------------------------------------------------------------------


def _assert_local_failure_says(failure: Exception, fix: str) -> None:
    """A failing local model makes the Answerer name the command that fixes it."""
    from graph.answering import Answerer, LocalModelUnavailable, ModelChoice

    def failing_model(**kwargs):
        raise failure

    with pytest.raises(LocalModelUnavailable, match=re.escape(fix)):
        Answerer(web_search_enabled=False, build_llm=failing_model).answer(
            "q", ModelChoice(provider="ollama")
        )


class TestOllamaLLM:
    """get_llm() with provider='ollama' should return a ChatOllama instance."""

    def test_returns_ollama_instance(self, monkeypatch):
        """Selecting ollama returns a ChatOllama model."""
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        from graph.llm_factory import get_llm

        llm = get_llm(provider="ollama")
        assert (
            "ollama" in type(llm).__module__.lower() or "Ollama" in type(llm).__name__
        )

    def test_no_api_key_required(self, monkeypatch):
        """Ollama should not raise APIKeyError — it needs no key."""
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        from graph.llm_factory import get_llm

        # Should NOT raise
        llm = get_llm(provider="ollama")
        assert llm is not None

    def test_uses_ollama_model_env(self, monkeypatch):
        """OLLAMA_MODEL env var controls which model is used."""
        monkeypatch.setenv("OLLAMA_MODEL", "mistral:7b")
        from graph.llm_factory import get_llm

        llm = get_llm(provider="ollama")
        assert getattr(llm, "model", None) == "mistral:7b"

    def test_default_model_fits_a_small_gpu(self, monkeypatch):
        """gemma4:e4b-it-qat runs fully in 6 GB of VRAM and beat qwen3.5:4b on
        tune by 16 questions net under Qwen and 7 under Mistral, at a median of
        50 s per question (ROADMAP, local model for privacy mode, stage 1)."""
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        from graph.llm_factory import get_llm

        llm = get_llm(provider="ollama")
        assert getattr(llm, "model", None) == "gemma4:e4b-it-qat"

    def test_a_missing_answer_model_names_the_one_to_pull(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        _assert_local_failure_says(
            Exception('model "gemma4:e4b-it-qat" not found (404)'),
            "ollama pull gemma4:e4b-it-qat",
        )

    def test_model_variant_overrides_env(self, monkeypatch):
        """model_variant parameter should take precedence over OLLAMA_MODEL env."""
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
        from graph.llm_factory import get_llm

        llm = get_llm(provider="ollama", model_variant="gemma2:9b")
        assert getattr(llm, "model", None) == "gemma2:9b"

    def test_uses_ollama_base_url_env(self, monkeypatch):
        """OLLAMA_BASE_URL env var controls the Ollama server URL."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:11434")
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        from graph.llm_factory import get_llm

        llm = get_llm(provider="ollama")
        base_url = getattr(llm, "base_url", None)
        assert base_url is not None
        assert "myhost" in str(base_url)

    def test_default_base_url_is_localhost(self, monkeypatch):
        """Without OLLAMA_BASE_URL, default should be localhost:11434."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        from graph.llm_factory import get_llm

        llm = get_llm(provider="ollama")
        base_url = str(getattr(llm, "base_url", ""))
        assert "localhost" in base_url or "127.0.0.1" in base_url

    def test_reads_the_whole_retrieved_context(self, monkeypatch):
        """Ollama's own 4096-token window cut the k=5 prompts (up to ~7200
        tokens) without a warning; the window defaults to 16384."""
        monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
        from graph.llm_factory import get_llm

        assert get_llm(provider="ollama").num_ctx == 16384

    def test_context_window_can_be_set_for_a_small_gpu(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
        from graph.llm_factory import get_llm

        assert get_llm(provider="ollama").num_ctx == 8192

    def test_ollama_in_allowed_providers(self):
        """'ollama' must be in ALLOWED_PROVIDERS."""
        from graph.llm_factory import ALLOWED_PROVIDERS

        assert "ollama" in ALLOWED_PROVIDERS


# ---------------------------------------------------------------------------
# 2. Embedding provider routing
# ---------------------------------------------------------------------------


class TestOllamaEmbeddingProvider:
    """When provider is ollama, embeddings should also be local."""

    def test_embedding_provider_returns_ollama(self):
        """get_embedding_provider('ollama') should return 'ollama', not 'gemini'."""
        from graph.llm_factory import get_embedding_provider

        assert get_embedding_provider("ollama") == "ollama"

    def test_embedding_provider_via_env(self, monkeypatch):
        """When LLM_PROVIDER=ollama in env, get_embedding_provider() returns 'ollama'."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        from graph.llm_factory import get_embedding_provider

        assert get_embedding_provider() == "ollama"

    def test_cloud_providers_use_the_chosen_cloud_embeddings(self, monkeypatch):
        """Cloud providers (gemini, openai, mistral) retrieve with EMBEDDING_PROVIDER, not locally."""
        from graph.llm_factory import get_embedding_provider

        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")

        assert get_embedding_provider("gemini") == "gemini"
        assert get_embedding_provider("openai") == "gemini"
        assert get_embedding_provider("mistral") == "gemini"

    def test_get_embeddings_returns_ollama_instance(self, monkeypatch):
        """get_embeddings('ollama') should return an OllamaEmbeddings instance."""
        monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        from graph.llm_factory import get_embeddings

        emb = get_embeddings("ollama")
        cls_name = type(emb).__name__
        mod_name = type(emb).__module__
        assert "ollama" in mod_name.lower() or "Ollama" in cls_name

    def test_embeddings_use_ollama_embed_model_env(self, monkeypatch):
        """OLLAMA_EMBED_MODEL env var controls the embedding model."""
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        from graph.llm_factory import get_embeddings

        emb = get_embeddings("ollama")
        assert getattr(emb, "model", None) == "mxbai-embed-large"

    def test_a_question_is_embedded_on_the_cpu_so_the_answer_model_keeps_the_gpu(
        self, monkeypatch
    ):
        """bge-m3 on the GPU pushed llama3.1:8b out of a 6 GB card, 4-7 s per
        switch; on the CPU a question takes 0.3 s and llama stays. Indexing
        many chunks still uses the GPU."""
        import ollama

        sent = []

        def fake_embed(self, model, input, **kwargs):
            sent.append((list(input), (kwargs.get("options") or {}).get("num_gpu")))
            return {"embeddings": [[0.0] for _ in input]}

        monkeypatch.setattr(ollama.Client, "embed", fake_embed)
        from graph.llm_factory import get_embeddings

        embeddings = get_embeddings("ollama")
        embeddings.embed_query("Hvad er dosisgrænsen?")
        embeddings.embed_documents(["§ 1. Bekendtgørelsen gælder ..."])

        assert sent == [
            (["Hvad er dosisgrænsen?"], 0),
            (["§ 1. Bekendtgørelsen gælder ..."], None),
        ]

    def test_default_embed_model_is_multilingual(self, monkeypatch):
        """bge-m3 (100+ languages) replaced the English-only nomic-embed-text:
        recall@5 0.40 -> 0.90, on Danish law 0.18 -> 0.89 (ROADMAP, step P1)."""
        monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        from graph.llm_factory import get_embeddings
        from graph.search_index import collection_names as get_collection_names

        emb = get_embeddings("ollama")
        assert getattr(emb, "model", None) == "bge-m3"
        assert get_collection_names("ollama") == (
            "radiation-iaea-ollama-bge-m3",
            "radiation-dk-law-ollama-bge-m3",
        )

    def test_a_missing_embedding_model_names_the_one_to_pull(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
        _assert_local_failure_says(
            Exception('model "bge-m3" not found (404)'), "ollama pull bge-m3"
        )


# ---------------------------------------------------------------------------
# 3. Chroma collection names: isolation from cloud
# ---------------------------------------------------------------------------


class TestOllamaCollections:
    """Ollama must use separate Chroma collections so cloud indexes are preserved."""

    def test_collection_names_have_ollama_suffix(self, monkeypatch):
        """nomic-embed-text, the earlier default, keeps its '-ollama' collections."""
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        from graph.search_index import collection_names as get_collection_names

        iaea, dk = get_collection_names("ollama")
        assert iaea == "radiation-iaea-ollama"
        assert dk == "radiation-dk-law-ollama"

    def test_another_local_embedding_model_gets_its_own_collections(self, monkeypatch):
        """Local embedders are built side by side and compared, as Scaleway's
        are; the default model keeps the collections privacy users have."""
        from graph.search_index import collection_names as get_collection_names

        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "bge-m3")
        assert get_collection_names("ollama") == (
            "radiation-iaea-ollama-bge-m3",
            "radiation-dk-law-ollama-bge-m3",
        )
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
        assert get_collection_names("ollama") == (
            "radiation-iaea-ollama",
            "radiation-dk-law-ollama",
        )

    def test_collection_names_dont_collide_with_gemini(self):
        """Ollama and Gemini must use different collection names."""
        from graph.search_index import collection_names as get_collection_names

        iaea_g, dk_g = get_collection_names("gemini")
        iaea_o, dk_o = get_collection_names("ollama")
        assert iaea_g != iaea_o
        assert dk_g != dk_o

    def test_collection_names_dont_collide_with_mistral(self):
        """Ollama and Mistral must use different collection names."""
        from graph.search_index import collection_names as get_collection_names

        iaea_m, dk_m = get_collection_names("mistral")
        iaea_o, dk_o = get_collection_names("ollama")
        assert iaea_m != iaea_o
        assert dk_m != dk_o


# ---------------------------------------------------------------------------
# 4. Embeddings readiness check
# ---------------------------------------------------------------------------


class TestOllamaReadinessCheck:
    """An unbuilt local index gives local-specific guidance."""

    def _message(self, tmp_path):
        from graph.i18n import index_not_ready_message
        from graph.search_index import ChromaIndex

        readiness = ChromaIndex("ollama", persist_dir=tmp_path / "empty_chroma").ready()
        assert not readiness
        return index_not_ready_message(readiness, "ollama", "en")

    def test_not_ready_message_mentions_ollama(self, tmp_path):
        """When ollama collections don't exist, the message should NOT mention GOOGLE_API_KEY."""
        msg = self._message(tmp_path)
        assert "GOOGLE_API_KEY" not in msg
        assert "ollama" in msg.lower() or "local" in msg.lower()

    def test_not_ready_message_tells_how_to_ingest(self, tmp_path):
        """Message should tell user how to run ingestion for ollama."""
        msg = self._message(tmp_path)
        assert "LLM_PROVIDER=ollama" in msg
        assert "ingestion" in msg.lower()


# ---------------------------------------------------------------------------
# 5. i18n: warning messages
# ---------------------------------------------------------------------------


class TestOllamaI18nWarnings:
    """Localized warning messages for missing Ollama embeddings."""

    def test_english_warning_mentions_local(self):
        from graph.i18n import get_warning_embeddings_not_built

        msg = get_warning_embeddings_not_built("ollama", "en")
        assert "local" in msg.lower() or "ollama" in msg.lower()
        assert "GOOGLE_API_KEY" not in msg

    def test_german_warning_exists(self):
        from graph.i18n import get_warning_embeddings_not_built

        msg = get_warning_embeddings_not_built("ollama", "de")
        # Should return a German string, not fall back to English "not built yet"
        assert "lokal" in msg.lower() or "ollama" in msg.lower()

    def test_danish_warning_exists(self):
        from graph.i18n import get_warning_embeddings_not_built

        msg = get_warning_embeddings_not_built("ollama", "da")
        assert "lokal" in msg.lower() or "ollama" in msg.lower()

    def test_gemini_warning_unchanged(self):
        """Existing gemini warning should not be affected."""
        from graph.i18n import get_warning_embeddings_not_built

        msg = get_warning_embeddings_not_built("gemini", "en")
        assert "GOOGLE_API_KEY" in msg


# ---------------------------------------------------------------------------
# 7. Frontend constants: Ollama in the model list
# ---------------------------------------------------------------------------


class TestFrontendConstants:
    """Verify that the frontend TypeScript constants include Ollama.

    These tests read the source files to verify the constants are set correctly,
    since we can't import TypeScript.
    """

    def test_models_array_includes_ollama(self):
        from pathlib import Path

        constants_path = (
            Path(__file__).resolve().parent.parent / "frontend" / "src" / "constants.ts"
        )
        content = constants_path.read_text()
        assert "'ollama'" in content or '"ollama"' in content

    def test_model_variants_has_ollama_entry(self):
        from pathlib import Path

        constants_path = (
            Path(__file__).resolve().parent.parent / "frontend" / "src" / "constants.ts"
        )
        content = constants_path.read_text()
        assert "ollama:" in content or "ollama :" in content

    def test_model_selector_has_ollama_label(self):
        from pathlib import Path

        # The dropdown's labels live in constants.PROVIDER_LABELS
        constants_path = (
            Path(__file__).resolve().parent.parent / "frontend" / "src" / "constants.ts"
        )
        content = constants_path.read_text()
        assert "ollama: 'Ollama (Local)'" in content

    def test_settings_modal_has_ollama_privacy_hint(self):
        from pathlib import Path

        modal_path = (
            Path(__file__).resolve().parent.parent
            / "frontend"
            / "src"
            / "components"
            / "SettingsModal.tsx"
        )
        content = modal_path.read_text()
        # Should mention privacy or local and not show API key input for ollama
        assert "Privacy" in content or "privacy" in content
        assert "ollama" in content.lower()


# ---------------------------------------------------------------------------
# 8. .env.example: Ollama vars documented
# ---------------------------------------------------------------------------


class TestEnvExample:
    """The .env.example file should document Ollama configuration."""

    def test_env_example_has_ollama_vars(self):
        from pathlib import Path

        env_path = Path(__file__).resolve().parent.parent / ".env.example"
        content = env_path.read_text()
        assert "OLLAMA_BASE_URL" in content
        assert "OLLAMA_MODEL" in content
        assert "OLLAMA_EMBED_MODEL" in content
        assert "OLLAMA_NUM_CTX" in content

    def test_env_example_mentions_privacy_mode(self):
        from pathlib import Path

        env_path = Path(__file__).resolve().parent.parent / ".env.example"
        content = env_path.read_text()
        assert "Privacy" in content or "privacy" in content
        assert "no data leaves" in content.lower() or "fully local" in content.lower()
