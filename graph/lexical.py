"""Lexical retrieval (BM25) and reciprocal rank fusion with dense retrieval (#131).

BM25 needs a real analyzer: a stemmer and stop words for the collection's
language. With a subword tokenizer instead, BM25 fell far below dense retrieval
in BGE-M3's comparison (Chen et al. 2024, appendix C.2); with Lucene's analyzer
it matched it on long documents. Fusion uses ranks only, with k = 60 untuned
(Cormack et al. 2009): BM25 and cosine scores are not on one scale, and tuned
weights would fit the 39 golden questions rather than the documents.
"""

import re
import threading
from functools import lru_cache

import snowballstemmer
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

RRF_K = 60

# The Snowball project's stop word lists (snowballstem.org).
_STOP_WORDS = {
    "danish": frozenset(
        """og i jeg det at en den til er som på de med han af for ikke der var mig
        sig men et har om vi min havde ham hun nu over da fra du ud sin dem os op
        man hans hvor eller hvad skal selv her alle vil blev kunne ind når være
        dog noget ville jo deres efter ned skulle denne end dette mit også under
        have dig anden hende mine alt meget sit sine vor mod disse hvis din nogle
        hos blive mange ad bliver hendes været thi jer sådan""".split()
    ),
    "english": frozenset(
        """i me my myself we our ours ourselves you your yours yourself yourselves
        he him his himself she her hers herself it its itself they them their
        theirs themselves what which who whom this that these those am is are was
        were be been being have has had having do does did doing would should
        could ought a an the and but if or because as until while of at by for
        with about against between into through during before after above below
        to from up down in out on off over under again further then once here
        there when where why how all any both each few more most other some such
        no nor not only own same so than too very""".split()
    ),
}

_TOKEN = re.compile(r"\w+")


@lru_cache(maxsize=None)
def _stemmer(language: str):
    return snowballstemmer.stemmer(language)


def analyze(text: str, language: str) -> list[str]:
    """Lower-cased word tokens without stop words, stemmed for the language.

    Numbers stay ("§ 3, nr. 54" → "3", "54"): paragraph and item numbers and
    document ids are exactly what lexical search is for.
    """
    stop = _STOP_WORDS[language]
    words = [w for w in _TOKEN.findall(text.lower()) if w not in stop]
    return _stemmer(language).stemWords(words)


class LexicalIndex:
    """BM25 over one collection's chunks."""

    def __init__(self, documents: list[Document], language: str):
        self.documents = documents
        self.language = language
        corpus = [analyze(d.page_content, language) for d in documents]
        self._tokens = [set(tokens) for tokens in corpus]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, n: int) -> list[Document]:
        """The n best chunks that share at least one term with the query."""
        terms = analyze(query, self.language)
        if not terms or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(terms)
        wanted = set(terms)
        hits = [i for i, tokens in enumerate(self._tokens) if tokens & wanted]
        hits.sort(key=lambda i: -scores[i])
        return [self.documents[i] for i in hits[:n]]


def _key(doc: Document) -> str:
    return doc.id or f"{doc.metadata.get('source', '')}|{doc.page_content[:240]}"


def reciprocal_rank_fusion(
    rankings: list[list[Document]], k: int = RRF_K
) -> list[Document]:
    """One ranking from several: score(d) = Σ 1 / (k + rank of d in each).

    Each chunk appears once; ties keep the order of first appearance, so the
    first ranking (dense) decides between equally fused chunks.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = _key(doc)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    order = sorted(scores, key=lambda key: -scores[key])  # stable
    return [docs[key] for key in order]


# --- indexes of the Chroma collections ------------------------------------------

_indexes: dict[str, tuple[int, LexicalIndex]] = {}
_lock = threading.Lock()


def collection_index(collection_name: str, language: str) -> LexicalIndex:
    """BM25 index of a Chroma collection, rebuilt when its chunk count changes
    (an added PDF, a re-ingestion)."""
    import chromadb

    from ingestion import _CHROMA_DIR

    collection = chromadb.PersistentClient(path=str(_CHROMA_DIR)).get_collection(
        collection_name
    )
    count = collection.count()
    with _lock:
        cached = _indexes.get(collection_name)
        if cached is not None and cached[0] == count:
            return cached[1]
        stored = collection.get(include=["documents", "metadatas"])
        documents = [
            Document(id=id_, page_content=text or "", metadata=meta or {})
            for id_, text, meta in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        ]
        index = LexicalIndex(documents, language)
        _indexes[collection_name] = (count, index)
        return index


def clear_indexes() -> None:
    with _lock:
        _indexes.clear()
