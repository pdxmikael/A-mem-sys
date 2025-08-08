import re
from typing import List, Dict, Optional

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # Graceful fallback if spaCy isn't installed yet

try:
    import yake  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("YAKE is required for QueryShaper. Please add 'yake' to dependencies.") from e


class QueryShaper:
    """
    Local, non-LLM query shaping for story continuation.

    Extracts entities, keyphrases, simple actions, and unresolved-goal cues
    to build a short semantic pseudo-query and keyword list for hybrid retrieval.

    - Prefers spaCy (if model available) for NER/noun-chunks/verbs.
    - Falls back to keyphrase extraction + regex heuristics if spaCy model is missing.
    """

    ENT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "WORK_OF_ART"}

    UNRESOLVED_PATTERNS = [
        r"\bneed(s)? to\b",
        r"\bmust\b",
        r"\bplan(s|ned)? to\b",
        r"\bpromise(d)? to\b",
        r"\bstill (has not|hasn't|haven't|not)\b",
        r"\blooking for\b",
        r"\btry(ing)? to\b",
        r"\bsearch(ing)? for\b",
    ]

    def __init__(
        self,
        language: str = "en",
        yake_max_ngram: int = 3,
        yake_top_k: int = 20,
        spacy_model: str = "en_core_web_sm",
        use_spacy: bool = True,
    ) -> None:
        self.language = language
        self.kw_extractor = yake.KeywordExtractor(lan=language, n=yake_max_ngram, top=yake_top_k)
        self._nlp = None

        if use_spacy and spacy is not None:
            try:
                # Prefer small English model; if not available, fall back gracefully
                self._nlp = spacy.load(spacy_model)  # requires local model install
            except Exception:
                try:
                    # Minimal fallback: blank pipeline with sentencizer
                    self._nlp = spacy.blank(language)
                    if "sentencizer" not in self._nlp.pipe_names:
                        self._nlp.add_pipe("sentencizer")
                except Exception:
                    self._nlp = None

        # Pre-compile patterns
        self._unresolved_regex = [re.compile(p, flags=re.IGNORECASE) for p in self.UNRESOLVED_PATTERNS]

    def _last_k_sentences(self, text: str, k: int) -> str:
        if not text:
            return ""
        if self._nlp is not None:
            doc = self._nlp(text)
            sents = list(doc.sents)
            if not sents:
                return text
            return " ".join(s.text for s in sents[-k:])
        # Lightweight sentence split fallback
        parts = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(parts[-k:]) if parts else text

    def _unresolved(self, text: str) -> bool:
        return any(r.search(text) is not None for r in self._unresolved_regex)

    def shape(self, text: str, last_k_sents: int = 4) -> Dict[str, object]:
        """
        Turn a continuation prompt into structured retrieval aids.

        Returns dict with keys:
          - semantic_query: str
          - keywords: List[str]
          - sub_queries: List[str]
          - unresolved: bool
        """
        window_text = self._last_k_sentences(text, last_k_sents)

        entities: List[str] = []
        noun_chunks: List[str] = []
        actions: List[str] = []

        if self._nlp is not None:
            doc = self._nlp(window_text)
            try:
                entities = [ent.text for ent in getattr(doc, "ents", []) if getattr(ent, "label_", None) in self.ENT_LABELS]
            except Exception:
                entities = [ent.text for ent in getattr(doc, "ents", [])]
            try:
                noun_chunks = [nc.text for nc in getattr(doc, "noun_chunks", []) if 1 <= len(nc.text.split()) <= 5]
            except Exception:
                noun_chunks = []
            try:
                actions = [t.lemma_ for t in doc if t.pos_ == "VERB" and t.dep_ in {"ROOT", "xcomp", "ccomp", "advcl"}]
            except Exception:
                actions = []
        # Keyphrases via YAKE
        try:
            kw_pairs = self.kw_extractor.extract_keywords(window_text)
            keyphrases = [k for k, _ in kw_pairs]
        except Exception:
            keyphrases = []

        unresolved = self._unresolved(window_text)

        # Build semantic pseudo-query
        def _uniq(seq: List[str]) -> List[str]:
            seen = set()
            out: List[str] = []
            for s in seq:
                x = s.strip()
                if not x:
                    continue
                if x.lower() in seen:
                    continue
                seen.add(x.lower())
                out.append(x)
            return out

        entities = _uniq(entities)
        noun_chunks = _uniq(noun_chunks)
        actions = _uniq(actions)
        keyphrases = _uniq(keyphrases)

        bits: List[str] = []
        if entities:
            bits.append(f"entities: {', '.join(entities)}")
        if actions:
            bits.append(f"actions: {', '.join(actions)}")
        if unresolved:
            bits.append("status: unresolved goals present")
        semantic_query = " | ".join(bits) if bits else window_text

        # Keywords for BM25/sparse search
        keywords: List[str] = _uniq([*entities, *noun_chunks, *keyphrases])[:40]

        # Sub-queries for multi-query retrieval
        sub_queries: List[str] = (entities[:5] if entities else keywords[:5])

        return {
            "semantic_query": semantic_query,
            "keywords": keywords,
            "sub_queries": sub_queries,
            "unresolved": unresolved,
        }
