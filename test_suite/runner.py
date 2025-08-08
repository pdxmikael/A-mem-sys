import argparse
import json
import os
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import ChromaRetriever
from .metrics import (
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    spearman_on_subset,
    kendall_on_subset,
    compound_score,
)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_system(session_id: str, persist_directory: str, use_query_shaper: bool, model_name: str) -> AgenticMemorySystem:
    ms = AgenticMemorySystem(
        session_id=session_id,
        model_name=model_name,
        persist_directory=persist_directory,
        llm_backend=os.getenv("DEFAULT_LLM_BACKEND", "openai"),
        llm_model=os.getenv("DEFAULT_LLM_MODEL", "gpt-5-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    # Swap retriever to toggle shaper
    ms.retriever = ChromaRetriever(
        collection_name="memories",
        model_name=model_name,
        persist_directory=persist_directory,
        use_query_shaper=use_query_shaper,
    )
    return ms


def ingest_dataset(ms: AgenticMemorySystem, dataset: Dict[str, Any]) -> None:
    docs: List[Dict[str, Any]] = dataset.get("documents", [])
    for d in docs:
        content = d["content"]
        kwargs = {k: v for k, v in d.items() if k != "content"}
        ms.add_note(content=content, **kwargs)


def ingest_raw_notes(ms: AgenticMemorySystem, notes: List[Dict[str, Any]]) -> None:
    """Ingest notes following add_note signature (content, optional time, and kwargs)."""
    for n in notes:
        if "content" not in n:
            continue
        content = n["content"]
        time = n.get("time")
        kwargs = {k: v for k, v in n.items() if k not in ("content", "time")}
        ms.add_note(content=content, time=time, **kwargs)


def run_query(ms: AgenticMemorySystem, query: str, k: int, where: Dict[str, Any]) -> List[str]:
    results = ms.retriever.search(query, k=k, where=where)
    ids = results.get("ids", [[]])
    return list(ids[0]) if ids and len(ids) > 0 else []


def evaluate_case(ranked_ids: List[str], case: Dict[str, Any], k_eval: int) -> Dict[str, float]:
    # Inputs from case
    primary_ids: List[str] = case.get("primary_ids", [])
    gains: Dict[str, float] = case.get("gains", {})
    desired_order: List[str] = case.get("desired_order", [])
    relevant_ids: List[str] = case.get("relevant_ids") or [mid for mid, g in gains.items() if g > 0]

    scores: Dict[str, float] = {}
    # Core
    scores["recall@1"] = recall_at_k(ranked_ids, relevant_ids, 1)
    scores["recall@3"] = recall_at_k(ranked_ids, relevant_ids, 3)
    scores["recall@5"] = recall_at_k(ranked_ids, relevant_ids, 5)
    scores["mrr@5"] = mrr_at_k(ranked_ids, primary_ids, 5)
    scores["ndcg@5"] = ndcg_at_k(ranked_ids, gains, 5)
    # Rank order checks on subset
    scores["spearman"] = spearman_on_subset(ranked_ids, desired_order)
    scores["kendall"] = kendall_on_subset(ranked_ids, desired_order)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation test suite")
    parser.add_argument("--raw", help="Path to combined raw notes+tests JSON")
    parser.add_argument("--dataset", help="Path to dataset JSON")
    parser.add_argument("--tests", help="Path to tests JSON")
    parser.add_argument("--persist", default="./memory_db", help="Chroma persist dir")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--use_shaper", action="store_true", help="Enable QueryShaper")
    parser.add_argument("--k", type=int, default=10, help="Max candidates to retrieve for metrics")
    args = parser.parse_args()

    # Validate input source
    if not args.raw and not (args.dataset and args.tests):
        parser.error("Provide --raw or both --dataset and --tests")

    session_id = f"test-suite-{uuid4()}"
    ms = build_system(session_id, args.persist, args.use_shaper, args.model)

    try:
        if args.raw:
            raw = load_json(args.raw)
            notes = raw.get("notes", [])
            ingest_raw_notes(ms, notes)
            tests = raw.get("tests", {})
        else:
            dataset = load_json(args.dataset)
            ingest_dataset(ms, dataset)
            tests = load_json(args.tests)

        test_cases: List[Dict[str, Any]] = tests.get("cases", [])
        metric_weights: Dict[str, float] = tests.get("metric_weights", {})
        global_thresholds: Dict[str, float] = tests.get("metric_thresholds", {})
        min_compound_default: float = float(tests.get("min_compound", 0.0))
        k_eval: int = tests.get("k_eval", 5)

        overall: List[Dict[str, Any]] = []

        for i, case in enumerate(test_cases, start=1):
            query_text: str = case["query"]
            where = {"session_id": ms.session_id}
            ranked_ids = run_query(ms, query_text, k=max(args.k, k_eval), where=where)

            scores = evaluate_case(ranked_ids, case, k_eval)
            comp = compound_score(scores, metric_weights)

            # Threshold checks (per-case overrides global)
            case_thresholds: Dict[str, float] = case.get("metric_thresholds", {})
            thresholds = {**global_thresholds, **case_thresholds}
            min_compound = float(case.get("min_compound", min_compound_default))

            passes: Dict[str, bool] = {}
            for m_name, t_val in thresholds.items():
                val = scores.get(m_name)
                if val is None:
                    continue
                passes[m_name] = (val >= float(t_val))
            passes["compound"] = (comp >= min_compound)
            case_pass = all(passes.values()) if passes else True

            record = {
                "case_index": i,
                "query": query_text,
                "ranked_ids": ranked_ids[:k_eval],
                "scores": scores,
                "compound": comp,
                "passes": passes,
                "pass": case_pass,
            }
            overall.append(record)
            print(json.dumps(record, ensure_ascii=False, indent=2))

        # Aggregate summary
        if overall:
            agg_scores: Dict[str, float] = {}
            for r in overall:
                for kname, val in r["scores"].items():
                    agg_scores.setdefault(kname, 0.0)
                    agg_scores[kname] += val
            for kname in list(agg_scores.keys()):
                agg_scores[kname] /= len(overall)
            comp_all = compound_score(agg_scores, metric_weights)
            # Overall pass uses global thresholds and min_compound_default
            overall_passes: Dict[str, bool] = {}
            for m_name, t_val in global_thresholds.items():
                avg_val = agg_scores.get(m_name)
                if avg_val is None:
                    continue
                overall_passes[m_name] = (avg_val >= float(t_val))
            overall_passes["compound"] = (comp_all >= min_compound_default)
            overall_pass = all(overall_passes.values()) if overall_passes else True

            summary = {
                "avg_scores": agg_scores,
                "compound_avg": comp_all,
                "cases": len(overall),
                "passes": overall_passes,
                "pass": overall_pass,
            }
            print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))

    finally:
        # Cleanup session to avoid polluting the persistent store
        try:
            ms.delete_all_by_session(session_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
