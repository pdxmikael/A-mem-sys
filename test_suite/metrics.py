from typing import Dict, List, Tuple
import math

# Utilities

def _safe_rank_map(ranked_ids: List[str]) -> Dict[str, int]:
    return {mid: i for i, mid in enumerate(ranked_ids, start=1)}  # 1-based ranks

# Core metrics

def recall_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = set(ranked_ids[:k])
    hit = sum(1 for rid in relevant_ids if rid in topk)
    return hit / float(len(relevant_ids))


def mrr_at_k(ranked_ids: List[str], primary_ids: List[str], k: int) -> float:
    if not primary_ids:
        return 0.0
    topk = ranked_ids[:k]
    best = None
    for pid in primary_ids:
        if pid in topk:
            r = topk.index(pid) + 1
            best = r if best is None else min(best, r)
    if best is None:
        return 0.0
    return 1.0 / float(best)


def ndcg_at_k(ranked_ids: List[str], gains: Dict[str, float], k: int) -> float:
    # DCG with standard log2 discount, using provided gains directly
    def dcg(seq: List[str]) -> float:
        val = 0.0
        for i, mid in enumerate(seq[:k], start=1):
            g = gains.get(mid, 0.0)
            if g > 0:
                val += g / math.log2(i + 1)
        return val

    dcg_val = dcg(ranked_ids)
    # Ideal order by gain desc, then any tie order
    ideal = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    ideal_ids = [m for m, _ in ideal]
    idcg_val = dcg(ideal_ids)
    if idcg_val <= 0:
        return 0.0
    return dcg_val / idcg_val


# Rank correlations on a subset where order matters

def spearman_on_subset(ranked_ids: List[str], desired_order: List[str]) -> float:
    if len(desired_order) < 2:
        return 0.0
    rank_map = _safe_rank_map(ranked_ids)
    # Assign missing items a large rank (after the list)
    miss_rank = len(ranked_ids) + 1
    sys_ranks = [rank_map.get(mid, miss_rank) for mid in desired_order]
    gold_ranks = list(range(1, len(desired_order) + 1))

    # Spearman rho = 1 - (6 * sum d_i^2) / (n(n^2-1))
    n = len(desired_order)
    d2 = sum((a - b) ** 2 for a, b in zip(sys_ranks, gold_ranks))
    denom = n * (n * n - 1)
    if denom == 0:
        return 0.0
    rho = 1.0 - (6.0 * d2) / float(denom)
    # Clamp numeric drift
    return max(-1.0, min(1.0, rho))


def kendall_on_subset(ranked_ids: List[str], desired_order: List[str]) -> float:
    # Simple Kendall's tau on the subset; missing placed at tail preserving relative order
    n = len(desired_order)
    if n < 2:
        return 0.0
    rank_map = _safe_rank_map(ranked_ids)
    miss_rank = len(ranked_ids) + 1
    sys_order = [(mid, rank_map.get(mid, miss_rank)) for mid in desired_order]
    # Sort by system rank to get system-induced order over the subset
    sys_sorted = [mid for mid, _ in sorted(sys_order, key=lambda x: x[1])]

    # Count concordant/discordant pairs compared to desired_order
    index = {mid: i for i, mid in enumerate(desired_order)}
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sys_sorted[i], sys_sorted[j]
            if index[a] < index[b]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    tau = (concordant - discordant) / float(total)
    return max(-1.0, min(1.0, tau))


# Compound scoring

def normalize_score(name: str, value: float) -> float:
    # Already 0..1: recall, mrr, ndcg
    if name in {"recall@1", "recall@3", "recall@5", "mrr@5", "ndcg@5"}:
        return max(0.0, min(1.0, value))
    # Map [-1,1] -> [0,1]
    if name in {"spearman", "kendall"}:
        return (value + 1.0) / 2.0
    return value


def compound_score(scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
    if not scores:
        return 0.0
    weights = weights or {}
    total_w = 0.0
    acc = 0.0
    for k, v in scores.items():
        w = float(weights.get(k, 1.0))
        acc += w * normalize_score(k, v)
        total_w += w
    return acc / total_w if total_w > 0 else 0.0
