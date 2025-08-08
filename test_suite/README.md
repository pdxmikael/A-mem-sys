# Retrieval Test Suite

Evaluate retrieval ranking for narrative continuity. Computes Recall@1/3/5, MRR@5, NDCG@5, Spearman, Kendall, and a compound score with weights and thresholds.

## Layout
- `metrics.py`: metric implementations and compound scoring
- `runner.py`: loads dataset/tests, runs retrieval on `AgenticMemorySystem`/`ChromaRetriever`
- `examples/`: sample dataset and tests

## Dataset schema (JSON)
```json
{
  "documents": [
    {
      "id": "note_echo7",
      "content": "...",
      "keywords": ["Echo-7"],
      "context": "Polar",
      "tags": ["polar"]
    }
  ]
}
```
All fields except `content` are optional; they are passed through to `add_note`.

## Tests schema (JSON)
```json
{
  "k_eval": 5,
  "metric_weights": {
    "recall@1": 2.0,
    "recall@3": 1.0,
    "recall@5": 0.5,
    "mrr@5": 2.0,
    "ndcg@5": 2.0,
    "spearman": 1.0,
    "kendall": 1.0
  },
  "metric_thresholds": {
    "recall@1": 0.5,
    "ndcg@5": 0.6
  },
  "min_compound": 0.6,
  "cases": [
    {
      "query": "...",
      "primary_ids": ["note_echo7"],
      "gains": {"note_echo7": 3, "note_polar_secondary": 2},
      "desired_order": ["note_echo7", "note_polar_secondary"],
      "relevant_ids": ["note_echo7", "note_polar_secondary"],
      "metric_thresholds": {"recall@1": 1.0},
      "min_compound": 0.7
    }
  ]
}
```
- `gains` used for NDCG (graded relevance).
- If `relevant_ids` omitted, defaults to notes with positive gain.
- `metric_thresholds` and `min_compound` can be specified globally and/or per-case.

## Usage
Run with or without query shaping:
```bash
python -m test_suite --dataset test_suite/examples/dataset.json --tests test_suite/examples/tests.json --persist ./memory_db --model all-MiniLM-L6-v2
python -m test_suite --dataset test_suite/examples/dataset.json --tests test_suite/examples/tests.json --use_shaper
```
Outputs JSON blocks per case and a final summary with averages and pass/fail results.

## Notes
- The suite uses your existing `AgenticMemorySystem` and `ChromaRetriever` with `session_id` filtering.
- No changes are made to retrieval logic; this is purely evaluative.
