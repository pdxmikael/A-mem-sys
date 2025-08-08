I'll add robust normalization and debug prints to show the raw and shaped query used by ChromaRetriever.search(). This will help verify what text is actually embedded and explain why distances worsened. I'll propose a minimal change inside 
agentic_memory/retrievers.py
.

Proposed edit:
retrievers.py
+25
-8
 147
Apply
{{ ... }}
    def search(self, query: str, k: int = 5, where: dict = None):
-        query_text = query
-        if getattr(self, "query_shaper", None) is not None:
-            try:
-                shaped = self.query_shaper.shape(query)
-                semantic_query = shaped.get("semantic_query") if isinstance(shaped, dict) else None
-                if semantic_query:
-                    query_text = str(semantic_query)
-            except Exception:
-                pass
+        raw_query = query
+        query_text = raw_query
+        if getattr(self, "query_shaper", None) is not None:
+            try:
+                shaped = self.query_shaper.shape(raw_query)
+                semantic_query = shaped.get("semantic_query") if isinstance(shaped, dict) else None
+
+                # Normalize to a clean text string (avoid list/dict string repr degrading embeddings)
+                normalized = None
+                if isinstance(semantic_query, str):
+                    normalized = semantic_query.strip()
+                elif isinstance(semantic_query, (list, tuple)):
+                    normalized = "; ".join(map(str, semantic_query))
+                elif isinstance(semantic_query, dict):
+                    normalized = semantic_query.get("text") or semantic_query.get("query")
+                    if not normalized:
+                        normalized = " ".join([
+                            str(v) for v in semantic_query.values() if isinstance(v, str)
+                        ])
+
+                if normalized:
+                    query_text = normalized
+
+                # Debug what we actually embed
+                print(f"[QueryShaper] raw: {raw_query}")
+                print(f"[QueryShaper] shaped: {query_text}")
+            except Exception as e:
+                print(f"[QueryShaper] shaping failed: {e}")
         query_params = {
             "query_texts": [query_text],
             "n_results": k
         }
{{ ... }}
Thought Process



















Changes Proposed
I added robust normalization and debug prints inside ChromaRetriever.search() to make the shaped query explicit and ensure we embed a clean string.

File: 
agentic_memory/retrievers.py
Method: ChromaRetriever.search()
What changed:
Normalize semantic_query to a clean text string:
If string: .strip().
If list/tuple: join with “; ”.
If dict: use .get("text") or .get("query"), else join string values.
Print debug lines to show exactly what is embedded:
[QueryShaper] raw: ...
[QueryShaper] shaped: ...
Log exceptions: [QueryShaper] shaping failed: ...
This prevents the previous behavior of embedding a Python str() of a list/dict (e.g., "['kw1', 'kw2']"), which likely degraded embeddings and explains why shaping increased distances in your latest run.

Why this matters
Your results show regressions: e.g., heist query top match worsened (0.4196 → 1.6326), arctic query also worsened (0.8603 → 1.6534).
Likely cause: shaped output wasn’t a clean sentence, but a list/dict string representation or overly generic text, hurting semantic match.
Fix: normalize shaped output into high-quality natural text before embedding and print both raw and shaped to verify.
How to Validate
Run the same demo:
python c:/projects/A-mem-sys/run_complex_memory_demo_shaping_compare.py
Inspect console (or your redirected 
results.txt
) for lines:
[QueryShaper] raw: <original roleplay query>
[QueryShaper] shaped: <normalized shaped string>
Check if the shaped string:
Is natural language (not ['kw1', 'kw2'] or a dict).
Retains critical, domain-specific nouns (e.g., Helios, Astronomer’s Guild, Echo-7, Black Ridge, Camp Borealis).
Compare distances again. Expectation: reduced degradation; potential improvements where shaping adds useful semantics.
If Shaped Output Is Still Too Generic
Concatenate instead of replace (next step if needed): set query_text = f"{raw_query}. Keywords: {top_k_keywords}" so the original intent is preserved.
Limit length: trimming overly long shaped strings can help embeddings focus.
Summary of Changes
Added shaped query normalization and debug prints in agentic_memory/retrievers.py::ChromaRetriever.search() to ensure we embed high-quality text and to make shaping effects observable. This should address the observed distance regressions by avoiding embedding list/dict representations and by giving you visibility into the actual embedded query. Run the demo and share the [QueryShaper] raw/shaped pairs if further tuning is needed.