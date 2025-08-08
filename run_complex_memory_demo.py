import os
import json
from uuid import uuid4
from typing import Any, Dict, List

from agentic_memory.memory_system import AgenticMemorySystem

import dotenv
dotenv.load_dotenv()

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_memories(ms: AgenticMemorySystem, note: str = "Current Memories") -> None:
    print_header(note)
    for mid, m in ms.memories.items():
        print(json.dumps({
            "id": mid,
            "content": m.content,
            "context": m.context,
            "keywords": m.keywords,
            "tags": m.tags,
            "category": m.category,
            "timestamp": m.timestamp,
            "retrieval_count": m.retrieval_count,
            "links": m.links,
        }, ensure_ascii=False, indent=2))
    if not ms.memories:
        print("<no memories>")


def print_results(title: str, results: List[Dict[str, Any]]) -> None:
    print_header(title)
    if not results:
        print("<no results>")
        return
    for r in results:
        print(json.dumps(r, ensure_ascii=False, indent=2))


def main() -> None:
    # You can export OPENAI_API_KEY in your environment for best results.
    # This script will clean up all data created for this session at the end.
    session_id = f"demo-{uuid4()}"
    backend = os.getenv("DEFAULT_LLM_BACKEND", "openai")
    # Use the requested model alias/id for OpenAI Responses API
    model = os.getenv("DEFAULT_LLM_MODEL", "gpt-5-mini")

    print_header("Initializing AgenticMemorySystem")
    print(json.dumps({
        "session_id": session_id,
        "llm_backend": backend,
        "llm_model": model,
    }, indent=2))

    ms = AgenticMemorySystem(
        session_id=session_id,
        llm_backend=backend,
        llm_model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        persist_directory="./memory_db"
    )

    try:
        # Step 1: Add 5 memories, mixing explicit metadata and implicit (to trigger LLM analysis)
        print_header("Step 1: Adding 5 memories (mixed metadata)")

        m1 = ms.add_note("Buy milk and eggs tomorrow from the grocery store.")
        print(f"Added memory id: {m1}")
        print_memories(ms, "After adding m1")

        m2 = ms.add_note(
            "Finalize the project report and send to the team.",
            keywords=["project", "report", "deadline"],
            context="Work",
            tags=["todo", "priority"],
            category="Task",
        )
        print(f"Added memory id: {m2}")
        print_memories(ms, "After adding m2")

        m3 = ms.add_note(
            "Met with Sarah to discuss Q4 roadmap. Main points: user growth, infra scaling, analytics.")
        print(f"Added memory id: {m3}")
        print_memories(ms, "After adding m3")

        m4 = ms.add_note(
            "Experimented with Python async patterns and concurrency.",
            tags=["learning", "python"],
        )
        print(f"Added memory id: {m4}")
        print_memories(ms, "After adding m4")

        m5 = ms.add_note(
            "Family dinner on Saturday at 7pm. Make a reservation.")
        print(f"Added memory id: {m5}")
        print_memories(ms, "After adding m5")

        print_memories(ms, "After adding 5 memories")

        # Step 2: Regular semantic search (session-scoped)
        query1 = "tasks related to groceries and errands"
        res1 = ms.search(query1, k=5)
        print_results(f"Step 2: Regular search -> '{query1}'", res1)

        # Step 3: Agentic search (expands with related/linked context)
        query2 = "project planning and deadlines"
        res2 = ms.search_agentic(query2, k=5)
        print_results(f"Step 3: Agentic search -> '{query2}'", res2)

    finally:
        # Clean up the session data to avoid polluting persistent store
        try:
            print_header("Cleaning up session data")
            deleted = ms.delete_all_by_session(session_id)
            print(json.dumps({"session_id": session_id, "deleted": deleted}, indent=2))
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    main()
