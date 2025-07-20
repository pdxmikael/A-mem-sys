# A-MEM Fork Implementation Guide

## Overview
This project now uses a forked version of A-mem-sys (https://github.com/WujiangXu/A-mem-sys) with enhancements for Anthropic API support and memory deduplication.

## Required Fork URL
Update `pyproject.toml` line 37 with your actual fork URL:
```toml
"agentic-memory @ git+https://github.com/YOUR_GITHUB_USERNAME/A-mem-sys.git",
```

## Changes Needed in Fork

### 1. Anthropic API Support
The original A-mem-sys only supports OpenAI and Ollama. Our fork needs:

**File: `agentic_memory/llm_controller.py`**
- Add Anthropic client initialization
- Add Anthropic completion methods
- Handle Anthropic-specific response formats

**Reference Implementation:**
See `agentic_memory_backup_for_fork/llm_controller.py` for our working Anthropic integration.

### 2. Robust JSON Parsing
**File: `agentic_memory/memory_system.py`**
- Add `_extract_json_from_response()` method to handle markdown-wrapped JSON
- Update `process_memory()` to use robust JSON extraction

**Reference Implementation:**
See `agentic_memory_backup_for_fork/memory_system.py` lines 720-749 for JSON extraction logic.

### 3. Memory Deduplication (New Feature)
**File: `agentic_memory/memory_system.py`**

Add semantic deduplication in `add_note()` method:
```python
def add_note(self, content: str, time: str = None, **kwargs) -> str:
    # Check for similar existing memories before adding
    similarity_threshold = 0.90
    existing_similar = self._find_similar_memories(content, similarity_threshold)
    
    if existing_similar:
        # Merge or update existing memory instead of adding duplicate
        return self._merge_or_update_memory(existing_similar[0], content, **kwargs)
    
    # Continue with normal addition if no duplicates found
    # ... rest of existing logic
```

**New Methods to Add:**
- `_find_similar_memories(content, threshold)` - Find memories above similarity threshold
- `_merge_or_update_memory(existing_memory, new_content, **kwargs)` - Merge similar memories
- `_calculate_content_similarity(content1, content2)` - Calculate semantic similarity

### 4. Persistent Storage Support
**File: `agentic_memory/retrievers.py`**
- Ensure ChromaRetriever uses PersistentClient by default
- Support `persist_directory` parameter in constructor

**Reference Implementation:**
See `agentic_memory_backup_for_fork/retrievers.py` lines 17-32.

## Installation After Fork Creation
1. Fork https://github.com/WujiangXu/A-mem-sys to your GitHub account
2. Update `pyproject.toml` with your fork URL
3. Implement the changes listed above in your fork
4. Install with: `pip install -e .`

## Testing the Fork
After implementation, test with:
```python
from agentic_memory import AgenticMemorySystem

# Should work with Anthropic
memory_system = AgenticMemorySystem(
    llm_backend="anthropic",
    llm_model="claude-3-5-sonnet-20241022"
)

# Should deduplicate similar memories
memory_system.add_note("The temple creature is mysterious")
memory_system.add_note("The temple creature seems mysterious and ancient")
# Second memory should merge with first, not create duplicate
```

## Benefits of This Approach
- ✅ Clean separation between RPG project and memory system
- ✅ Ability to contribute improvements back to A-MEM community
- ✅ Easier to maintain and update
- ✅ Proper package management and dependencies
