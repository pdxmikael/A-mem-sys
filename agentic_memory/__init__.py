"""
Agentic Memory System (Refactor) - A novel agentic memory system for LLM agents.

This module provides an advanced memory system that can dynamically organize memories
in an agentic way, with features including:
- Dynamic memory organization based on Zettelkasten principles
- Intelligent indexing and linking of memories via ChromaDB
- Comprehensive note generation with structured attributes
- Interconnected knowledge networks
- Continuous memory evolution and refinement
- Agent-driven decision making for adaptive memory management
"""

# Public API exports
from .memory_system import AgenticMemorySystem, MemoryNote
from .llm_controller import LLMController, BaseLLMController
from .retrievers import ChromaRetriever

__version__ = "0.1.7"
__author__ = "Agentic Memory Development Team"
__all__ = [
    "AgenticMemorySystem",
    "MemoryNote",
    "LLMController",
    "BaseLLMController",
    "ChromaRetriever",
]
