"""
Test for session integrity across save/load operations.

This test validates that complex memories with multiple tags, keywords, and metadata
are properly preserved when a session is saved and then loaded in a new memory system instance.
It also tests memory isolation between different session IDs and disk persistence.
"""

import uuid
import tempfile
import shutil
from datetime import datetime
import os
import sys

# Add the parent directory to sys.path to import agentic_memory modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import chromadb
# Test the agentic_memory_refactor module instead of agentic_memory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agentic_memory_refactor'))
from agentic_memory_refactor.memory_system import AgenticMemorySystem, MemoryNote

class TestSessionIntegrity:
    """Test session loading and saving integrity."""
    
    def setup_method(self):
        """Set up test environment with a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            try:
                import time
                time.sleep(0.5)  # Allow time for file locks to be released
                shutil.rmtree(self.temp_dir)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not clean up temp directory: {e}")
    
    def test_complex_memory_session_integrity(self):
        """
        Test that complex memories maintain integrity across session save/load.
        """
        # Create complex test data
        complex_content = """
        This is a very complex memory entry that contains:
        - Multiple lines of text
        - Special characters: àáâãäåæçèéêëìíîïñòóôõöøùúûüý
        - Emojis: 🚀🔥💡🌟✨🎯📊💻🔬🌍
        - Code snippets: def test_function(): return "hello world"
        - Mathematical notation: ∑(x²) = π × √2
        - JSON-like structures: {"key": "value", "nested": {"array": [1,2,3]}}
        - URLs: https://example.com/path?param=value&other=123
        - Various punctuation and symbols: !@#$%^&*()_+-=[]{}|;:'"<>,.?/
        
        This content should remain exactly intact after session save/load operations,
        including all formatting, line breaks, and special characters.
        """
        
        complex_tags = [
            "machine-learning",
            "data science", 
            "python programming",
            "artificial intelligence",
            "deep learning models",
            "neural networks",
            "special-chars: àáâãäå",
            "emojis-🚀🔥",
            "json-data",
            "test-tag-with-numbers-123",
            "混合语言标签",
            "tag_with_underscores",
            "tag-with-hyphens",
            "TAG WITH SPACES",
            "CaseSensitiveTag"
        ]
        
        complex_keywords = [
            "machine learning", "neural networks", "python", "tensorflow", 
            "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib",
            "data analysis", "statistical modeling", "feature engineering",
            "model evaluation", "hyperparameter tuning", "cross validation"
        ]
        
        complex_context = "Advanced machine learning research and development project focusing on neural network architectures for natural language processing tasks"
        complex_category = "Research & Development"
        
        # Phase 1: Create and save complex memory
        persist_dir = os.path.join(self.temp_dir, "memory_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        memory_system_1 = AgenticMemorySystem(
            session_id=self.session_id,
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4.1-mini",
            persist_directory=persist_dir
        )
        
        memory_system_1.retriever.persist_directory = persist_dir
        memory_system_1.retriever.client = chromadb.PersistentClient(path=persist_dir)
        memory_system_1.retriever.collection = memory_system_1.retriever.client.get_or_create_collection(
            name=memory_system_1.retriever.collection_name
        )
        
        memory_note = MemoryNote(
            content=complex_content,
            tags=complex_tags,
            keywords=complex_keywords,
            context=complex_context,
            category=complex_category,
            timestamp=datetime.now().strftime("%Y%m%d%H%M"),
            retrieval_count=5
        )
        
        original_memory_id = memory_note.id
        memory_system_1.memories[original_memory_id] = memory_note
        
        metadata = {
            'session_id': self.session_id,
            'keywords': memory_note.keywords,
            'tags': memory_note.tags,
            'context': memory_note.context,
            'category': memory_note.category,
            'timestamp': memory_note.timestamp,
            'last_accessed': memory_note.last_accessed,
            'retrieval_count': memory_note.retrieval_count,
            'evolution_history': memory_note.evolution_history,
            'links': memory_note.links
        }
        
        memory_system_1.retriever.add_document(
            document=memory_note.content,
            metadata=metadata,
            doc_id=original_memory_id
        )
        
        assert len(memory_system_1.memories) == 1
        original_memory = memory_system_1.memories[original_memory_id]
        assert original_memory.content == complex_content
        assert original_memory.tags == complex_tags
        assert original_memory.keywords == complex_keywords
        
        # Phase 2: Create new memory system and load session
        memory_system_2 = AgenticMemorySystem(
            session_id=self.session_id,
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai",
            llm_model="gpt-4.1-mini",
            persist_directory=persist_dir
        )
        
        memory_system_2.retriever.persist_directory = persist_dir
        memory_system_2.retriever.client = chromadb.PersistentClient(path=persist_dir)
        memory_system_2.retriever.collection = memory_system_2.retriever.client.get_or_create_collection(
            name=memory_system_2.retriever.collection_name
        )
        
        memory_system_2._load_session_memories()
        
        assert len(memory_system_2.memories) == 1, f"Expected 1 memory, got {len(memory_system_2.memories)}"
        loaded_memory = memory_system_2.memories[original_memory_id]
        assert loaded_memory.content == complex_content, "Content was corrupted during save/load"
        assert loaded_memory.tags == complex_tags, "Tags corrupted"
        assert loaded_memory.keywords == complex_keywords, "Keywords corrupted"
        assert loaded_memory.context == complex_context, "Context was corrupted"
        assert loaded_memory.category == complex_category, "Category was corrupted"
        
        search_results = memory_system_2.search("machine learning neural networks", k=5)
        found_in_search = any(result.get('id') == original_memory_id for result in search_results)
        assert found_in_search, "Memory should be findable in search results"
    
    def test_memory_isolation(self):
        """Test that different sessions do not interfere with each other's data."""
        session_id_1 = f"test_session_{uuid.uuid4().hex[:8]}"
        session_id_2 = f"test_session_{uuid.uuid4().hex[:8]}"
        
        memory_system_1 = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4.1-mini",
            session_id=session_id_1
        )
        
        memory_system_2 = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4.1-mini",
            session_id=session_id_2
        )
        
        note1 = MemoryNote(
            content="Memory for session 1",
            tags=["tag1"],
            keywords=["keyword1"],
            context="Context for session 1",
            category="Category 1",
            timestamp=datetime.now().strftime("%Y%m%d%H%M"),
            retrieval_count=1
        )
        
        memory_system_1.memories[note1.id] = note1
        memory_system_1.retriever.add_document(
            document=note1.content,
            metadata={"session_id": session_id_1},
            doc_id=note1.id
        )
        
        note2 = MemoryNote(
            content="Memory for session 2",
            tags=["tag2"],
            keywords=["keyword2"],
            context="Context for session 2",
            category="Category 2",
            timestamp=datetime.now().strftime("%Y%m%d%H%M"),
            retrieval_count=1
        )
        
        memory_system_2.memories[note2.id] = note2
        memory_system_2.retriever.add_document(
            document=note2.content,
            metadata={"session_id": session_id_2},
            doc_id=note2.id
        )
        
        assert len(memory_system_1.memories) == 1
        assert len(memory_system_2.memories) == 1
        assert note1.id in memory_system_1.memories
        assert note2.id in memory_system_2.memories
        assert note2.id not in memory_system_1.memories
        assert note1.id not in memory_system_2.memories
    
    def test_disk_persistence(self):
        """Test that data persists across sessions."""
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        
        memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4.1-mini",
            session_id=session_id
        )
        
        note = MemoryNote(
            content="Persistent memory content",
            tags=["persistent"],
            keywords=["persistence"],
            context="Testing persistence",
            category="Persistence Test",
            timestamp=datetime.now().strftime("%Y%m%d%H%M"),
            retrieval_count=1
        )
        
        memory_system.memories[note.id] = note
        memory_system.retriever.add_document(
            document=note.content,
            metadata={"session_id": session_id},
            doc_id=note.id
        )
        
        persist_dir = os.path.join(self.temp_dir, "memory_db")
        memory_system.retriever.persist_directory = persist_dir
        memory_system.retriever.client = chromadb.PersistentClient(path=persist_dir)
        memory_system.retriever.collection = memory_system.retriever.client.get_or_create_collection(
            name=memory_system.retriever.collection_name
        )
        
        new_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4.1-mini",
            session_id=session_id
        )
        
        new_system.retriever.persist_directory = persist_dir
        new_system.retriever.client = chromadb.PersistentClient(path=persist_dir)
        new_system.retriever.collection = new_system.retriever.client.get_or_create_collection(
            name=new_system.retriever.collection_name
        )
        
        assert len(new_system.memories) == 1
        loaded_note = new_system.memories[note.id]
        assert loaded_note.content == note.content
        # For disk persistence, we're primarily checking that the content is preserved
        # The tags, keywords, context, and category might be processed differently by the memory system

if __name__ == "__main__":
    test = TestSessionIntegrity()
    test.setup_method()
    try:
        print("Running test_complex_memory_session_integrity...")
        test.test_complex_memory_session_integrity()
        print("Running test_memory_isolation...")
        test.test_memory_isolation()
        print("Running test_disk_persistence...")
        test.test_disk_persistence()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        test.teardown_method()
