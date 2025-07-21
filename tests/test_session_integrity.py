"""
Test for session integrity across save/load operations.

This test validates that complex memories with multiple tags, keywords, and metadata
are properly preserved when a session is saved and then loaded in a new memory system instance.
"""
import pytest
import uuid
import tempfile
import shutil
from datetime import datetime
import os
import sys

# Add the parent directory to sys.path to import agentic_memory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote


class TestSessionIntegrity:
    """Test session loading and saving integrity."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        
    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            try:
                # On Windows, ChromaDB may lock files. Try cleanup with retry.
                import time
                time.sleep(0.5)  # Give ChromaDB time to release files
                shutil.rmtree(self.temp_dir)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not clean up temp directory: {e}")
                # Don't fail the test due to cleanup issues
    
    def test_complex_memory_session_integrity(self):
        """
        Test that complex memories maintain integrity across session save/load.
        
        Creates a memory with:
        - Long, complex content with special characters
        - Multiple tags including ones with spaces and special characters
        - Multiple keywords
        - Context information
        - Category classification
        - Custom metadata
        
        Then verifies all data is preserved when loaded in a new memory system.
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
            "混合语言标签",  # Mixed language tag
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
        print(f"Phase 1: Creating memory system with session ID: {self.session_id}")
        
        memory_system_1 = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4o-mini",
            session_id=self.session_id
        )
        
        # Override the persist directory to use our temp directory
        import chromadb
        persist_dir = os.path.join(self.temp_dir, "memory_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        memory_system_1.retriever.persist_directory = persist_dir
        memory_system_1.retriever.client = chromadb.PersistentClient(path=persist_dir)
        memory_system_1.retriever.collection = memory_system_1.retriever.client.get_or_create_collection(
            name=memory_system_1.retriever.collection_name
        )
        
        # Create the complex memory
        print("Creating complex memory...")
        memory_note = MemoryNote(
            content=complex_content,
            tags=complex_tags,
            keywords=complex_keywords,
            context=complex_context,
            category=complex_category,
            timestamp=datetime.now().strftime("%Y%m%d%H%M"),
            retrieval_count=5  # Non-zero retrieval count
        )
        
        # Add to memory system
        original_memory_id = memory_note.id
        memory_system_1.memories[original_memory_id] = memory_note
        
        # Add to ChromaDB with session metadata
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
        
        print(f"Memory created with ID: {original_memory_id}")
        print(f"Original tags count: {len(complex_tags)}")
        print(f"Original keywords count: {len(complex_keywords)}")
        print(f"Content length: {len(complex_content)}")
        
        # Verify memory was stored
        assert len(memory_system_1.memories) == 1
        original_memory = memory_system_1.memories[original_memory_id]
        assert original_memory.content == complex_content
        assert original_memory.tags == complex_tags
        assert original_memory.keywords == complex_keywords
        
        # Phase 2: Create new memory system and load session
        print(f"\nPhase 2: Loading session in new memory system instance...")
        
        memory_system_2 = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai",
            llm_model="gpt-4o-mini", 
            session_id=self.session_id  # Same session ID
        )
        
        # Use the same persist directory
        persist_dir = os.path.join(self.temp_dir, "memory_db")
        
        memory_system_2.retriever.persist_directory = persist_dir
        memory_system_2.retriever.client = chromadb.PersistentClient(path=persist_dir)
        memory_system_2.retriever.collection = memory_system_2.retriever.client.get_or_create_collection(
            name=memory_system_2.retriever.collection_name
        )
        
        # Load session memories (this should trigger our fixed deserialization)
        memory_system_2._load_session_memories()
        
        print(f"Loaded {len(memory_system_2.memories)} memories")
        
        # Phase 3: Verify integrity
        print(f"\nPhase 3: Verifying data integrity...")
        
        # Check that memory was loaded
        assert len(memory_system_2.memories) == 1, f"Expected 1 memory, got {len(memory_system_2.memories)}"
        
        # Get the loaded memory
        loaded_memory_id = list(memory_system_2.memories.keys())[0]
        loaded_memory = memory_system_2.memories[loaded_memory_id]
        
        # Verify ID preservation
        assert loaded_memory_id == original_memory_id, f"Memory ID mismatch: {loaded_memory_id} vs {original_memory_id}"
        
        # Verify content integrity (exact match)
        assert loaded_memory.content == complex_content, "Content was corrupted during save/load"
        
        # Verify tags integrity (this was the main bug)
        assert isinstance(loaded_memory.tags, list), f"Tags should be list, got {type(loaded_memory.tags)}"
        assert loaded_memory.tags == complex_tags, f"Tags corrupted: {loaded_memory.tags} vs {complex_tags}"
        
        # Verify keywords integrity
        assert isinstance(loaded_memory.keywords, list), f"Keywords should be list, got {type(loaded_memory.keywords)}"
        assert loaded_memory.keywords == complex_keywords, f"Keywords corrupted: {loaded_memory.keywords} vs {complex_keywords}"
        
        # Verify other metadata
        assert loaded_memory.context == complex_context, "Context was corrupted"
        assert loaded_memory.category == complex_category, "Category was corrupted"
        assert loaded_memory.retrieval_count == 5, "Retrieval count was corrupted"
        
        # Verify no character array corruption in tags
        for tag in loaded_memory.tags:
            assert isinstance(tag, str), f"Tag should be string, got {type(tag)}: {tag}"
            assert len(tag) > 1 or tag.isalnum(), f"Tag appears to be corrupted single character: '{tag}'"
        
        # Verify no character array corruption in keywords  
        for keyword in loaded_memory.keywords:
            assert isinstance(keyword, str), f"Keyword should be string, got {type(keyword)}: {keyword}"
            assert len(keyword) > 1 or keyword.isalnum(), f"Keyword appears to be corrupted single character: '{keyword}'"
        
        print("✓ Content integrity verified")
        print("✓ Tags integrity verified (no character array corruption)")
        print("✓ Keywords integrity verified")
        print("✓ Metadata integrity verified")
        
        # Phase 4: Test search functionality with loaded data
        print(f"\nPhase 4: Testing search functionality...")
        
        # Search for the memory using content
        search_results = memory_system_2.search("machine learning neural networks", k=5)
        assert len(search_results) >= 1, "Should find the loaded memory in search"
        
        # The main test was verifying session load integrity, which passed.
        # Search format may vary, so let's just verify the memory is findable
        found_in_search = False
        for result in search_results:
            if result.get('id') == original_memory_id:
                found_in_search = True
                break
        
        assert found_in_search, "Memory should be findable in search results"
        
        print("✓ Search functionality verified")
        
        print(f"\n🎉 Test passed! Complex memory with {len(complex_tags)} tags and {len(complex_keywords)} keywords")
        print("   successfully preserved across session save/load operation.")


if __name__ == "__main__":
    """Run the test directly."""
    test = TestSessionIntegrity()
    test.setup_method()
    
    try:
        test.test_complex_memory_session_integrity()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        test.teardown_method()
