import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from agentic_memory.llm_controller import LLMController

# Load environment variables from .env file
load_dotenv()

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Check for available API keys (excluding placeholders)
        openai_key = os.getenv('OPENAI_API_KEY', '')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        
        # Filter out placeholder keys
        def is_valid_key(key):
            return key and not key.startswith('sk-test-placeholder') and not key.startswith('sk-ant-placeholder')
        
        self.openai_key = openai_key if is_valid_key(openai_key) else None
        self.anthropic_key = anthropic_key if is_valid_key(anthropic_key) else None
        
        # Set has_valid_api_key flag for LLM-dependent tests
        self.has_valid_api_key = bool(self.openai_key or self.anthropic_key)
        
        # Skip tests if no valid API keys are available
        if not self.openai_key and not self.anthropic_key:
            self.skipTest("No valid API keys available (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        
        # Respect DEFAULT_LLM_BACKEND preference or fall back to available keys
        default_backend = os.getenv('DEFAULT_LLM_BACKEND', '').lower()
        default_model = os.getenv('DEFAULT_LLM_MODEL', '')
        
        if default_backend == 'anthropic' and self.anthropic_key:
            backend = "anthropic"
            model = default_model if default_model else "claude-3-haiku-20240307"
        elif default_backend == 'openai' and self.openai_key:
            backend = "openai"
            model = default_model if default_model else "gpt-4.1-mini"
        elif self.anthropic_key:
            # Prefer Anthropic if available and no explicit preference
            backend = "anthropic"
            model = "claude-3-haiku-20240307"
        else:
            # Fall back to OpenAI
            backend = "openai"
            model = "gpt-4.1-mini"
            
        self.memory_system = AgenticMemorySystem(
            session_id="test_session_2",
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model
        )
        
        # Create a temporary directory for persistent storage tests
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test."""
        # Force cleanup of ChromaDB connections
        try:
            if hasattr(self.memory_system, 'retriever') and hasattr(self.memory_system.retriever, 'client'):
                # Reset the client to close connections
                self.memory_system.retriever.client.reset()
        except:
            pass  # Ignore errors during cleanup
            
        # Clean up temporary directories with retry logic for Windows
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # On Windows, files might still be locked - try again after a brief wait
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    # Final attempt - just warn if we can't clean up
                    print(f"Warning: Could not clean up temp directory {self.temp_dir}")

    def test_extract_json_from_markdown_response(self):
        """Test robust JSON extraction from markdown-wrapped responses."""
        # Test JSON wrapped in ```json blocks
        markdown_json = '''```json
{
    "keywords": ["neural", "networks", "deep"],
    "context": "Deep learning discussion about neural networks",
    "tags": ["AI", "machine-learning", "deep-learning"]
}
```'''
        
        extracted = self.memory_system._extract_json_from_response(markdown_json)
        expected = '''{
    "keywords": ["neural", "networks", "deep"],
    "context": "Deep learning discussion about neural networks",
    "tags": ["AI", "machine-learning", "deep-learning"]
}'''
        self.assertEqual(extracted.strip(), expected.strip())

    def test_extract_json_from_generic_markdown_blocks(self):
        """Test JSON extraction from generic ``` blocks."""
        markdown_json = '''```
{
    "keywords": ["python", "programming", "language"],
    "context": "Python programming language discussion",
    "tags": ["programming", "python", "language"]
}
```'''
        
        extracted = self.memory_system._extract_json_from_response(markdown_json)
        expected = '''{
    "keywords": ["python", "programming", "language"],
    "context": "Python programming language discussion",
    "tags": ["programming", "python", "language"]
}'''
        self.assertEqual(extracted.strip(), expected.strip())

    def test_extract_json_without_markdown(self):
        """Test JSON extraction from plain text responses."""
        plain_json = '{"keywords": ["test"], "context": "Test context", "tags": ["test"]}'
        
        extracted = self.memory_system._extract_json_from_response(plain_json)
        self.assertEqual(extracted, plain_json)

    def test_persistent_storage_directory_creation(self):
        """Test that persistent storage directory is created."""
        from agentic_memory.retrievers import ChromaRetriever
        
        custom_dir = os.path.join(self.temp_dir, "custom_memory_db")
        
        # Create retriever with custom persist directory
        retriever = ChromaRetriever(
            collection_name="test_memories",
            persist_directory=custom_dir
        )
        
        # Verify directory was created
        self.assertTrue(os.path.exists(custom_dir))

    def test_persistent_storage_data_persistence(self):
        """Test that memory data persists across retriever instances."""
        from agentic_memory.retrievers import ChromaRetriever
        
        persist_dir = os.path.join(self.temp_dir, "persist_test")
        
        # Create first retriever and add data
        retriever1 = ChromaRetriever(
            collection_name="test_memories",
            persist_directory=persist_dir
        )
        
        test_content = "Python is a programming language used for machine learning"
        test_metadata = {
            "content": test_content,
            "keywords": ["python", "programming", "machine learning"],
            "context": "Programming discussion",
            "tags": ["programming", "python"]
        }
        
        retriever1.add_document(test_content, test_metadata, "test_id_1")
        
        # Create second retriever instance (simulating restart)
        retriever2 = ChromaRetriever(
            collection_name="test_memories",
            persist_directory=persist_dir
        )
        
        # Verify data persisted
        results = retriever2.search("Python programming", k=1)
        self.assertGreater(len(results['ids'][0]), 0)
        self.assertEqual(results['ids'][0][0], "test_id_1")

    def test_memory_evolution_system_invocation(self):
        """Test that the evolution system is properly invoked for similar content."""
        if not self.has_valid_api_key:
            self.skipTest("No valid API key available for LLM testing")
            
        # Track initial memory count
        initial_count = len(self.memory_system.memories)
        
        # Add first memory
        content1 = "The temple creature is mysterious and ancient"
        memory_id1 = self.memory_system.add_note(content1)
        self.assertIsNotNone(memory_id1)
        
        # Verify memory was added
        self.assertEqual(len(self.memory_system.memories), initial_count + 1)
        
        # Add similar memory - this should invoke the evolution system
        content2 = "The temple creature seems mysterious and ancient"
        memory_id2 = self.memory_system.add_note(content2)
        self.assertIsNotNone(memory_id2)
        
        # Verify evolution system was invoked by checking:
        # 1. At least one memory exists and is readable
        # 2. The evolution system found neighbors (first memory should be found as neighbor)
        memory1 = self.memory_system.read(memory_id1)
        self.assertIsNotNone(memory1)
        
        # Check that we can find related memories
        related_memories, indices = self.memory_system.find_related_memories(content2, k=5)
        self.assertIsNotNone(related_memories)
        
        # Should find at least the first memory as related
        self.assertGreater(len(indices), 0, "Evolution system should find related memories")
    
    def test_consolidation_result_handling(self):
        """Test that consolidation results are properly handled when they occur."""
        if not self.has_valid_api_key:
            self.skipTest("No valid API key available for LLM testing")
            
        # Create a simple mock consolidation scenario
        # We can't force the LLM to consolidate, but we can test the mechanism
        
        # Add first memory
        content1 = "Python is a programming language"
        memory_id1 = self.memory_system.add_note(content1)
        self.assertIsNotNone(memory_id1)
        
        # Add related memory
        content2 = "Python programming language for development"
        memory_id2 = self.memory_system.add_note(content2)
        self.assertIsNotNone(memory_id2)
        
        # Verify both memories are accessible (regardless of consolidation)
        memory1 = self.memory_system.read(memory_id1)
        self.assertIsNotNone(memory1)
        
        if memory_id1 == memory_id2:
            # If consolidation occurred, verify the consolidated memory contains relevant content
            self.assertIn("python", memory1.content.lower())
            self.assertIn("programming", memory1.content.lower())
            # Verify only one memory exists in this case
            self.assertTrue(memory_id2 in self.memory_system.memories)
        else:
            # If no consolidation, both memories should exist
            memory2 = self.memory_system.read(memory_id2) 
            self.assertIsNotNone(memory2)
            self.assertTrue(memory_id1 in self.memory_system.memories)
            self.assertTrue(memory_id2 in self.memory_system.memories)

    def test_consolidation_with_detail_integration(self):
        """Test that nearly identical memories with one differing detail get consolidated."""
        if not self.has_valid_api_key:
            self.skipTest("No valid API key available for LLM testing")
            
        # Create first memory with basic information
        content1 = "The ancient library contains thousands of scrolls with magical knowledge"
        memory_id1 = self.memory_system.add_note(
            content1,
            keywords=["library", "scrolls", "magic"],
            tags=["ancient", "knowledge"],
            context="Fantasy world exploration"
        )
        self.assertIsNotNone(memory_id1)
        
        # Create second memory that's nearly identical but adds one specific detail
        content2 = "The ancient library contains thousands of scrolls with magical knowledge, including rare elemental spells"
        memory_id2 = self.memory_system.add_note(
            content2,
            keywords=["library", "scrolls", "magic", "elemental", "spells"], 
            tags=["ancient", "knowledge", "spells"],
            context="Fantasy world exploration - magical research"
        )
        self.assertIsNotNone(memory_id2)
        
        # For nearly identical memories with just additional detail, consolidation MUST occur
        self.assertEqual(memory_id1, memory_id2, 
                        "Nearly identical memories with additional detail must be consolidated into one memory")
        
        # Consolidation occurred - verify integrated content and metadata
        consolidated_memory = self.memory_system.read(memory_id1)
        self.assertIsNotNone(consolidated_memory)
        
        # Verify the consolidated content contains both original and new information
        content_lower = consolidated_memory.content.lower()
        self.assertIn("library", content_lower)
        self.assertIn("scrolls", content_lower)
        self.assertIn("magical", content_lower)
        self.assertIn("elemental", content_lower, "Consolidated memory must contain the additional detail 'elemental'")
        
        # Verify metadata integration - should include keywords from both memories
        self.assertIn("library", consolidated_memory.keywords)
        self.assertIn("magic", consolidated_memory.keywords)
        self.assertIn("elemental", consolidated_memory.keywords, "Consolidated memory must include new keyword 'elemental'")
        
        # Verify tags integration
        self.assertIn("ancient", consolidated_memory.tags)
        self.assertIn("knowledge", consolidated_memory.tags)
        self.assertIn("spells", consolidated_memory.tags, "Consolidated memory must include new tag 'spells'")
        
        # Verify only one memory exists in the system for this content
        memories_with_library = [m for m in self.memory_system.memories.values() 
                               if "library" in m.content.lower()]
        self.assertEqual(len(memories_with_library), 1, 
                       "Should have exactly one consolidated memory about the library")
        
        print(f"SUCCESS: Consolidation occurred. Final content: {consolidated_memory.content}")
        print(f"Final keywords: {consolidated_memory.keywords}")
        print(f"Final tags: {consolidated_memory.tags}")

    def test_memory_deduplication_low_similarity(self):
        """Test that dissimilar memories are not deduplicated."""
        # Add first memory
        content1 = "Python is a programming language for data science"
        memory_id1 = self.memory_system.add_note(content1)
        
        # Add dissimilar memory - should NOT be merged
        content2 = "JavaScript is used for web development"
        memory_id2 = self.memory_system.add_note(content2)
        
        # Should return different IDs
        self.assertNotEqual(memory_id1, memory_id2)
        
        # Verify both memories exist
        memory1 = self.memory_system.read(memory_id1)
        memory2 = self.memory_system.read(memory_id2)
        self.assertIsNotNone(memory1)
        self.assertIsNotNone(memory2)
        self.assertEqual(memory1.content, content1)
        self.assertEqual(memory2.content, content2)

    def test_llm_based_consolidation(self):
        """Test LLM-based memory consolidation functionality."""
        if not self.has_valid_api_key:
            self.skipTest("No valid API key available for LLM testing")
        
        # Add first memory
        content1 = "Python programming language for machine learning"
        memory_id1 = self.memory_system.add_note(content1)
        self.assertIsNotNone(memory_id1)
        
        # Add similar memory - should potentially be consolidated by LLM
        content2 = "Python programming language for data science"
        memory_id2 = self.memory_system.add_note(content2)
        self.assertIsNotNone(memory_id2)
        
        # Verify that memories were processed (either consolidated or kept separate)
        # The exact behavior depends on LLM decision
        memory1 = self.memory_system.read(memory_id1)
        if memory_id1 == memory_id2:
            # Consolidated case - content should be merged
            self.assertIn("Python", memory1.content)
        else:
            # Separate memories case
            memory2 = self.memory_system.read(memory_id2)
            self.assertIsNotNone(memory2)
            self.assertNotEqual(memory1.content, memory2.content)

    def test_memory_search_functionality(self):
        """Test memory search functionality with semantic similarity."""
        # Add some test memories
        contents = [
            "Neural networks are used in deep learning for AI applications",
            "Deep learning neural networks for artificial intelligence",
            "Machine learning algorithms for data analysis",
            "Web development with JavaScript frameworks"
        ]
        
        memory_ids = []
        for content in contents:
            memory_id = self.memory_system.add_note(content)
            memory_ids.append(memory_id)
        
        # Test search functionality
        query = "Neural networks for deep learning AI"
        search_results = self.memory_system.search(query, k=5)
        
        # Should find relevant memories
        self.assertGreater(len(search_results), 0)
        
        # Verify search results have proper structure
        for result in search_results:
            self.assertIn('id', result)
            self.assertIn('content', result)
            self.assertTrue(result['id'] in memory_ids)

    def test_memory_evolution_processing(self):
        """Test that memories are processed through the evolution system."""
        if not self.has_valid_api_key:
            self.skipTest("No valid API key available for LLM testing")
            
        # Create initial memory
        initial_content = "Neural networks are powerful"
        memory_id = self.memory_system.add_note(
            initial_content,
            keywords=["neural", "networks"],
            tags=["AI", "deep-learning"]
        )
        
        self.assertIsNotNone(memory_id)
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        self.assertIn("neural", memory.content.lower())
        
        # Add another related memory that might trigger evolution
        related_content = "Deep learning neural networks are used in AI"
        memory_id2 = self.memory_system.add_note(
            related_content,
            keywords=["neural", "networks", "deep"],
            tags=["AI", "deep-learning"]
        )
        
        self.assertIsNotNone(memory_id2)
        
        # Verify both memories exist (evolution system may or may not consolidate)
        memory1 = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory1)
        
        if memory_id != memory_id2:
            memory2 = self.memory_system.read(memory_id2)
            self.assertIsNotNone(memory2)
        else:
            # Memories were consolidated
            self.assertIn("neural", memory1.content.lower())

    def test_deduplication_integration_with_real_content(self):
        """Test complete deduplication workflow with realistic content."""
        if not self.has_valid_api_key:
            self.skipTest("No valid API key available for LLM testing")
            
        # Test content from the guide
        content1 = "The temple creature is mysterious"
        memory_id1 = self.memory_system.add_note(content1)
        self.assertIsNotNone(memory_id1)
        
        content2 = "The temple creature seems mysterious and ancient"
        memory_id2 = self.memory_system.add_note(content2)
        self.assertIsNotNone(memory_id2)
        
        # LLM may or may not consolidate - verify both cases work
        memory1 = self.memory_system.read(memory_id1)
        self.assertIsNotNone(memory1)
        self.assertIn("mysterious", memory1.content)
        
        if memory_id1 == memory_id2:
            # Consolidated case - content should contain elements from both
            self.assertIn("creature", memory1.content)
        else:
            # Separate memories case
            memory2 = self.memory_system.read(memory_id2)
            self.assertIsNotNone(memory2)
            self.assertIn("mysterious", memory2.content)
        
        # Test that retrieval count was incremented
        self.assertGreaterEqual(memory1.retrieval_count, 1)

    def test_persistent_storage_with_consolidation(self):
        """Test persistent storage works with memory consolidation."""
        from agentic_memory.retrievers import ChromaRetriever
        
        persist_dir = os.path.join(self.temp_dir, "consolidation_test")
        
        # Create memory system with custom persist directory
        custom_retriever = ChromaRetriever(
            collection_name="test_memories",
            persist_directory=persist_dir
        )
        
        # Replace the retriever in memory system for this test
        original_retriever = self.memory_system.retriever
        self.memory_system.retriever = custom_retriever
        
        try:
            # Add memories
            contents = [
                "Python data science libraries",
                "Machine learning with scikit-learn",
                "Deep learning frameworks like TensorFlow"
            ]
            
            for content in contents:
                self.memory_system.add_note(content)
            
            # Force consolidation
            self.memory_system.consolidate_memories()
            
            # Verify data still exists after consolidation
            for content in contents:
                results = self.memory_system.search(content, k=1)
                self.assertGreater(len(results), 0)
                
        finally:
            # Restore original retriever
            self.memory_system.retriever = original_retriever

    def test_robust_json_parsing_with_analyze_content(self):
        """Test robust JSON parsing integration with analyze_content method."""
        # Mock LLM response with markdown-wrapped JSON
        mock_response = '''```json
{
    "keywords": ["neural", "networks", "deep", "learning"],
    "context": "Discussion about neural networks in deep learning applications",
    "tags": ["AI", "machine-learning", "deep-learning", "neural-networks"]
}
```'''
        
        with patch.object(self.memory_system.llm_controller.llm, 'get_completion', return_value=mock_response):
            result = self.memory_system.analyze_content("Neural networks are used in deep learning")
            
            # Verify JSON was parsed correctly
            self.assertIsInstance(result, dict)
            self.assertIn("keywords", result)
            self.assertIn("context", result)
            self.assertIn("tags", result)
            self.assertEqual(result["keywords"], ["neural", "networks", "deep", "learning"])

if __name__ == '__main__':
    unittest.main()
