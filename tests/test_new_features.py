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
from uuid import uuid4

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
            
        # Unique session per test instance for isolation
        self.session_id = f"test_session_{uuid4()}"
        self.memory_system = AgenticMemorySystem(
            session_id=self.session_id,
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model
        )
        
        # Create a temporary directory for persistent storage tests
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test."""
        # Delete all docs for this test session, then reset client
        try:
            if hasattr(self, 'memory_system'):
                try:
                    self.memory_system.delete_all_by_session(self.session_id)
                except Exception:
                    pass
                try:
                    if hasattr(self.memory_system, 'retriever') and hasattr(self.memory_system.retriever, 'client'):
                        # Reset the client to close connections
                        self.memory_system.retriever.client.reset()
                except Exception:
                    pass
        except Exception:
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

    def test_memory_search_functionality(self):
        """Create notes in two sessions and verify searches are session-scoped."""
        # Build two distinct session IDs
        from uuid import uuid4
        sess_a = f"search_session_A_{uuid4()}"
        sess_b = f"search_session_B_{uuid4()}"

        # Choose backend/model similar to setUp
        default_backend = os.getenv('DEFAULT_LLM_BACKEND', '').lower()
        default_model = os.getenv('DEFAULT_LLM_MODEL', '')
        if default_backend == 'anthropic' and self.anthropic_key:
            backend = "anthropic"
            model = default_model if default_model else "claude-3-haiku-20240307"
        elif default_backend == 'openai' and self.openai_key:
            backend = "openai"
            model = default_model if default_model else "gpt-4.1-mini"
        elif self.anthropic_key:
            backend = "anthropic"
            model = "claude-3-haiku-20240307"
        else:
            backend = "openai"
            model = "gpt-4.1-mini"

        ms_a = AgenticMemorySystem(session_id=sess_a, model_name='all-MiniLM-L6-v2', llm_backend=backend, llm_model=model)
        ms_b = AgenticMemorySystem(session_id=sess_b, model_name='all-MiniLM-L6-v2', llm_backend=backend, llm_model=model)

        # Add distinct memories with unique anchor keywords per session
        contents_a = [
            "Neural networks sessionA key: sessA_key1",
            "Deep learning sessionA key: sessA_key2",
            "AI research sessionA key: sessA_key3",
        ]
        contents_b = [
            "Data science sessionB key: sessB_key1",
            "ML pipelines sessionB key: sessB_key2",
            "Feature engineering sessionB key: sessB_key3",
        ]

        ids_a = [ms_a.add_note(c) for c in contents_a]
        ids_b = [ms_b.add_note(c) for c in contents_b]

        # Search within session A and ensure results are from session A only
        for kw in ["sessA_key1", "sessA_key2", "sessA_key3"]:
            res_a = ms_a.search(kw, k=5)
            self.assertGreater(len(res_a), 0)
            for r in res_a:
                self.assertIn('id', r)
                self.assertIn('content', r)
                # Ensure we did not accidentally retrieve a memory from session B
                self.assertNotIn(r['id'], ids_b)

        # Search within session B and ensure results are from session B only
        for kw in ["sessB_key1", "sessB_key2", "sessB_key3"]:
            res_b = ms_b.search(kw, k=5)
            self.assertGreater(len(res_b), 0)
            for r in res_b:
                self.assertIn('id', r)
                self.assertIn('content', r)
                self.assertNotIn(r['id'], ids_a)
        
        # Cleanup the two temporary sessions
        try:
            ms_a.delete_all_by_session(sess_a)
            ms_b.delete_all_by_session(sess_b)
        except Exception:
            pass
        try:
            if hasattr(ms_a, 'retriever') and hasattr(ms_a.retriever, 'client'):
                ms_a.retriever.client.reset()
            if hasattr(ms_b, 'retriever') and hasattr(ms_b.retriever, 'client'):
                ms_b.retriever.client.reset()
        except Exception:
            pass

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
        
        with patch.object(self.memory_system.llm_controller, 'get_completion', return_value=mock_response):
            result = self.memory_system.analyze_content("Neural networks are used in deep learning")
            
            # Verify JSON was parsed correctly
            self.assertIsInstance(result, dict)
            self.assertIn("keywords", result)
            self.assertIn("context", result)
            self.assertIn("tags", result)
            self.assertEqual(result["keywords"], ["neural", "networks", "deep", "learning"])

if __name__ == '__main__':
    unittest.main()
