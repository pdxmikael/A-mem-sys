import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from agentic_memory.llm_controller import LLMController, AnthropicController

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
            model = default_model if default_model else "gpt-4o-mini"
        elif self.anthropic_key:
            # Prefer Anthropic if available and no explicit preference
            backend = "anthropic"
            model = "claude-3-haiku-20240307"
        else:
            # Fall back to OpenAI
            backend = "openai"
            model = "gpt-4o-mini"
            
        self.memory_system = AgenticMemorySystem(
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

    @unittest.skipUnless(os.getenv('ANTHROPIC_API_KEY'), "Anthropic API key not available")
    def test_anthropic_controller_initialization(self):
        """Test Anthropic API controller initialization."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('anthropic.Anthropic') as mock_anthropic:
                # Test controller creation
                controller = AnthropicController(model="claude-3-5-sonnet-20241022", api_key="test-key")
                
                # Verify initialization
                self.assertEqual(controller.model, "claude-3-5-sonnet-20241022")
                mock_anthropic.assert_called_once_with(api_key="test-key")

    @unittest.skipUnless(os.getenv('ANTHROPIC_API_KEY'), "Anthropic API key not available")
    def test_anthropic_llm_controller_backend(self):
        """Test LLMController with Anthropic backend."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('anthropic.Anthropic'):
                # Test controller creation with anthropic backend
                controller = LLMController(
                    backend="anthropic",
                    model="claude-3-5-sonnet-20241022",
                    api_key="test-key"
                )
                
                # Verify it's using AnthropicController
                self.assertIsInstance(controller.llm, AnthropicController)

    def test_anthropic_memory_system_integration(self):
        """Test AgenticMemorySystem with Anthropic backend."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('anthropic.Anthropic'):
                # Create memory system with Anthropic backend
                memory_system = AgenticMemorySystem(
                    llm_backend="anthropic",
                    llm_model="claude-3-5-sonnet-20241022"
                )
                
                # Verify the system was created successfully
                self.assertIsNotNone(memory_system.llm_controller)
                self.assertIsInstance(memory_system.llm_controller.llm, AnthropicController)

    @patch('anthropic.Anthropic')
    def test_anthropic_completion_call(self, mock_anthropic):
        """Test Anthropic API completion call."""
        # Mock the Anthropic client response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '{"keywords": ["test"], "context": "Test context", "tags": ["test"]}'
        
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            controller = AnthropicController(api_key="test-key")
            
            # Test completion call
            response = controller.get_completion(
                "Test prompt", 
                {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
                temperature=0.7
            )
            
            # Verify the call was made
            mock_client.messages.create.assert_called_once()
            self.assertIn("keywords", response)

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

    def test_memory_deduplication_high_similarity(self):
        """Test memory deduplication with high similarity content."""
        # Add first memory
        content1 = "The temple creature is mysterious and ancient"
        memory_id1 = self.memory_system.add_note(content1)
        
        # Add very similar memory - should be merged
        content2 = "The temple creature seems mysterious and ancient"
        memory_id2 = self.memory_system.add_note(content2)
        
        # Should return same ID (merged)
        self.assertEqual(memory_id1, memory_id2)
        
        # Verify only one memory exists
        memory = self.memory_system.read(memory_id1)
        self.assertIsNotNone(memory)
        # Content should be updated with additional info
        self.assertIn("Additional:", memory.content)

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

    def test_calculate_content_similarity(self):
        """Test content similarity calculation."""
        content1 = "Python programming language for machine learning"
        content2 = "Python programming language for data science"
        content3 = "JavaScript web development framework"
        
        # Test high similarity
        similarity_high = self.memory_system._calculate_content_similarity(content1, content2)
        self.assertGreaterEqual(similarity_high, 0.5)
        
        # Test low similarity
        similarity_low = self.memory_system._calculate_content_similarity(content1, content3)
        self.assertLess(similarity_low, 0.5)
        
        # Test identical content
        similarity_identical = self.memory_system._calculate_content_similarity(content1, content1)
        self.assertEqual(similarity_identical, 1.0)

    def test_find_similar_memories(self):
        """Test finding similar memories above threshold."""
        # Add some test memories
        contents = [
            "Neural networks are used in deep learning for AI applications",
            "Deep learning neural networks for artificial intelligence",
            "Machine learning algorithms for data analysis",
            "Web development with JavaScript frameworks"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
        
        # Test finding similar memories
        query = "Neural networks for deep learning AI"
        similar_memories = self.memory_system._find_similar_memories(query, threshold=0.3)
        
        # Should find at least the first two similar memories
        self.assertGreaterEqual(len(similar_memories), 2)
        
        # Test with high threshold - should find fewer
        similar_memories_high = self.memory_system._find_similar_memories(query, threshold=0.8)
        self.assertLessEqual(len(similar_memories_high), len(similar_memories))

    def test_merge_or_update_memory_content(self):
        """Test merging or updating memory content."""
        # Create initial memory
        initial_content = "Neural networks are powerful"
        memory_id = self.memory_system.add_note(
            initial_content,
            keywords=["neural", "networks"],
            tags=["AI", "deep-learning"]
        )
        
        existing_memory = self.memory_system.read(memory_id)
        
        # Test merging with new content
        new_content = "Neural networks can solve complex problems"
        new_keywords = ["neural", "networks", "complex"]
        new_tags = ["AI", "deep-learning", "problem-solving"]
        
        merged_id = self.memory_system._merge_or_update_memory(
            existing_memory,
            new_content,
            keywords=new_keywords,
            tags=new_tags
        )
        
        # Should return same ID
        self.assertEqual(merged_id, memory_id)
        
        # Verify content was updated
        updated_memory = self.memory_system.read(memory_id)
        self.assertIn("Additional:", updated_memory.content)
        
        # Verify metadata was merged
        self.assertIn("complex", updated_memory.keywords)
        self.assertIn("problem-solving", updated_memory.tags)

    def test_deduplication_integration_with_real_content(self):
        """Test complete deduplication workflow with realistic content."""
        # Test content from the guide
        content1 = "The temple creature is mysterious"
        memory_id1 = self.memory_system.add_note(content1)
        
        content2 = "The temple creature seems mysterious and ancient"
        memory_id2 = self.memory_system.add_note(content2)
        
        # Should deduplicate (return same ID)
        self.assertEqual(memory_id1, memory_id2)
        
        # Verify merged content
        final_memory = self.memory_system.read(memory_id1)
        self.assertIsNotNone(final_memory)
        self.assertIn("mysterious", final_memory.content)
        
        # Test that retrieval count was incremented
        self.assertGreaterEqual(final_memory.retrieval_count, 1)

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
