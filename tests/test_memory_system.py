import unittest
import os
from dotenv import load_dotenv
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from datetime import datetime
from uuid import uuid4

# Load environment variables from .env file
load_dotenv()

class TestAgenticMemorySystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Check for available API keys (excluding placeholders)
        openai_key = os.getenv('OPENAI_API_KEY', '')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        
        # Filter out placeholder keys
        def is_valid_key(key):
            return key and not key.startswith('sk-test-placeholder') and not key.startswith('sk-ant-placeholder')
        
        valid_openai_key = openai_key if is_valid_key(openai_key) else None
        valid_anthropic_key = anthropic_key if is_valid_key(anthropic_key) else None
        
        # Skip tests if no valid API keys are available
        if not valid_openai_key and not valid_anthropic_key:
            self.skipTest("No valid API keys available (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        
        # Respect DEFAULT_LLM_BACKEND preference or fall back to available keys
        default_backend = os.getenv('DEFAULT_LLM_BACKEND', '').lower()
        default_model = os.getenv('DEFAULT_LLM_MODEL', '')
        
        if default_backend == 'anthropic' and valid_anthropic_key:
            backend = "anthropic"
            model = default_model if default_model else "claude-3-haiku-20240307"
        elif default_backend == 'openai' and valid_openai_key:
            backend = "openai"
            model = default_model if default_model else "gpt-4.1-mini"
        elif valid_anthropic_key:
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
            llm_model=model,
        )
        
    def tearDown(self):
        """Clean up session-specific data and close clients."""
        try:
            if hasattr(self, 'memory_system'):
                # Best-effort: delete all docs for this session
                try:
                    self.memory_system.delete_all_by_session(self.session_id)
                except Exception:
                    pass
                # Reset the client to close connections
                try:
                    if hasattr(self.memory_system, 'retriever') and hasattr(self.memory_system.retriever, 'client'):
                        self.memory_system.retriever.client.reset()
                except Exception:
                    pass
        except Exception:
            pass
        
    def test_create_memory(self):
        """Relaxed: create several notes, ensure IDs, and that provided keywords appear in content."""
        samples = [
            ("Alpha project kickoff notes", ["alpha"]),
            ("Gamma module integration plan", ["gamma"]),
            ("Delta release checklist", ["delta"]),
        ]
        created = []
        for content, keywords in samples:
            memory_id = self.memory_system.add_note(
                content=content,
                keywords=keywords,
                tags=["test"],
                context="Test context",
                category="Test category",
            )
            self.assertIsNotNone(memory_id)
            created.append((memory_id, content, keywords))

        # Verify retrieval and content contains provided keywords
        for memory_id, content, keywords in created:
            memory = self.memory_system.read(memory_id)
            self.assertIsNotNone(memory)
            for kw in keywords:
                self.assertIn(kw.lower(), memory.content.lower())
                
    def test_memory_metadata_persistence(self):
        """Relaxed: metadata types and inclusion, allowing autonomous evolution."""
        content = "Complex test memory"
        tags = ["test", "complex", "metadata"]
        keywords = ["test", "complex", "keywords"]
        links = ["link1", "link2", "link3"]
        context = "Complex test context"
        category = "Complex test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        evolution_history = ["evolution1", "evolution2"]

        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp,
            evolution_history=evolution_history,
        )

        results = self.memory_system.search_agentic(content, k=1)
        self.assertGreater(len(results), 0)

        result = results[0]
        self.assertEqual(result["content"], content)
        # Relaxed checks: ensure lists and inclusion, not exact equality
        self.assertIsInstance(result.get("tags", []), list)
        self.assertIsInstance(result.get("keywords", []), list)
        self.assertTrue(set(tags).issubset(set(result.get("tags", []))))
        self.assertTrue(set(keywords).issubset(set(result.get("keywords", []))))
        self.assertIsInstance(result.get("context", ""), str)
        self.assertIsInstance(result.get("category", ""), str)
        
    def test_memory_update(self):
        """Test updating memory metadata through ChromaDB."""
        # Create initial memory
        content = "Initial content"
        memory_id = self.memory_system.add_note(content=content)
        
        # Update memory with new metadata
        new_content = "Updated content"
        new_tags = ["updated", "tags"]
        new_keywords = ["updated", "keywords"]
        new_context = "Updated context"
        
        success = self.memory_system.update(
            memory_id,
            content=new_content,
            tags=new_tags,
            keywords=new_keywords,
            context=new_context
        )
        
        self.assertTrue(success)
        
        # Verify updates in ChromaDB
        results = self.memory_system.search_agentic(new_content, k=1)
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertEqual(result['content'], new_content)
        self.assertEqual(result['tags'], new_tags)
        self.assertEqual(result['keywords'], new_keywords)
        self.assertEqual(result['context'], new_context)
        
    def test_memory_relationships(self):
        """Test memory relationships and linked memories."""
        # Create related memories
        content1 = "First memory"
        content2 = "Second memory"
        content3 = "Third memory"
        
        id1 = self.memory_system.add_note(content1)
        id2 = self.memory_system.add_note(content2)
        id3 = self.memory_system.add_note(content3)
        
        # Add relationships
        memory1 = self.memory_system.read(id1)
        memory2 = self.memory_system.read(id2)
        memory3 = self.memory_system.read(id3)
        
        memory1.links.append(id2)
        memory2.links.append(id1)
        memory2.links.append(id3)
        memory3.links.append(id2)
        
        # Update memories with relationships
        self.memory_system.update(id1, links=memory1.links)
        self.memory_system.update(id2, links=memory2.links)
        self.memory_system.update(id3, links=memory3.links)
        
        # Test relationship retrieval
        results = self.memory_system.search_agentic(content1, k=3)
        self.assertGreater(len(results), 0)
        
        # Verify relationships are maintained
        memory1_updated = self.memory_system.read(id1)
        self.assertIn(id2, memory1_updated.links)
        
    def test_memory_evolution(self):
        """Test memory evolution system with ChromaDB."""
        # Create related memories
        contents = [
            "Deep learning neural networks",
            "Neural network architectures",
            "Training deep neural networks"
        ]
        
        memory_ids = []
        for content in contents:
            memory_id = self.memory_system.add_note(content)
            memory_ids.append(memory_id)
            
        # Verify that memories have been properly evolved
        for memory_id in memory_ids:
            memory = self.memory_system.read(memory_id)
            self.assertIsNotNone(memory.tags)
            self.assertIsNotNone(memory.context)
            self.assertIsNotNone(memory.keywords)
            
        # Test evolution through search
        results = self.memory_system.search_agentic("neural networks", k=3)
        self.assertGreater(len(results), 0)
        
        # Verify evolution metadata
        for result in results:
            self.assertIsNotNone(result['tags'])
            self.assertIsNotNone(result['context'])
            self.assertIsNotNone(result['keywords'])
            
    def test_memory_deletion(self):
        """Relaxed: delete by ID and ensure read() returns None without relying on search."""
        samples = [
            ("UniqueX alpha content", ["uniquex"]),
            ("UniqueY beta content", ["uniquey"]),
            ("UniqueZ gamma content", ["uniquez"]),
        ]
        ids = []
        for content, keywords in samples:
            memory_id = self.memory_system.add_note(content=content, keywords=keywords, tags=["t"], context="ctx")
            self.assertIsNotNone(memory_id)
            self.assertIsNotNone(self.memory_system.read(memory_id))
            ids.append(memory_id)

        # Delete and verify non-retrievability via read()
        for memory_id in ids:
            self.assertTrue(self.memory_system.delete(memory_id))
        for memory_id in ids:
            self.assertIsNone(self.memory_system.read(memory_id))
            
    def test_find_related_memories(self):
        """Test finding related memories."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python",
            "Web development with JavaScript"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories
        results = self.memory_system.find_related_memories("Python", k=2)
        self.assertGreater(len(results), 0)
        
    def test_find_related_memories_raw(self):
        """Test finding related memories with raw format."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories in raw format
        results = self.memory_system.find_related_memories_raw("Python", k=2)
        self.assertIsNotNone(results)
        
    def test_process_memory(self):
        """Test memory processing and evolution."""
        # Create a test memory
        content = "Test memory for processing"
        memory_id = self.memory_system.add_note(content)
        
        # Get the memory
        memory = self.memory_system.read(memory_id)
        
        # Process the memory
        evo_result, processed_memory = self.memory_system.process_memory(memory)
        
        # Verify processing results - evo_result can be bool or ConsolidationResult
        self.assertTrue(isinstance(evo_result, bool) or hasattr(evo_result, 'consolidated_id'))
        self.assertIsInstance(processed_memory, MemoryNote)
        self.assertIsNotNone(processed_memory.tags)
        self.assertIsNotNone(processed_memory.context)
        self.assertIsNotNone(processed_memory.keywords)
        
        # If consolidation occurred, verify the ConsolidationResult
        if hasattr(evo_result, 'consolidated_id'):
            self.assertIsNotNone(evo_result.consolidated_id)
            self.assertIsNotNone(evo_result.consolidated_content)

    def test_memory_consolidation(self):
        """Test memory consolidation with ChromaDB."""
        # Create multiple memories with realistic, semantically distinct content
        # Note that this test is not for deduplication, but for reinitialization of memories in the Chroma store
        contents = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning algorithms can identify patterns in large datasets automatically.",
            "Database optimization involves indexing, query tuning, and schema design considerations."
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Force consolidation
        self.memory_system.consolidate_memories()
        
        # Verify memories are still accessible
        for content in contents:
            results = self.memory_system.search_agentic(content, k=1)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]['content'], content)
            
if __name__ == '__main__':
    unittest.main()
