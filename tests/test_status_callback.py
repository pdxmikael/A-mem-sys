"""
Test for status callback functionality in the refactored memory system.

This test validates that the status callback system properly reports status messages
during memory operations with the correct content and styling.
"""

import uuid
import tempfile
import shutil
from datetime import datetime
import os
import sys
from unittest.mock import Mock

# Add the parent directory to sys.path to import agentic_memory modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import chromadb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agentic_memory'))
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote

class TestStatusCallback:
    """Test status callback functionality."""
    
    def setup_method(self):
        """Set up test environment with a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        # Create a mock status callback to capture calls
        self.status_callback_mock = Mock()
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            try:
                import time
                time.sleep(0.5)  # Allow time for file locks to be released
                shutil.rmtree(self.temp_dir)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not clean up temp directory: {e}")
    
    def test_status_callback_during_add_note(self):
        """
        Test that status callbacks are properly called during add_note operation.
        """
        # Create memory system with status callback
        persist_dir = os.path.join(self.temp_dir, "memory_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        memory_system = AgenticMemorySystem(
            session_id=self.session_id,
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4.1-mini",
            persist_directory=persist_dir,
            status_callback=self.status_callback_mock
        )
        
        # Add a simple note
        content = "This is a test memory for status callback testing."
        memory_system.add_note(content)
        
        # Verify that status callback was called with expected messages
        expected_calls = [
            # Basic flow messages
            ("Creating new memory...", 2.0, "dim yellow"),
            ("Analyzing content with LLM...", 2.5, "dim cyan"),
            ("Processing memory evolution...", 2.5, "dim cyan"),
            ("Adding memory to storage...", 2.0, "dim cyan"),
            ("Memory created successfully", 2.0, "green"),
        ]
        
        # Check that all expected calls were made
        assert self.status_callback_mock.call_count >= len(expected_calls), \
            f"Expected at least {len(expected_calls)} calls, got {self.status_callback_mock.call_count}"
        
        # Check specific calls (order might vary due to async operations)
        call_args_list = self.status_callback_mock.call_args_list
        called_messages = [call[0][0] for call in call_args_list]
        
        for expected_message, expected_duration, expected_style in expected_calls:
            # Find if this message was called
            found = False
            for call in call_args_list:
                args, kwargs = call
                if len(args) >= 1 and args[0] == expected_message:
                    found = True
                    # Check duration and style if provided
                    if len(args) >= 2:
                        assert args[1] == expected_duration or args[1] == 2.0, \
                            f"Duration mismatch for '{expected_message}': expected {expected_duration}, got {args[1]}"
                    if len(args) >= 3:
                        assert args[2] == expected_style, \
                            f"Style mismatch for '{expected_message}': expected {expected_style}, got {args[2]}"
                    break
            
            assert found, f"Expected status message '{expected_message}' was not called. Called messages: {called_messages}"
    
    def test_status_callback_during_consolidate_memories(self):
        """
        Test that status callbacks are properly called during consolidate_memories operation.
        """
        # Create memory system with status callback
        persist_dir = os.path.join(self.temp_dir, "memory_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        memory_system = AgenticMemorySystem(
            session_id=self.session_id,
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4.1-mini",
            persist_directory=persist_dir,
            status_callback=self.status_callback_mock
        )
        
        # Print memories before adding new ones
        print(f"Memories in session {self.session_id} before adding new memories:")
        for mem_id, memory in memory_system.memories.items():
            print(f"  - ID: {mem_id}, Content: {memory.content[:50]}...")
        
        # Add multiple notes to trigger consolidation
        num_notes = 5
        for i in range(num_notes):
            content = f"Test memory {i} for consolidation testing."
            memory_system.add_note(content)
        
        # Print memories after adding new ones but before consolidation
        print(f"Memories in session {self.session_id} after adding {num_notes} new memories:")
        for mem_id, memory in memory_system.memories.items():
            print(f"  - ID: {mem_id}, Content: {memory.content[:50]}...")
        
        # Get the actual number of memories (might be different from what we added)
        actual_memory_count = len(memory_system.memories)
        
        # Reset mock to clear previous calls
        self.status_callback_mock.reset_mock()
        
        # Manually trigger consolidation
        memory_system.consolidate_memories()
        
        # Print memories after consolidation
        print(f"Memories in session {self.session_id} after consolidation:")
        for mem_id, memory in memory_system.memories.items():
            print(f"  - ID: {mem_id}, Content: {memory.content[:50]}...")
        
        # Verify that status callback was called with consolidation messages
        expected_calls = [
            ("Consolidating related memories...", 3.0, "dim yellow"),
        ]
        
        # Check that all expected calls were made
        assert self.status_callback_mock.call_count >= len(expected_calls), \
            f"Expected at least {len(expected_calls)} calls, got {self.status_callback_mock.call_count}"
        
        # Check specific calls
        call_args_list = self.status_callback_mock.call_args_list
        called_messages = [call[0][0] for call in call_args_list]
        
        # Check for the rebuilding message which contains the actual number of memories
        rebuilding_message_found = False
        consolidated_message_found = False
        for call in call_args_list:
            args, kwargs = call
            if len(args) >= 1 and "Rebuilding memory index for" in args[0] and str(actual_memory_count) in args[0]:
                rebuilding_message_found = True
                # Check style if provided
                if len(args) >= 3:
                    assert args[2] == "dim cyan", \
                        f"Style mismatch for rebuilding message: expected 'dim cyan', got {args[2]}"
            elif len(args) >= 1 and "Consolidated" in args[0] and str(actual_memory_count) in args[0]:
                consolidated_message_found = True
                # Check style if provided
                if len(args) >= 3:
                    assert args[2] == "green", \
                        f"Style mismatch for consolidated message: expected 'green', got {args[2]}"
        
        assert rebuilding_message_found, f"Expected rebuilding message with '{actual_memory_count}' memories was not called. Called messages: {called_messages}"
        assert consolidated_message_found, f"Expected consolidated message with '{actual_memory_count}' memories was not called. Called messages: {called_messages}"
        
        # Check other expected calls
        for expected_message, expected_duration, expected_style in expected_calls:
            # Find if this message was called
            found = False
            for call in call_args_list:
                args, kwargs = call
                if len(args) >= 1 and args[0] == expected_message:
                    found = True
                    # Check duration and style if provided
                    if len(args) >= 2:
                        # Duration might vary, just check it's a number
                        assert isinstance(args[1], (int, float)), \
                            f"Duration should be a number for '{expected_message}', got {type(args[1])}"
                    if len(args) >= 3:
                        assert args[2] == expected_style, \
                            f"Style mismatch for '{expected_message}': expected {expected_style}, got {args[2]}"
                    break
            
            assert found, f"Expected status message '{expected_message}' was not called. Called messages: {called_messages}"
    
    def test_status_callback_during_memory_processing(self):
        """
        Test that status callbacks are properly called during memory processing operations.
        """
        # Create memory system with status callback
        persist_dir = os.path.join(self.temp_dir, "memory_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        memory_system = AgenticMemorySystem(
            session_id=self.session_id,
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai", 
            llm_model="gpt-4.1-mini",
            persist_directory=persist_dir,
            status_callback=self.status_callback_mock
        )
        
        # Add first note (no evolution processing)
        content1 = "First test memory for processing testing."
        memory_system.add_note(content1)
        
        # Reset mock to clear previous calls
        self.status_callback_mock.reset_mock()
        
        # Add second note which should trigger processing
        content2 = "Second test memory that should trigger processing."
        memory_system.add_note(content2)
        
        # Verify that status callback was called with processing messages
        expected_processing_calls = [
            ("Analyzing memory relationships...", 2.0, "dim cyan"),
            ("Analyzing memory evolution needs...", 3.0, "dim yellow"),
        ]
        
        # Check that processing calls were made
        call_args_list = self.status_callback_mock.call_args_list
        called_messages = [call[0][0] for call in call_args_list]
        
        for expected_message, expected_duration, expected_style in expected_processing_calls:
            # Find if this message was called
            found = False
            for call in call_args_list:
                args, kwargs = call
                if len(args) >= 1 and args[0] == expected_message:
                    found = True
                    # Check style if provided
                    if len(args) >= 3:
                        assert args[2] == expected_style, \
                            f"Style mismatch for '{expected_message}': expected {expected_style}, got {args[2]}"
                    break
            
            assert found, f"Expected processing status message '{expected_message}' was not called. Called messages: {called_messages}"

if __name__ == "__main__":
    test = TestStatusCallback()
    test.setup_method()
    try:
        print("Running test_status_callback_during_add_note...")
        test.test_status_callback_during_add_note()
        print("Running test_status_callback_during_consolidate_memories...")
        test.test_status_callback_during_consolidate_memories()
        print("Running test_status_callback_during_memory_processing...")
        test.test_status_callback_during_memory_processing()
        print("\n✅ All status callback tests passed!")
    except Exception as e:
        print(f"\n❌ Status callback test failed: {e}")
        raise
    finally:
        test.teardown_method()
