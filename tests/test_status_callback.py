"""
Test suite for A-MEM status callback integration.
Demonstrates how the consuming application can receive status updates during memory operations.
"""

import pytest
import time
import os
from dotenv import load_dotenv
from agentic_memory.memory_system import AgenticMemorySystem
from typing import List, Tuple


# Load environment variables
load_dotenv()

# Get LLM configuration from environment
DEFAULT_BACKEND = os.getenv("DEFAULT_LLM_BACKEND", "anthropic")
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "claude-3-5-haiku-20241022")

class StatusCallbackCollector:
    """Collects status messages for testing purposes."""
    
    def __init__(self):
        self.messages: List[Tuple[str, float, str]] = []
        
    def callback(self, message: str, duration: float = 2.0, style: str = "dim cyan"):
        """Mock status callback that collects messages instead of displaying them."""
        self.messages.append((message, duration, style))
        print(f"[{style}] {message} (duration: {duration}s)")  # For demo purposes
        
    def get_messages(self) -> List[Tuple[str, float, str]]:
        """Get all collected messages."""
        return self.messages.copy()
        
    def clear(self):
        """Clear collected messages."""
        self.messages.clear()
        
    def get_messages_by_style(self, style: str) -> List[Tuple[str, float, str]]:
        """Get messages filtered by style."""
        return [(msg, dur, s) for msg, dur, s in self.messages if s == style]


def test_status_callback_integration():
    """Test that status callbacks are properly called during memory operations."""
    
    # Create status collector
    status_collector = StatusCallbackCollector()
    
    # Initialize memory system with status callback
    memory_system = AgenticMemorySystem(
        llm_backend=DEFAULT_BACKEND,
        llm_model=DEFAULT_MODEL, 
        status_callback=status_collector.callback
    )
    
    # Test adding a memory note
    print("\n=== Testing Memory Creation ===")
    status_collector.clear()
    
    memory_id = memory_system.add_note("The user prefers dark themes in their applications.")
    
    # Verify callbacks were made
    messages = status_collector.get_messages()
    assert len(messages) > 0, "Status callbacks should have been called during memory creation"
    
    # Check for expected message types
    starting_messages = status_collector.get_messages_by_style("dim yellow")
    processing_messages = status_collector.get_messages_by_style("dim cyan")
    completion_messages = status_collector.get_messages_by_style("green")
    
    assert len(starting_messages) > 0, "Should have starting phase messages"
    assert len(completion_messages) > 0, "Should have completion messages"
    
    print(f"Captured {len(messages)} status messages:")
    for msg, duration, style in messages:
        print(f"  [{style}] {msg} ({duration}s)")


def test_consolidation_callback():
    """Test status callbacks during memory consolidation."""
    
    status_collector = StatusCallbackCollector()
    
    # Create system with low evolution threshold to trigger consolidation
    memory_system = AgenticMemorySystem(
        llm_backend=DEFAULT_BACKEND,
        llm_model=DEFAULT_MODEL,
        evo_threshold=2,  # Low threshold to trigger consolidation quickly
        status_callback=status_collector.callback
    )
    
    print("\n=== Testing Consolidation Callbacks ===")
    status_collector.clear()
    
    # Add memories to trigger consolidation
    memory_system.add_note("User likes coffee in the morning.")
    memory_system.add_note("User prefers tea in the afternoon.")
    
    messages = status_collector.get_messages()
    
    # Look for consolidation-related messages
    consolidation_messages = [msg for msg, _, _ in messages if "consolidat" in msg.lower()]
    
    if len(consolidation_messages) > 0:
        print("Consolidation callbacks detected:")
        for msg, duration, style in [(m, d, s) for m, d, s in messages if "consolidat" in m.lower()]:
            print(f"  [{style}] {msg} ({duration}s)")
    else:
        print("No consolidation triggered in this test run.")


def test_error_callback():
    """Test status callbacks during error conditions."""
    
    status_collector = StatusCallbackCollector()
    
    # Create system with invalid model to trigger errors
    memory_system = AgenticMemorySystem(
        llm_backend=DEFAULT_BACKEND,
        llm_model="invalid-model-name", 
        status_callback=status_collector.callback
    )
    
    print("\n=== Testing Error Callbacks ===")
    status_collector.clear()
    
    try:
        # This should trigger error callbacks
        memory_id = memory_system.add_note("This might cause an error.")
        
        # Check for error messages
        error_messages = status_collector.get_messages_by_style("red")
        
        if len(error_messages) > 0:
            print("Error callbacks detected:")
            for msg, duration, style in error_messages:
                print(f"  [{style}] {msg} ({duration}s)")
        else:
            print("No error callbacks captured (operation may have succeeded unexpectedly).")
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        # Check if error callbacks were still made
        error_messages = status_collector.get_messages_by_style("red")
        print(f"Error callbacks made: {len(error_messages)}")


def demo_callback_styles():
    """Demonstrate different callback styles available."""
    
    def demo_callback(message: str, duration: float = 2.0, style: str = "dim cyan"):
        print(f"[{style.upper()}] {message} (shows for {duration}s)")
    
    print("\n=== Available Status Callback Styles ===")
    demo_callback("Default processing messages", 2.0, "dim cyan")
    demo_callback("Starting/preparation phases", 2.0, "dim yellow") 
    demo_callback("Progress/completion phases", 2.0, "dim green")
    demo_callback("Success/final completion", 2.0, "green")
    demo_callback("Errors/failures", 2.0, "red")
    demo_callback("Information messages", 2.0, "blue")


if __name__ == "__main__":
    """Run status callback integration tests."""
    
    print("A-MEM Status Callback Integration Tests")
    print("=" * 50)
    
    # Demonstrate callback styles
    demo_callback_styles()
    
    # Run integration tests
    try:
        test_status_callback_integration()
        test_consolidation_callback() 
        test_error_callback()
        
        print("\n" + "=" * 50)
        print("Status callback integration tests completed!")
        print("\nTo integrate in your consuming application:")
        print("1. Define a status callback function with signature:")
        print("   status_callback(message: str, duration: float = 2.0, style: str = 'dim cyan')")
        print("2. Pass it when creating AgenticMemorySystem:")
        print("   memory_system = AgenticMemorySystem(status_callback=your_callback)")
        print("3. Use the provided styles for consistent UI feedback.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
