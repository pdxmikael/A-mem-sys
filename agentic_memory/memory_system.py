import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever
import json
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import time

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 session_id: Optional[str] = None):  
        """Initialize the memory system.
        
        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            session_id: Optional session ID for memory segregation. If None, generates a new session.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.memories = {}
        self.model_name = model_name
        # Initialize ChromaDB retriever
        # Note: We don't reset the collection to avoid tenant connection issues
        # Each test should use its own collection name or clean up properly
        self.retriever = ChromaRetriever(collection_name="memories",model_name=self.model_name)
        
        # Initialize LLM controller
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        
        # Initialize sentence transformer for similarity computation
        self.embedder = SentenceTransformer(model_name)
        
        # Load existing memories for this session
        self._load_session_memories()

        # Evolution system prompt
        self._evolution_system_prompt = '''
                            You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                            Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.
                            Make decisions about its evolution and whether it should be consolidated with existing memories.

                            The new memory context:
                            {context}
                            content: {content}
                            keywords: {keywords}

                            The nearest neighbors memories:
                            {nearest_neighbors_memories}

                            Semantic similarity scores between new memory and each neighbor:
                            {similarity_scores}
                            (Scores range from 0.0 to 1.0)

                            CONSOLIDATION RULES:
                            - Strongly consider consolidation if similarity score > 0.8
                            - Weakly consider consolidation if similarity score > 0.6
                            - Do not consolidate if similarity score < 0.6
                            - Consolidate only if memories are on the same topic and one memory is mostly a subset of the other
                            - When in doubt, DO NOT consolidate

                            Based on this information, determine:
                            1. Should this memory be evolved? Consider its relationships with other memories.
                            2. What specific actions should be taken?
                               - "strengthen": Create stronger connections between memories
                               - "update_neighbor": Update context and tags of neighboring memories
                               - "consolidate": Merge with an existing similar memory if appropriate
                               
                            For consolidation:
                            - Identify which neighbor memory to consolidate with by using the EXACT "memory id" from the neighbor list (e.g., if you see "memory id:abc123", use "abc123" as consolidate_with_id)
                            - Create a new consolidated content that integrates both memories intelligently
                            - The consolidated memory should be more complete and informative than either original
                            - Preserve important information from both memories
                                
                                OTHER ACTIONS:
                                - If choose to strengthen: which memory should it be connected to? Updated tags?
                                - If choose to update_neighbor: update context and tags of neighbors in sequential order
                                
                            Tags should be determined by the content characteristics for retrieval and categorization.
                            Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                            The number of neighbors is {neighbor_number}.
                                Tags should be determined by the content characteristics for retrieval and categorization.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor", "consolidate"],
                                    "consolidate_with_id": "memory_id_to_consolidate_with" or null,
                                    "consolidated_content": "new integrated content" or null,
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from response, handling markdown code blocks."""
        response = response.strip()
        
        # Check if response is wrapped in markdown code blocks
        if response.startswith('```json') and response.endswith('```'):
            # Extract content between ```json and ```
            lines = response.split('\n')
            # Remove first line (```json) and last line (```)
            json_lines = lines[1:-1]
            return '\n'.join(json_lines)
        elif response.startswith('```') and response.endswith('```'):
            # Extract content between ``` and ``` (generic code block)
            lines = response.split('\n')
            # Remove first line (```) and last line (```)
            json_lines = lines[1:-1]
            return '\n'.join(json_lines)
        else:
            # Return as-is if no markdown formatting
            return response
    



        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            # Use robust JSON extraction
            clean_response = self._extract_json_from_response(response)
            return json.loads(clean_response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note with evolution-based processing"""
        # Create MemoryNote 
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)
        
        # Process memory through evolution system (includes deduplication logic)
        evo_result, processed_note = self.process_memory(note)
        
        # DEBUG: Log what process_memory returned
        print(f"\n=== DEBUG: add_note process_memory result ===")
        print(f"evo_result type: {type(evo_result)}")
        print(f"evo_result value: {evo_result}")
        print(f"hasattr consolidated_id: {hasattr(evo_result, 'consolidated_id') if evo_result else False}")
        if hasattr(evo_result, 'consolidated_id'):
            print(f"consolidated_id: {evo_result.consolidated_id}")
        
        # Check if evolution system decided to consolidate with existing memory
        if hasattr(evo_result, 'consolidated_id'):
            # Return the ID of the consolidated memory instead of creating new one
            print(f"CONSOLIDATION: Returning consolidated memory ID: {evo_result.consolidated_id}")
            return evo_result.consolidated_id
        
        # Add new memory if not consolidated
        self.memories[processed_note.id] = processed_note
        
        # Add to ChromaDB with complete metadata including session_id
        metadata = {
            "id": processed_note.id,
            "content": processed_note.content,
            "keywords": processed_note.keywords,
            "links": processed_note.links,
            "retrieval_count": processed_note.retrieval_count,
            "timestamp": processed_note.timestamp,
            "last_accessed": processed_note.last_accessed,
            "context": processed_note.context,
            "evolution_history": processed_note.evolution_history,
            "category": processed_note.category,
            "tags": processed_note.tags,
            "session_id": self.session_id  # Add session namespace
        }
        self.retriever.add_document(processed_note.content, metadata, processed_note.id)
        
        if evo_result == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return processed_note.id
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        # Get current retriever configuration to preserve settings
        current_collection_name = getattr(self.retriever, 'collection_name', "memories")
        current_persist_dir = getattr(self.retriever, 'persist_directory', "./memory_db")
        
        # Reset ChromaDB collection with same configuration
        self.retriever = ChromaRetriever(collection_name=current_collection_name, model_name=self.model_name, persist_directory=current_persist_dir)
        
        # Re-add all memory documents with their complete metadata
        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags,
                "session_id": self.session_id  # Include session namespace for filtering
            }
            self.retriever.add_document(memory.content, metadata, memory.id)
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """Find related memories using ChromaDB retrieval with session-based filtering"""
        if not self.memories:
            return "", []
            
        try:
            # Get results from ChromaDB filtered by current session
            results = self.retriever.search(query, k, where={"session_id": self.session_id})
            
            # Convert to list of memories
            memory_str = ""
            indices = []
            
            if 'ids' in results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Only include memories that exist in self.memories (filter out stale ChromaDB entries)
                    if doc_id in self.memories and i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        # Format memory string with actual memory ID instead of index
                        memory_str += f"memory id:{doc_id}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        indices.append(doc_id)
                        
            return memory_str, indices
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""
            
        # Get results from ChromaDB filtered by current session
        results = self.retriever.search(query, k, where={"session_id": self.session_id})
        
        # Convert to list of memories
        memory_str = ""
        
        if 'ids' in results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    # Get metadata from ChromaDB results
                    metadata = results['metadatas'][0][i]
                    
                    # Add main memory info
                    memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                    
                    # Add linked memories if available
                    links = metadata.get('links', [])
                    j = 0
                    for link_id in links:
                        if link_id in self.memories and j < k:
                            neighbor = self.memories[link_id]
                            memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                            j += 1
                            
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        memory = self.memories.get(memory_id)
        if memory:
            # Increment retrieval count when memory is accessed
            memory.retrieval_count += 1
            # Update last accessed timestamp
            from datetime import datetime
            memory.last_accessed = datetime.now().isoformat()
        return memory
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in ChromaDB
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
            "session_id": self.session_id  # Add session namespace
        }
        
        # Delete and re-add to update
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from ChromaDB.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from ChromaDB
        """
        results = self.retriever.search(query, k)
        return [{'id': doc_id, 'score': score} 
                for doc_id, score in zip(results['ids'][0], results['distances'][0])]
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach."""
        # Get results from ChromaDB (only do this once) - filter by session_id
        search_results = self.retriever.search(query, k, where={"session_id": self.session_id})
        
        memories = []
        
        # Process ChromaDB results
        if 'ids' in search_results and search_results['ids'] and len(search_results['ids']) > 0:
            for i, doc_id in enumerate(search_results['ids'][0]):
                memory = self.memories.get(doc_id)
                if memory:
                    memories.append({
                        'id': doc_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': search_results['distances'][0][i]
                    })
        return memories[:k]
    
    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach.
        
        This method combines results from both:
        1. ChromaDB vector store (semantic similarity)
        2. Embedding-based retrieval (dense vectors)
        
        The results are deduplicated and ranked by relevance.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - score: Similarity score
                - metadata: Additional memory metadata
        """
        # Get results from ChromaDB
        chroma_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': chroma_results['distances'][0][i]
                })
                
        # Get results from embedding retriever
        embedding_results = self.retriever.search(query, k)
        
        # Combine results with deduplication
        seen_ids = set(m['id'] for m in memories)
        for result in embedding_results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append({
                        'id': memory_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': result.get('score', 0.0)
                    })
                    seen_ids.add(memory_id)
                    
        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using ChromaDB retrieval with session-based filtering."""
        if not self.memories:
            return []
            
        try:
            # Build where clause for session filtering
            where_clause = None
            if self.session_id:
                where_clause = {"session_id": self.session_id}
            
            # Get results from ChromaDB with session filtering
            results = self.retriever.search(query, k, where=where_clause)
            
            # Process results
            memories = []
            seen_ids = set()
            
            # Check if we have valid results
            if ('ids' not in results or not results['ids'] or 
                len(results['ids']) == 0 or len(results['ids'][0]) == 0):
                return []
                
            # Process ChromaDB results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Create result dictionary with all metadata fields
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': metadata.get('keywords', []),
                        'tags': metadata.get('tags', []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'is_neighbor': False
                    }
                    
                    # Add score if available
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                    
                    # Prioritize exact matches by adjusting score
                    if memory_dict['content'].strip().lower() == query.strip().lower():
                        memory_dict['score'] = -1.0  # Best possible score for exact matches
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories (neighbors)
            neighbor_count = 0
            for memory in list(memories):  # Use a copy to avoid modification during iteration
                if neighbor_count >= k:
                    break
                    
                # Get links from metadata
                links = memory.get('links', [])
                if not links and 'id' in memory:
                    # Try to get links from memory object
                    mem_obj = self.memories.get(memory['id'])
                    if mem_obj:
                        links = mem_obj.links
                        
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            
            return memories[:k]
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.
        
        Args:
            note: The memory note to process
            
        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note) or (consolidation_result, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note
            
        try:
            # Get nearest neighbors
            neighbors_text, indices = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not indices:
                return False, note
            
            # Compute semantic similarity scores with neighbors
            similarity_scores = []
            try:
                neighbor_contents = []
                for memory_id in indices:
                    if memory_id in self.memories:
                        neighbor_contents.append(self.memories[memory_id].content)
                
                if neighbor_contents:
                    # Encode the new memory content and all neighbor contents
                    all_contents = [note.content] + neighbor_contents
                    embeddings = self.embedder.encode(all_contents)
                    
                    # Calculate cosine similarity between new memory and each neighbor
                    for i in range(1, len(embeddings)):
                        similarity = cosine_similarity([embeddings[0]], [embeddings[i]])[0][0]
                        similarity_scores.append(f"Memory {indices[i-1]}: {similarity:.3f}")
            except Exception as e:
                logger.warning(f"Similarity computation failed: {str(e)}")
                similarity_scores = ["Similarity computation unavailable"]
                
            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(indices),
                similarity_scores="\n".join(similarity_scores)
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "consolidate_with_id": {
                                    "type": ["string", "null"]
                                },
                                "consolidated_content": {
                                    "type": ["string", "null"]
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )
                
                # DEBUG: Log the raw LLM response
                print(f"\n=== DEBUG: LLM Response ===")
                print(f"Raw response: {response}")
                
                response_json = json.loads(response)
                
                # DEBUG: Log parsed response
                print(f"Parsed JSON: {response_json}")
                print(f"Should evolve: {response_json.get('should_evolve')}")
                print(f"Actions: {response_json.get('actions')}")
                print(f"Consolidate with ID: {response_json.get('consolidate_with_id')}")
                print(f"Consolidated content: {response_json.get('consolidated_content')}")
                should_evolve = response_json["should_evolve"]
                actions = response_json.get("actions", [])
                
                # Check for consolidation action (independent of should_evolve)
                if "consolidate" in actions:
                    consolidate_with_id = response_json.get("consolidate_with_id")
                    consolidated_content = response_json.get("consolidated_content")
                    
                    if consolidate_with_id and consolidated_content:
                        # Create consolidation result object
                        class ConsolidationResult:
                            def __init__(self, target_id, content):
                                self.consolidated_id = target_id
                                self.consolidated_content = content
                        
                        # Update the target memory with consolidated content and metadata
                        if consolidate_with_id in self.memories:
                            target_memory = self.memories[consolidate_with_id]
                            target_memory.content = consolidated_content
                            
                            # Extract metadata from LLM response
                            tags_to_update = response_json.get("tags_to_update", [])
                            new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                            new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                            
                            # Merge keywords from multiple sources for comprehensive coverage
                            consolidated_keywords = set(target_memory.keywords)  # Start with existing keywords
                            consolidated_keywords.update(note.keywords)  # Add new memory's keywords
                            if tags_to_update:
                                consolidated_keywords.update(tags_to_update)  # Add LLM suggestions
                            if new_tags_neighborhood and len(new_tags_neighborhood) > 0 and len(new_tags_neighborhood[0]) > 0:
                                consolidated_keywords.update(new_tags_neighborhood[0])  # Add comprehensive keyword list
                            target_memory.keywords = list(consolidated_keywords)  # Convert back to list
                            
                            # Update context from LLM suggestions or keep existing
                            if new_context_neighborhood and len(new_context_neighborhood) > 0:
                                target_memory.context = new_context_neighborhood[0]
                            
                            # Merge tags from original memory, new memory, and LLM suggestions
                            consolidated_tags = set(target_memory.tags)  # Start with existing tags
                            consolidated_tags.update(note.tags)  # Add new memory's tags (CRUCIAL!)
                            if tags_to_update:
                                consolidated_tags.update(tags_to_update)  # Add LLM tag suggestions
                            if new_tags_neighborhood and len(new_tags_neighborhood) > 0 and len(new_tags_neighborhood[0]) > 0:
                                consolidated_tags.update(new_tags_neighborhood[0])  # Add all tags from LLM
                            target_memory.tags = list(consolidated_tags)  # Convert back to list
                            
                            # Update retriever with new metadata and add session_id
                            metadata = {
                                'content': target_memory.content,
                                'context': target_memory.context,
                                'keywords': target_memory.keywords,
                                'tags': target_memory.tags,
                                'timestamp': target_memory.timestamp,
                                'category': target_memory.category,
                                'session_id': self.session_id
                            }
                            self.retriever.add_document(target_memory.content, metadata, consolidate_with_id)
                            
                            return ConsolidationResult(consolidate_with_id, consolidated_content), note
                
                # Handle other actions (only if should_evolve is true)
                if should_evolve:
                    for action in actions:
                        if action == "strengthen":
                            suggest_connections = response_json["suggested_connections"]
                            new_tags = response_json["tags_to_update"]
                            note.links.extend(suggest_connections)
                            note.tags = new_tags
                        elif action == "update_neighbor":
                            new_context_neighborhood = response_json["new_context_neighborhood"]
                            new_tags_neighborhood = response_json["new_tags_neighborhood"]
                            
                            for i in range(min(len(indices), len(new_tags_neighborhood))):
                                # Skip if we don't have enough neighbors
                                if i >= len(indices):
                                    continue
                                    
                                # Get memory ID from indices (they are string IDs, not integer indices)
                                memory_id = indices[i]
                                if memory_id not in self.memories:
                                    continue
                                    
                                tag = new_tags_neighborhood[i]
                                if i < len(new_context_neighborhood):
                                    context = new_context_neighborhood[i]
                                else:
                                    context = self.memories[memory_id].context
                                        
                                # Update the memory directly using its ID
                                memory_to_update = self.memories[memory_id]
                                memory_to_update.tags = tag
                                memory_to_update.context = context
                                
                return should_evolve, note
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error in memory evolution: {str(e)}")
                return False, note
                
        except Exception as e:
            # For testing purposes, catch all exceptions and return the original note
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note
    
    def _load_session_memories(self):
        """Load all existing memories for the current session from ChromaDB."""
        try:
            # Query ChromaDB for all memories with this session_id
            # Note: This requires ChromaDB to have stored session_id in metadata
            results = self.retriever.collection.get(
                where={"session_id": self.session_id},
                include=["metadatas", "documents"]
            )
            
            if results and results.get('ids'):
                for i, memory_id in enumerate(results['ids']):
                    if i < len(results['metadatas']):
                        raw_metadata = results['metadatas'][i]
                        content = results['documents'][i] if i < len(results['documents']) else ""
                        
                        # Deserialize metadata - same logic as ChromaRetriever.search()
                        metadata = {}
                        for key, value in raw_metadata.items():
                            try:
                                # Try to parse JSON for lists and dicts
                                if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                    metadata[key] = json.loads(value)
                                # Convert numeric strings back to numbers
                                elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                    if '.' in value:
                                        metadata[key] = float(value)
                                    else:
                                        metadata[key] = int(value)
                                else:
                                    metadata[key] = value
                            except (json.JSONDecodeError, ValueError):
                                # If parsing fails, keep the original string
                                metadata[key] = value
                        
                        # Recreate MemoryNote from properly deserialized metadata
                        memory = MemoryNote(
                            content=content,
                            id=memory_id,
                            keywords=metadata.get('keywords', []),
                            links=metadata.get('links', []),
                            retrieval_count=metadata.get('retrieval_count', 0),
                            timestamp=metadata.get('timestamp', datetime.now().isoformat()),
                            last_accessed=metadata.get('last_accessed', datetime.now().isoformat()),
                            context=metadata.get('context', ''),
                            evolution_history=metadata.get('evolution_history', []),
                            category=metadata.get('category', ''),
                            tags=metadata.get('tags', [])
                        )
                        self.memories[memory_id] = memory
                        
                logger.info(f"Loaded {len(self.memories)} memories for session {self.session_id}")
        except Exception as e:
            logger.warning(f"Could not load session memories: {str(e)}")
            # Continue with empty memories if loading fails
