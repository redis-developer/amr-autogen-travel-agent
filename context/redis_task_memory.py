from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import redis

from auth.entra import get_redis_client  # single public helper

from autogen_ext.experimental.task_centric_memory import MemoryController as _BaseMemoryController
from autogen_ext.experimental.task_centric_memory._memory_bank import Memo
from autogen_ext.experimental.task_centric_memory.utils.page_logger import PageLogger

# RedisVL: schema, index & vectorizers
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, FilterQuery, CountQuery
from redisvl.utils.vectorize import BaseVectorizer


from .travel_prompter import TravelPrompter


class RedisMemoryBank:
    """
    A unified MemoryBank for TCM backed entirely by RedisVL index.
    
    Stores complete memo documents with vector embeddings for semantic search.
    Each document contains: memo_id, insight, task, topics, timestamp, and topic embedding.

    Public API (matches autogen's task-centric memory bank interface):
      - add_memo(insight_str, topics, task_str=None) -> None
      - add_task_with_solution(task, solution, topics) -> None
      - get_relevant_memos(topics) -> List[Memo]
      - contains_memos() -> bool
      - save_memos() -> None
      - reset() -> None
      - _reset_memos() -> None
      - _map_topics_to_memo(topics, memo_id, memo) -> None

    Notes:
      * All memo data stored in a single RedisVL index with vector embeddings
      * No separate Redis keys or sets needed - everything is in the index
      * Vector search handles topic-based retrieval directly
    """

    def __init__(
        self,
        *,
        namespace: str,
        vectorizer: BaseVectorizer, 
        redis_client: Optional[redis.Redis] = None,
        distance_threshold: float = 0.7,        # Match base class default
        n_results: int = 25,                    # Match base class default  
        relevance_conversion_threshold: float = 1.7,  # Restored from base class
        distance_metric: str = "cosine",       # "cosine" | "l2"
        algorithm: str = "hnsw",               # HNSW works well for demo scale
        reset: bool = False,
        logger: Optional[PageLogger] = None,
    ) -> None:
        self.namespace = namespace.strip().replace(" ", "_") or "default"
        self.prefix = f"memory:{self.namespace}"      # key prefix for memo docs
        self.index_name = self.prefix
        self.redis_client = redis_client
        self.distance_threshold = distance_threshold
        self.n_results = n_results
        self.relevance_conversion_threshold = relevance_conversion_threshold
        self.vectorizer = vectorizer
        self.logger = logger
        
        # Track last memo ID like base class - use Redis counter for persistence
        self.memo_counter_key = f"{self.prefix}:memo_counter"  

        schema = {
            "index": {
                "name": self.index_name,
                "prefix": self.prefix,
                "storage_type": "json",
            },
            "fields": [
                {"name": "memo_id", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": self.vectorizer.dims,
                        "algorithm": algorithm,
                        "distance_metric": distance_metric,
                        "datatype": "float32",
                    },
                },
            ],
        }
       
        self.index = SearchIndex.from_dict(schema, redis_client=self.redis_client)

        try:
            # idempotent create / reset
            self.index.create(overwrite=True if reset else False, drop=True if reset else False)
        except Exception as e:
            print(f"FT.INFO failed: {e}")
            return

        if reset:
            self.reset()

    def _get_next_memo_id(self) -> str:
        """Get next sequential memo ID, like base class."""
        return str(self.redis_client.incr(self.memo_counter_key))

    # ---- Core storage -------------------------------------------------------

    def add_memo(self, insight_str: str, topics: List[str], task_str: Optional[str] = None) -> None:
        """
        Adds an insight to the memory bank, given topics related to the insight, and optionally the task.
        Follows base class pattern exactly.
        """
        if self.logger:
            self.logger.enter_function()
        
        # Get sequential ID like base class
        memo_id = self._get_next_memo_id()
        
        # Create memo object like base class (simple constructor)
        memo = Memo(insight=insight_str, task=task_str)
        
        # Use _map_topics_to_memo like base class
        self._map_topics_to_memo(topics, memo_id, memo)
        
        if self.logger:
            self.logger.leave_function()

    def add_task_with_solution(self, task: str, solution: str, topics: List[str]) -> None:
        """
        Adds a task-solution pair to the memory bank, to be retrieved together later as a combined insight.
        This is useful when the insight is a demonstration of how to solve a given type of task.
        Follows base class pattern exactly.
        """
        if self.logger:
            self.logger.enter_function()
        
        # Get sequential ID like base class
        memo_id = self._get_next_memo_id()
        
        # Format insight like base class - prepend the task description for context
        insight_str = f"Example task:\n\n{task}\n\nExample solution:\n\n{solution}"
        memo = Memo(insight=insight_str, task=task)
        
        # Use _map_topics_to_memo like base class
        self._map_topics_to_memo(topics, memo_id, memo)
        
        if self.logger:
            self.logger.leave_function()

    def contains_memos(self) -> bool:
        """
        Returns True if the memory bank contains any memo.
        """
        return self.index.query(CountQuery("*"))

    # ---- Retrieval ----------------------------------------------------------

    def get_relevant_memos(self, topics: List[str]) -> List[Memo]:
        """
        Returns memos relevant to the input topics, aggregating similarity scores across topics.
        """
        if self.logger:
            self.logger.enter_function()

        if not topics:
            return []

        # Aggregate similarity scores by memo_id across all topics
        memo_scores: Dict[str, float] = {}
        
        for topic in topics:
            query = VectorQuery(
                vector=self.vectorizer.embed(topic),
                vector_field_name="embedding",
                num_results=self.n_results,
                return_fields=["memo_id", "vector_distance"],
            )
            
            for doc in self.index.query(query) or []:
                dist = float(doc.get("vector_distance", 1.0))
                if dist <= self.distance_threshold:
                    memo_id = doc["memo_id"]
                    similarity = (2 - dist) / 2
                    memo_scores[memo_id] = memo_scores.get(memo_id, 0) + similarity

        if self.logger:
            self.logger.info(f"\n{len(memo_scores)} POTENTIALLY RELEVANT MEMOS")

        # Return sorted memos with positive relevance scores
        memo_list = []
        for memo_id, score in sorted(memo_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= 0 and (memo := self.get_memo(memo_id)):
                memo.relevance = score
                memo_list.append(memo)
                
                if self.logger:
                    task_info = f"\n  TASK: {memo.task}\n" if memo.task else ""
                    self.logger.info(f"{task_info}\n  INSIGHT: {memo.insight}\n\n  RELEVANCE: {score:.3f}\n")

        if self.logger:
            self.logger.leave_function()
        return memo_list


    def get_memo(self, memo_id: str) -> Optional[Memo]:
        """Retrieve a specific memo by ID. Returns simple Memo like base class."""
        # Use a filter query to find the specific memo
        query = FilterQuery(
            return_fields=["$.insight", "$.task"],
            filter_expression=f"@memo_id:{{{memo_id}}}",
            num_results=1,
        )
        results = self.index.query(query)
        
        if not results:
            return None
            
        doc = results[0]
        # Create simple Memo like base class (just insight and task)
        return Memo(
            insight=doc.get("$.insight", ""),
            task=doc.get("$.task") or None,  # Convert empty string back to None
        )

    def list_memos(self, limit: int = 100) -> List[Memo]:
        """Return all memos. Base class doesn't sort, so we don't either."""
        # Get all memos (no filter, just return fields)
        query = FilterQuery(
            return_fields=["$.insight", "$.task"],
            filter_expression="*",
            num_results=limit,
        )
        results = self.index.query(query) or []
        
        out: List[Memo] = []
        for doc in results:
            memo = Memo(
                insight=doc.get("$.insight", ""),
                task=doc.get("$.task") or None,  # Convert empty string back to None
            )
            out.append(memo)
        
        return out

    def _reset_memos(self) -> None:
        """Internal method to reset memos - same as reset() for Redis implementation."""
        self.reset()

    def save_memos(self) -> None:
        """Save memos to persistent storage. No-op for Redis as it's already persistent."""
        pass

    def _map_topics_to_memo(self, topics: List[str], memo_id: str, memo: Memo) -> None:
        """
        Adds a mapping in the Redis index from each topic to the memo.
        This replaces the base class's string_map.add_input_output_pair() calls.
        """
        if self.logger:
            self.logger.enter_function()
            self.logger.info(f"\nINSIGHT\n{memo.insight}")
        
        # Create embedding from all topics combined (like we do in add_memo)
        topics_text = " ".join(topics) if topics else memo.insight
        emb = self.vectorizer.embed(topics_text)
        
        for topic in topics:
            if self.logger:
                self.logger.info(f"\n TOPIC = {topic}")
        
        # Store the memo in Redis index with vector embedding
        row = {
            "memo_id": memo_id,
            "insight": memo.insight,
            "task": memo.task or "",
            "topics": topics,
            "ts": int(time.time()),
            "embedding": emb,
        }
        self.index.load([row], id_field="memo_id")
        
        if self.logger:
            self.logger.leave_function()

    def reset(self) -> None:
        """
        Forces immediate deletion of all contents, in memory and on disk.
        Matches base class behavior.
        """
        self.index.create(overwrite=True, drop=True)
        # Reset memo counter like base class resets last_memo_id
        self.redis_client.delete(self.memo_counter_key)


# ---------------------------
# MemoryController that uses RedisMemoryBank
# ---------------------------
class RedisMemoryController(_BaseMemoryController):
    def __init__(
        self,
        reset: bool,
        client,
        namespace: str,
        *,
        logger: Optional[PageLogger] = None,
        config: Optional[dict] = None,
        redis_client: Optional[redis.Redis] = None,
        vectorizer: Optional[object] = None,
        task_assignment_callback=None,
    ) -> None:
        super().__init__(
            reset=reset,
            client=client,
            task_assignment_callback=task_assignment_callback,
            config=config,
            logger=logger,
        )

        # Recreate with Redis-backed bank using namespace for isolation
        self.memory_bank = RedisMemoryBank(
            namespace=namespace,
            redis_client=redis_client,
            vectorizer=vectorizer,
            reset=reset,
            logger=logger,
        )
        # Update the memory controller prompter to use our customized travel agent prompter
        self.prompter = TravelPrompter(self.client, self.logger)
