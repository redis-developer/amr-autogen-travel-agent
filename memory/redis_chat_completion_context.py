import json
import os
from typing import Any, List, Mapping, Optional

from redis import Redis
from pydantic import BaseModel, Field
from typing_extensions import Self

from autogen_core._component_config import Component
from autogen_core.models import FunctionExecutionResultMessage, LLMMessage
from autogen_core.model_context._chat_completion_context import ChatCompletionContext, ChatCompletionContextState
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
    FunctionExecutionResultMessage,
)

class RedisChatCompletionContextConfig(BaseModel):
    buffer_size: int = Field(default=6, description="Maximum number of messages to keep in buffer")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_key_prefix: str = Field(default="chat_history", description="Redis key prefix for storing messages")
    user_id: str = Field(description="Unique identifier for this chat context")
    initial_messages: List[LLMMessage] | None = None


class RedisChatCompletionContext(ChatCompletionContext, Component[RedisChatCompletionContextConfig]):
    """A Redis-backed chat completion context that stores messages in Redis and maintains
    a buffer of the last n messages, where n is the buffer size.

    This implementation stores all messages persistently in Redis while providing
    efficient access to the most recent messages based on the buffer size.

    Args:
        buffer_size (int): The maximum number of messages to keep in the buffer.
        redis_url (str): Redis connection URL.
        redis_key_prefix (str): Prefix for Redis keys.
        user_id (str): Unique identifier for this chat context.
        initial_messages (List[LLMMessage] | None): The initial messages.
    """

    component_config_schema = RedisChatCompletionContextConfig

    def __init__(
        self,
        buffer_size: int = 6,
        redis_url: str = "redis://localhost:6379",
        redis_key_prefix: str = "chat_history",
        user_id: str = "default",
        initial_messages: List[LLMMessage] | None = None,
    ) -> None:
        # super().__init__(initial_messages)
        if buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0.")
        
        self._buffer_size = buffer_size
        self._redis_url = redis_url
        self._redis_key_prefix = redis_key_prefix
        self._user_id = user_id
        
        # Create Redis client
        self._redis_client = Redis.from_url(redis_url, decode_responses=True)
        
        # Redis keys for this context
        self._messages_key = f"{redis_key_prefix}:{user_id}:messages"
        
        # Initialize Redis with initial messages if provided
        if initial_messages:
            self._initialize_redis_with_messages(initial_messages)

    def _initialize_redis_with_messages(self, messages: List[LLMMessage]) -> None:
        """Initialize Redis with the provided messages."""
        # Clear existing messages
        self._redis_client.delete(self._messages_key)
        
        # Add initial messages
        for message in messages:
            self._add_message_to_redis(message)

    def _add_message_to_redis(self, message: LLMMessage) -> None:
        """Add a message to Redis with proper ordering."""        
        # Serialize message with custom handling for complex objects
        serialized_data = self._serialize_message(message)
        # Add to Redis list (LPUSH for newest first)
        self._redis_client.lpush(self._messages_key, serialized_data)

    def _serialize_message(self, message: LLMMessage) -> str:
        """Serialize a message using Pydantic model_dump if available, fallback to __dict__."""
        return json.dumps(message.model_dump())


    def _deserialize_message(self, message_data: str) -> LLMMessage:
        """Deserialize a message from Redis storage."""
        data = json.loads(message_data)
        
        message_classes = {
            "AssistantMessage": AssistantMessage,
            "SystemMessage": SystemMessage,
            "UserMessage": UserMessage,
            "FunctionExecutionResultMessage": FunctionExecutionResultMessage,
        }
        
        # Get message type (works for both Pydantic model_dump and __dict__ formats)
        message_type = data.get("type")
        if not message_type:
            raise ValueError("Message data missing 'type' field")
        
        if message_type not in message_classes:
            raise ValueError(f"Unknown message type: {message_type}")
    
        return message_classes[message_type](**data)

    async def add_message(self, message: LLMMessage) -> None:
        """Add a message to the Redis-backed context."""
        self._add_message_to_redis(message)

    async def get_messages(self, n: Optional[int] = None) -> List[LLMMessage]:
        """Get at most `n` recent messages from Redis.
        
        Args:
            n (Optional[int]): Number of messages to retrieve. If None, uses buffer_size.
                              If -1, retrieves all messages.
        
        Returns:
            List[LLMMessage]: List of messages in chronological order (oldest to newest).
        """
        # Determine how many messages to retrieve
        if n is None:
            n = self._buffer_size
        
        # Get messages from Redis (LRANGE gets from newest to oldest)
        if n == -1:
            # Get all messages
            raw_messages = self._redis_client.lrange(self._messages_key, 0, -1)
        else:
            # Get the most recent n messages
            raw_messages = self._redis_client.lrange(self._messages_key, 0, n - 1)
        
        if not raw_messages:
            return []
        
        # Deserialize messages and reverse to get chronological order
        messages = []
        for raw_message in reversed(raw_messages):
            try:
                message = self._deserialize_message(raw_message)
                messages.append(message)
            except Exception as e:
                # Log error but continue processing other messages
                print(f"Error deserializing message: {e}")
                continue
        
        # Handle the case where first message is a function call result message
        if messages and isinstance(messages[0], FunctionExecutionResultMessage):
            # Remove the first message from the list
            messages = messages[1:]
        
        return messages
    
    @property
    def _messages(self) -> List[LLMMessage]:
        """Grab all messages from Redis in chronological order (oldest to newest)."""
        # This is a synchronous property, so we need to call the Redis operations directly
        # Get all messages from Redis (LRANGE gets from newest to oldest)
        raw_messages = self._redis_client.lrange(self._messages_key, 0, -1)
        
        if not raw_messages:
            return []
        
        # Deserialize messages and reverse to get chronological order
        messages = []
        for raw_message in reversed(raw_messages):
            try:
                message = self._deserialize_message(raw_message)
                messages.append(message)
            except Exception as e:
                # Log error but continue processing other messages
                print(f"Error deserializing message: {e}")
                continue
        
        # Handle the case where first message is a function call result message
        if messages and isinstance(messages[0], FunctionExecutionResultMessage):
            # Remove the first message from the list
            messages = messages[1:]
        
        return messages
    
    @_messages.setter
    def _messages(self, messages: List[LLMMessage]) -> None:
        """Set messages by clearing Redis and reinitializing with new messages.
        
        This setter is required because the base class methods like clear() and load_state()
        assign directly to self._messages.
        """
        # Clear existing messages in Redis
        self._redis_client.delete(self._messages_key)
        
        # Add all new messages to Redis
        if messages:
            self._initialize_redis_with_messages(messages)

    async def clear(self) -> None:
        """Clear all messages from the context and Redis."""
        self._redis_client.delete(self._messages_key)


    def __del__(self):
        """Cleanup Redis connection when instance is destroyed."""
        if hasattr(self, '_redis_client'):
            try:
                self._redis_client.close()
            except Exception:
                pass  # Ignore errors during cleanup
