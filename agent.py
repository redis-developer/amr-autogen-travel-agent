import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncGenerator

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.experimental.task_centric_memory.utils import Teachability, PageLogger
from autogen_ext.models.openai import OpenAIChatCompletionClient
from redisvl.utils.vectorize import HFTextVectorizer
from tavily import TavilyClient

from config import AppConfig
from context.redis_chat_completion_context import RedisChatCompletionContext
from context.redis_task_memory import RedisMemoryController


# Constants
DEFAULT_USER_ID = "demo_user"
DEFAULT_BUFFER_SIZE = 10
DEFAULT_MAX_MEMOS = 4
DEFAULT_MAX_SEARCH_RESULTS = 5
LOG_LEVEL = "INFO"


@dataclass
class UserCtx:
    """User-specific context containing memory components and agent instances.
    
    Attributes:
        controller: Redis-backed memory controller for task-centric memory
        teachability: Memory adapter for long-term learning capabilities  
        supervisor: Main assistant agent with tools and memory integration
        logger: Page logger for debugging and monitoring
    """
    controller: RedisMemoryController
    teachability: Teachability
    supervisor: AssistantAgent
    logger: PageLogger


class TravelAgent:
    """Travel planning agent with per-user task-centric memory capabilities.
    
    This agent provides personalized travel planning services by maintaining
    separate memory contexts for each user. Each user gets their own memory
    controller, teachability adapter, and supervisor agent instance that are
    cached for performance.
    
    Features:
        - Per-user memory isolation using Redis namespaces
        - Long-term learning via task-centric memory
        - Web search integration for current travel information
        - Chat history management with configurable buffer sizes
        - Insight extraction and application for personalized recommendations
    
    Attributes:
        config: Application configuration containing API keys and model settings
        tavily_client: Web search client for travel information
        agent_model: OpenAI client for the main travel agent
        memory_model: OpenAI client for memory operations
        vectorizer: Text vectorizer for semantic similarity
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the TravelAgent with configuration and shared resources.
        
        Args:
            config: Application configuration. If None, loads default config.
        """
        if config is None:
            from config import get_config
            config = get_config()
        self.config = config

        # Set environment variables for SDK clients
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key

        # Initialize shared clients
        self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
        self.agent_model = OpenAIChatCompletionClient(
            model=config.travel_agent_model_name, 
            parallel_tool_calls=False
        )
        self.memory_model = OpenAIChatCompletionClient(model=config.memory_model_name)
        self.vectorizer = HFTextVectorizer()

        # User context registry for caching per-user components
        self._users: Dict[str, UserCtx] = {}

    # ------------------------------
    # User Context Management
    # ------------------------------
    
    def _get_or_create_user_ctx(self, user_id: str) -> UserCtx:
        """Get or create user-specific context with memory and agent components.
        
        Creates and caches a complete user context including memory controller,
        teachability adapter, chat history management, and supervisor agent.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            UserCtx: Complete user context with all components initialized
        """
        if user_id in self._users:
            return self._users[user_id]

        # Initialize user-specific logger
        logger = PageLogger(config={
            "level": LOG_LEVEL, 
            "path": f"./context/logs/{user_id}"
        })
        
        # Create Redis-backed memory controller with user namespace
        controller = RedisMemoryController(
            reset=False,  # Preserve insights across sessions
            client=self.memory_model,
            logger=logger,
            namespace=user_id,  # Isolate user data
            vectorizer=self.vectorizer,
            redis_url=self.config.redis_url,
            config={
                "generalize_task": True,
                "generate_topics": True,
                "validate_memos": True,
                "max_memos_to_retrieve": DEFAULT_MAX_MEMOS,
            },
        )
        
        # Create teachability adapter for long-term learning
        teachability = Teachability(controller, name=f"{user_id}_memory")
        
        # Initialize chat history management
        model_context = RedisChatCompletionContext(
            redis_url=self.config.redis_url,
            user_id=user_id,
            buffer_size=DEFAULT_BUFFER_SIZE
        )
        
        # Create supervisor agent with memory integration
        supervisor = self._create_supervisor_agent(
            model_context=model_context,
            memory_adapter=teachability
        )
        
        # Cache and return complete user context
        ctx = UserCtx(
            controller=controller,
            teachability=teachability,
            supervisor=supervisor,
            logger=logger
        )
        self._users[user_id] = ctx
        return ctx

    def _create_supervisor_agent(
        self, 
        model_context: RedisChatCompletionContext, 
        memory_adapter: Teachability
    ) -> AssistantAgent:
        """Create supervisor agent with memory integration and tools.
        
        Args:
            model_context: Redis-backed chat completion context for history
            memory_adapter: Teachability adapter for long-term memory
            
        Returns:
            AssistantAgent: Configured supervisor with memory and tools
        """
        return AssistantAgent(
            name="supervisor",
            model_client=self.agent_model,
            model_context=model_context,  # Chat history management
            memory=[memory_adapter],      # Long-term memory integration
            tools=[
                FunctionTool(
                    func=self.search_web,
                    description=(
                        "Search the web for current travel information: destinations, "
                        "flights, hotels, activities, events, transit, and prices."
                    )
                ),
            ],
            system_message=self._get_system_message(),
            max_tool_iterations=self.config.max_tool_iterations,
            model_client_stream=True,     # Enable token streaming
        )
    
    def _get_system_message(self) -> str:
        """Get the system message for the travel agent supervisor.
        
        Returns:
            str: Complete system message with role, responsibilities, and workflow
        """
        return (
            "You are an expert Travel Concierge AI that helps users plan and book "
            "complete trips from start to finish with the ability to recall user "
            "preferences across sessions.\n\n"
            
            "ROLE & RESPONSIBILITIES:\n"
            "- Help users discover destinations, plan itineraries, find accommodations, "
            "and organize travel logistics\n"
            "- Gather essential trip details: dates, budget, group size, preferences, "
            "and special requirements\n"
            "- Research current travel options, prices, availability, and recommendations\n"
            "- Create detailed day-by-day itineraries with specific times, locations, "
            "and booking information\n"
            "- Save learned user preferences and insights for personalized future "
            "recommendations\n\n"
            
            "WORKFLOW:\n"
            "1. DISCOVERY: Ask clarifying questions when necessary to get more useful details.\n"
            "2. RESEARCH: Use web search to find up-to-date options for flights, hotels, "
            "activities, restaurants, and transportation.\n"
            "3. PLANNING: Create a comprehensive itinerary with specific recommendations, "
            "times, and logistics\n\n"

            "Before answering, consider any 'Important insights' from the user that may "
            "be appended by long term memory and use them."
        )

    # -----------------
    # Tools
    # -----------------

    def search_web(self, query: str) -> Dict[str, Any]:
        """Search the web for current travel information and options.
        
        Args:
            query: Search query for travel-related information
            
        Returns:
            Dict[str, Any]: Search results from Tavily API containing relevant
                           travel information, bookings, and events
        """
        return self.tavily_client.search(
            query=query,
            topic="general",
            search_depth="basic",
            max_results=DEFAULT_MAX_SEARCH_RESULTS
        )

    # -----------------
    # Chat and Memory Interface  
    # -----------------
    
    async def _relevant_insights_for_task(
        self, 
        controller: RedisMemoryController, 
        task_text: str, 
        limit: int = DEFAULT_MAX_MEMOS
    ) -> List[str]:
        """Retrieve relevant insights for task transparency.
        
        Mirrors the insights that Teachability retrieves and applies, providing
        visibility into what memories are being used for the current task.
        
        Args:
            controller: Memory controller to retrieve insights from
            task_text: Current task/message text to find relevant insights for
            limit: Maximum number of insights to return
            
        Returns:
            List[str]: Unique relevant insights, deduplicated and limited
        """
        try:
            memos = await controller.retrieve_relevant_memos(task=task_text)
            unique_insights, seen = [], set()
            
            for memo in memos:
                insight = (memo.insight or "").strip()
                if insight and insight not in seen:
                    unique_insights.append(insight)
                    seen.add(insight)
                if len(unique_insights) >= limit:
                    break
                    
            return unique_insights
        except Exception as e:
            print(f"Error during memory retrieval: {e}", flush=True)
            return []

    async def stream_chat_turn(self, user_id: str, user_message: str) -> AsyncGenerator[str, None]:
        """
        Yield the growing assistant reply as tokens arrive from the streaming agent.
        
        Consumes ctx.supervisor.run_stream(...) and yields partial text as it builds up.
        Handles token chunks, tool calls, and final messages from the AutoGen stream.
        
        Args:
            user_id: User identifier for context isolation
            user_message: The user's input message
            
        Yields:
            str: Growing assistant response text as tokens arrive
        """
        ctx = self._get_or_create_user_ctx(user_id)
        stream = ctx.supervisor.run_stream(task=user_message)

        buffer = ""
        last_yielded = ""
        tool_feedback_active = False
        
        async for event in stream:
            event_name = event.__class__.__name__

            # Handle token chunks from streaming
            if event_name == "ModelClientStreamingChunkEvent":
                chunk = getattr(event, "content", "") or ""
                if chunk:
                    buffer += chunk
                    # Only yield if content actually changed
                    if buffer != last_yielded:
                        last_yielded = buffer
                        yield buffer

            # Handle tool calls with user feedback
            elif event_name == "FunctionCall":
                tool_name = getattr(event, "name", "tool")
                tool_message = buffer + f"\n\n🔧 Calling `{tool_name}`..."
                tool_feedback_active = True
                yield tool_message
                
            elif event_name == "FunctionExecutionResult":
                if tool_feedback_active:
                    completion_message = buffer + "\n\n✅ Tool finished."
                    tool_feedback_active = False
                    yield completion_message

            # Handle final LLM message (non-chunk) - only if we haven't been streaming
            elif event_name == "TextMessage" and getattr(event, "source", "") == "assistant":
                content = getattr(event, "content", "") or ""
                # Only use TextMessage content if we haven't been streaming chunks
                if not buffer and content:
                    buffer = content
                    if buffer != last_yielded:
                        last_yielded = buffer
                        yield buffer

    async def insights_for_task(self, user_id: str, task_text: str, limit: int = DEFAULT_MAX_MEMOS) -> List[str]:
        """
        Retrieve insights that were applied for this specific task.
        
        This mirrors what Teachability retrieves and applies internally, providing
        visibility into which memories influenced the agent's response.
        
        Args:
            user_id: User identifier for context isolation
            task_text: Current task/message text to find relevant insights for
            limit: Maximum number of insights to return
            
        Returns:
            List[str]: Relevant insights that were applied for this task
        """
        ctx = self._get_or_create_user_ctx(user_id)
        return await self._relevant_insights_for_task(ctx.controller, task_text, limit)

    async def chat(self, user_message: str, user_id: Optional[str] = None) -> str:
        """Process a chat message and return the agent's response.
        
        Handles a single conversation turn for the specified user, applying
        relevant insights from long-term memory and returning the response.
        
        Args:
            user_message: The user's input message
            user_id: User identifier. Defaults to DEFAULT_USER_ID if None
            
        Returns:
            str: The agent's response to the user message
        """
        uid = user_id or DEFAULT_USER_ID
        ctx = self._get_or_create_user_ctx(uid)

        # Process message through supervisor agent
        stream = ctx.supervisor.run_stream(task=user_message)
        result = await Console(stream)

        # Display applied insights for transparency
        insights = await self._relevant_insights_for_task(
            ctx.controller, 
            user_message, 
            limit=DEFAULT_MAX_MEMOS
        )
        
        if insights:
            print(f"\n\n--- INSIGHTS APPLIED (user: {uid}) ---", flush=True)
            for insight in insights:
                print(f"• {insight}")
            print("--- END INSIGHTS ---\n", flush=True)
        else:
            print("--- NO INSIGHTS EXTRACTED ---", flush=True)

        # Extract and return response content
        if hasattr(result, "messages") and result.messages:
            last_message = result.messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content
                
        return "I'm sorry, I couldn't process that. Please try again."
    
    # -----------------
    # Utility Methods
    # -----------------

    async def get_chat_history(self, user_id: str, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve chat history for a user from Redis storage.
        
        Converts internal message objects to Gradio-compatible format,
        filtering for user and assistant messages with text content.
        
        Args:
            user_id: User identifier to get history for
            n: Number of messages to retrieve. If None, uses buffer_size.
               If -1, retrieves all messages.
            
        Returns:
            List[Dict[str, str]]: Message dictionaries with 'role' and 'content' 
                                keys suitable for Gradio chat interface
        """
        if user_id not in self._users:
            return []
        
        ctx = self._users[user_id]
        model_context = ctx.supervisor.model_context
        
        try:
            messages = await model_context.get_messages(n=n)
            gradio_messages = []
            
            for msg in messages:
                msg_type = msg.__class__.__name__
                
                # Process user messages with text content
                if msg_type == 'UserMessage':
                    if (hasattr(msg, 'content') and msg.content and 
                        isinstance(msg.content, str)):
                        gradio_messages.append({
                            'role': 'user',
                            'content': msg.content
                        })
                        
                # Process assistant messages with text content (skip tool calls)
                elif msg_type == 'AssistantMessage':
                    if (hasattr(msg, 'content') and msg.content and 
                        isinstance(msg.content, str)):
                        gradio_messages.append({
                            'role': 'assistant', 
                            'content': msg.content
                        })
                        
            return gradio_messages
            
        except Exception as e:
            print(f"Error retrieving chat history for user {user_id}: {e}")
            return []

    def get_user_list(self) -> List[str]:
        """Get list of all active user IDs.
        
        Returns:
            List[str]: List of user IDs that have been created and cached
        """
        return list(self._users.keys())

    def user_exists(self, user_id: str) -> bool:
        """Check if a user context exists in the cache.
        
        Args:
            user_id: User identifier to check
            
        Returns:
            bool: True if user context exists, False otherwise
        """
        return user_id in self._users

    def reset_user_memory(self, user_id: str) -> None:
        """Reset a user's memory by removing their cached context.
        
        This clears the user's insight bank and forces recreation of
        a fresh memory controller on next interaction.
        
        Args:
            user_id: User identifier whose memory should be reset
        """
        if user_id in self._users:
            self._users.pop(user_id, None)
    