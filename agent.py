import os
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    MemoryQueryEvent,
    ModelClientStreamingChunkEvent,
    ThoughtEvent,
    UserInputRequestedEvent,
    CodeGenerationEvent,
    CodeExecutionEvent,
    SelectSpeakerEvent,
    ToolCallSummaryMessage,
    MultiModalMessage,
    HandoffMessage,
    StopMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.experimental.task_centric_memory.utils import Teachability, PageLogger
from autogen_ext.models.openai import OpenAIChatCompletionClient
from redisvl.utils.vectorize import HFTextVectorizer
import redis
from tavily import TavilyClient

from config import AppConfig
from context.redis_chat_completion_context import RedisChatCompletionContext
from context.redis_task_memory import RedisMemoryController


# Constants
DEFAULT_BUFFER_SIZE = 10
DEFAULT_MAX_MEMOS = 4
DEFAULT_MAX_SEARCH_RESULTS = 8
LOG_LEVEL = "INFO"


@dataclass
class UserCtx:
    """User-specific context containing memory components and agent instances.
    
    Attributes:
        controller: Redis-backed memory controller for task-centric memory
        teachability: Memory adapter for long-term learning capabilities  
        agent: Main assistant agent with tools and memory integration
        logger: Page logger for debugging and monitoring
    """
    controller: RedisMemoryController
    teachability: Teachability
    agent: AssistantAgent
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

        # Initialize seed users and their memories
        self._user_ctx_cache = {}
        self._init_seed_users()

    # ------------------------------
    # User Context Management
    # ------------------------------
    
    def _get_or_create_user_ctx(self, user_id: str) -> UserCtx:
        """Get or create user-specific context with memory and agent components.
        
        Creates and caches a complete user context including memory controller,
        teachability adapter, chat history management, and supervisor agent.
        Also handles preseeding user memories if they don't exist yet.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            UserCtx: Complete user context with all components initialized
        """
        if user_ctx := self._user_ctx_cache.get(user_id):
            return user_ctx

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
        agent = self._create_agent(
            model_context=model_context,
            memory_adapter=teachability
        )
        
        # Cache and return complete user context
        user_ctx = UserCtx(
            controller=controller,
            teachability=teachability,
            agent=agent,
            logger=logger
        )
        self._user_ctx_cache[user_id] = user_ctx

        return user_ctx

    def _load_seed_data(self) -> Dict[str, Any]:
        """Load seed data from JSON file."""
        seed_file = Path(__file__).parent / "context" / "seed.json"
        try:
            with open(seed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _init_seed_users(self) -> None:
        """Simple seed init: load seed.json, create contexts, add memos."""
        seed_data = self._load_seed_data()
        user_memories = seed_data.get("user_memories", {})
        
        for user_id, memories in user_memories.items():
            ctx = self._get_or_create_user_ctx(str(user_id))
            if ctx.controller.memory_bank.contains_memos():
                print(f"⏭️  Skipping seed for user {user_id} - memory bank already contains memos")
            else:
                for memo in memories:
                    ctx.controller.memory_bank.add_memo(
                        insight_str=memo["insight"],
                        topics=memo["topics"],
                        task_str=memo.get("task")
                    )
                print(f"✅ Seeded {len(memories)} memories for user: {user_id}")

    def _create_agent(
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
            name="agent",
            model_client=self.agent_model,
            model_context=model_context,  # Chat history management
            memory=[memory_adapter],      # Long-term memory integration
            tools=[
                FunctionTool(
                    func=self.search_logistics,
                    description=(
                        "Time-aware logistics search ONLY: flights, hotels, and intercity/local transport. "
                        "Use for availability, schedules, prices, carriers/properties, or routes. "
                        "Arguments: query (required), start_date (optional, YYYY-MM-DD), end_date (optional, YYYY-MM-DD). "
                        "Always include dates when the user mentions a travel window; if ambiguous, ask for dates before booking guidance. "
                        "NEVER use this for activities, attractions, neighborhoods, or dining. "
                        "Results are restricted to reputable flight/hotel/transport sources; top URLs are deeply extracted."
                    )
                ),
                FunctionTool(
                    func=self.search_general,
                    description=(
                        "Time-aware destination research: activities, attractions, neighborhoods, dining, events, local tips. "
                        "Use for up-to-date things to do, cultural context, and planning inspiration. "
                        "Arguments: query (required), start_date (optional, YYYY-MM-DD), end_date (optional, YYYY-MM-DD). "
                        "Scope searches to the relevant season/year when possible and prefer recent sources. "
                        "NEVER use this for flights, hotels, or transport logistics. "
                        "Example: 'things to do in Lisbon in June 2026'."
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
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return (
            f"You are an expert, time-aware Travel Concierge AI. Today is {today} (UTC). "
            "Assume your training data may be outdated; for anything time-sensitive, verify with tools.\n\n"
            "ROLE:\n"
            "- Discover destinations, plan itineraries, recommend accommodations, and organize logistics.\n"
            "- Research current options, prices, availability, and on-the-ground activities.\n"
            "- Produce clear, actionable itineraries and booking guidance.\n"
            "- Regardless of your prior knowledge, always use search tools for current or future-state information.\n\n"
            "TOOLING POLICY (TIME AWARENESS):\n"
            "- Use search_logistics ONLY for flights, hotels, or transport. Include start_date/end_date (YYYY-MM-DD) when known.\n"
            "- Use search_general for activities, attractions, neighborhoods, dining, events, or local tips. Include dates when relevant.\n"
            "- Prefer recent sources (past 12–24 months) and pass explicit dates to tools whenever the user provides a time window.\n"
            "- If dates are ambiguous (e.g., 'this spring'), ask for clarification before booking-critical steps.\n\n"
            "DISCOVERY:\n"
            "- If missing details, ask targeted questions (exact dates or window, origin/destination, budget, party size, interests,\n"
            "  lodging preferences, accessibility, loyalty programs).\n\n"
            "OUTPUT STYLE:\n"
            "- Be concise and prescriptive with your suggestions, followups, and recommendations.\n"
            "- Seek to be the best and friendliest travel agent possible. You are the expert after all.\n"
            "- Cite sources with titles and URLs for any tool-based claim.\n"
            "- Normalize to a single currency if prices appear; state assumptions.\n"
            "- For itineraries, list day-by-day with times and logistics.\n\n"
            "MEMORY:\n"
            "- Consider any appended Important insights (long-term memory) before answering and adapt to them."
        )

    # -----------------
    # Tools
    # -----------------

    def search_logistics(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """✈️🏨🚆 Logistics search: flights, hotels, and transport only.

        What it is for
        - Airfare and airline schedules, hotels/stays, and intercity transport (train/bus/ferry/car rental).

        How to use
        - Provide a concise query that includes the route or destination and constraints, e.g.:
          "JFK to LHR, nonstop preferred" or "hotels in Kyoto near Gion, mid-range" or "train Paris to Amsterdam".
        - Optionally include start_date and end_date as YYYY-MM-DD strings to guide availability windows.

        Behavior
        - Restricts sources to reputable flight/hotel/transport providers and aggregators.
        - Returns the strongest matches first and deeply extracts the top URLs for rich context.
        """
        include_domains = [
            # Flights / OTAs
            "expedia.com", "kayak.com", "travel.google.com",
            # Hotels / stays
            "booking.com", "hotels.com",
        ]

        # Build search kwargs
        date_hint = None
        if start_date and end_date:
            date_hint = f" travel dates {start_date} to {end_date}"
        elif start_date:
            date_hint = f" travel date on/after {start_date}"
        elif end_date:
            date_hint = f" travel date on/before {end_date}"

        augmented_query = (query.strip() + (date_hint or "")).strip()

        search_kwargs = {
            "query": augmented_query,
            "topic": "general",
            "search_depth": "advanced",
            "include_raw_content": True,
            "include_domains": include_domains,
            "max_results": DEFAULT_MAX_SEARCH_RESULTS,
        }

        results = self.tavily_client.search(**search_kwargs)

        # Sort by score descending and filter out low-quality results
        all_results = results.get("results", [])
        sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
        trimmed = [r for r in sorted_results if r.get("score", 0) > 0.2]
        results["results"] = trimmed

        # Extract top 2 URLs for deeper context
        top_urls = [r.get("url") for r in trimmed[:2] if r.get("url")]
        extractions: List[Dict[str, Any]] = []
        if top_urls:
            extracted = self.tavily_client.extract(urls=top_urls)
            if isinstance(extracted, dict) and extracted.get("results"):
                extractions = extracted.get("results", [])
            elif isinstance(extracted, list):
                extractions = extracted

        results["extractions"] = extractions

        # print("\nLOGISTICS SEARCH RESULTS", flush=True)
        # print("QUERY:", query, flush=True)
        # print("RESULTS:", results, flush=True)
        return results

    def search_general(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """📍 General destination research: activities, attractions, neighborhoods, dining, events.

        What it is for
        - Up-to-date things to do, local highlights, neighborhoods to stay, dining ideas, and cultural context.

        How to use
        - Provide a destination/time-focused query, e.g., "things to do in Lisbon in June",
          "Barcelona food tours", "best neighborhoods to stay in Tokyo".

        Behavior
        - Runs an open web search (no logistics domains restriction) with raw content for context.
        - Optionally scope the search by start_date/end_date (YYYY-MM-DD) for time-relevant results.
        """
        date_hint = None
        if start_date and end_date:
            date_hint = f" for {start_date} to {end_date}"
        elif start_date:
            date_hint = f" for {start_date}"
        elif end_date:
            date_hint = f" up to {end_date}"

        augmented_query = (query.strip() + (date_hint or "")).strip()

        search_kwargs = {
            "query": augmented_query,
            "topic": "general",
            "search_depth": "advanced",
            "include_raw_content": True,
            "max_results": DEFAULT_MAX_SEARCH_RESULTS,
        }
        results = self.tavily_client.search(**search_kwargs)
        results["results"] = [r for r in results.get("results", []) if r.get("score", 0) > 0.2]

        # print("\nGENERAL SEARCH RESULTS", flush=True)
        # print("QUERY:", query, flush=True)
        # print("RESULTS:", results, flush=True)
        return results

    # -----------------
    # Chat and Memory Interface  
    # -----------------
    

    async def stream_chat_turn_with_events(self, user_id: str, user_message: str) -> AsyncGenerator[tuple[str, dict | None], None]:
        """
        Yield (growing assistant reply, normalized event | None) pairs as the agent streams.
        
        Emits a small set of meaningful events suitable for UI display while
        coalescing token chunk noise:
          - user_message_submitted (when user message is processed)
          - llm_token_stream_start (first token chunk)
          - tool_call (FunctionCall)
          - tool_result (FunctionExecutionResult)
          - llm_message_complete (final assistant text)
        """
        ctx = self._get_or_create_user_ctx(user_id)
        
        def _html(icon: str, title: str, message: str) -> str:
            safe_icon = icon or ""
            safe_title = title or ""
            safe_msg = message or ""
            return (
                f"<div class='event-card'>"
                f"<div class='event-title'>{safe_icon} {safe_title}</div>"
                f"<div class='event-message'>{safe_msg}</div>"
                f"</div>"
            )

        def _event(event_type: str, icon: str, title: str, message: str) -> dict:
            return {
                "type": event_type,
                "html": _html(icon, title, message),
            }
        
        stream = ctx.agent.run_stream(task=user_message)

        buffer = ""
        last_yielded = ""
        emitted_stream_start = False
        llm_call_index = 0

        async for event in stream:
            event_name = event.__class__.__name__

            def _get_model_name(ev) -> Optional[str]:
                # Try multiple places for model name; fall back to supervisor's model
                val = getattr(ev, "model", None)
                if isinstance(val, str) and val:
                    return val
                meta = getattr(ev, "metadata", None)
                if isinstance(meta, dict) and meta.get("model"):
                    return meta.get("model")
                if hasattr(meta, "model") and getattr(meta, "model"):
                    return getattr(meta, "model")
                sup_client = getattr(ctx.agent, "model_client", None)
                if sup_client is not None:
                    return getattr(sup_client, "model", None)
                return None

            if isinstance(event, ModelClientStreamingChunkEvent):
                chunk = getattr(event, "content", "") or ""
                if chunk:
                    buffer += chunk
                    if not emitted_stream_start:
                        emitted_stream_start = True
                        llm_call_index += 1
                        yield buffer, _event(
                            "llm_token_stream_start",
                            "⏳",
                            f"LLM #{llm_call_index}: streaming",
                            f"Model: {_get_model_name(event) or ''}",
                        )
                    if buffer != last_yielded:
                        last_yielded = buffer
                        # No event for regular token chunks to reduce UI churn
                        yield buffer + ' <span class="thinking-animation">●●●</span>', None

            elif isinstance(event, ToolCallRequestEvent):
                # content: List[FunctionCall]
                calls = getattr(event, "content", None) or []
                tool_names = []
                tool_args = None
                try:
                    for c in calls:
                        if tool_args is None:
                            tool_args = getattr(c, "arguments", None)
                        name = getattr(c, "name", None)
                        if name:
                            tool_names.append(name)
                except Exception:
                    pass
                tool_name = ", ".join(tool_names) or "tool"
                icon = "🔧"
                title = f"Calling {tool_name}"
                if tool_name == "search_logistics":
                    icon = "✈️"
                    title = "Searching logistics"
                elif tool_name == "search_general":
                    icon = "📍"
                    title = "Searching (general)"
                yield buffer, _event(
                    "tool_call",
                    icon,
                    title,
                    f"Invoking {tool_name}",
                )

            elif event_name == "ModelClientRequestEvent":
                # Any model request (covers non-streaming LLM calls like memory/tool)
                llm_call_index += 1
                model_name = _get_model_name(event)
                yield buffer + ' <span class="thinking-animation">● ● ●</span>', _event(
                    "llm_request",
                    "📤",
                    f"LLM #{llm_call_index}: request",
                    f"Model: {model_name or ''}",
                )

            elif event_name in ("ModelClientResponseEvent", "ModelClientStreamEndEvent", "ModelClientResponseDoneEvent"):
                model_name = _get_model_name(event)
                yield buffer, _event(
                    "llm_response_received",
                    "📥",
                    f"LLM #{llm_call_index}: response received",
                    f"Model: {model_name or 'OpenAI'}",
                )



            elif isinstance(event, ToolCallRequestEvent):
                # content: List[FunctionCall] with name, arguments, id
                calls = getattr(event, "content", None) or []
                call_infos = []
                try:
                    for c in calls:
                        call_infos.append({
                            "name": getattr(c, "name", None),
                            "id": getattr(c, "id", None),
                            "arguments": getattr(c, "arguments", None),
                        })
                except Exception:
                    pass
                # Choose icon/title based on first call
                first_name = next((ci.get("name") for ci in call_infos if ci.get("name")), None)
                icon = "🛠️"
                title = "Tool requested"
                if first_name == "search_logistics":
                    icon = "✈️"
                    title = "Logistics search requested"
                elif first_name == "search_general":
                    icon = "📍"
                    title = "General search requested"
                yield buffer, _event(
                    "tool_request",
                    icon,
                    title,
                    ", ".join([str(ci.get("name")) for ci in call_infos if ci.get("name")]) or "Tool call",
                )

            elif isinstance(event, ToolCallExecutionEvent):
                # content: List[FunctionExecutionResult] with name, call_id, content, is_error
                results = getattr(event, "content", None) or []
                tool_name = "Tool"
                try:
                    if results:
                        first_result = results[0]
                        tool_name = getattr(first_result, "name", None) or "Tool"
                except Exception:
                    pass
                
                # Set icon based on tool name
                icon = "✅"
                if tool_name == "search_logistics":
                    icon = "✈️"
                elif tool_name == "search_general":
                    icon = "📍"
                
                yield buffer, _event(
                    "tool_result",
                    icon,
                    f"{tool_name} finished",
                    "Tool execution completed",
                )

            elif isinstance(event, MemoryQueryEvent):
                # content: List[MemoryContent]
                mems = getattr(event, "content", None) or []
                insights = []
                try:
                    for m in mems:
                        insights.append(getattr(m, "content", None) or str(m))
                except Exception:
                    pass
                yield buffer, _event(
                    "memory_injected",
                    "🧠",
                    "Insights applied",
                    "\n".join([f"• {i}" for i in insights if i]) or "Memory context injected",
                )

            elif isinstance(event, UserInputRequestedEvent):
                yield buffer, _event(
                    "user_input_requested",
                    "⌛",
                    "Awaiting input",
                    "Agent requested user input",
                )

            elif isinstance(event, ThoughtEvent):
                yield buffer, _event(
                    "thought",
                    "💭",
                    "Agent thought",
                    getattr(event, "content", "") or "",
                )

            elif isinstance(event, SelectSpeakerEvent):
                speakers = getattr(event, "content", None) or []
                try:
                    speakers = list(speakers)
                except Exception:
                    speakers = [str(speakers)] if speakers else []
                yield buffer, _event(
                    "select_speaker",
                    "🎙️",
                    "Speaker selected",
                    ", ".join([str(s) for s in speakers]) or "",
                )

            elif isinstance(event, CodeGenerationEvent):
                msg = getattr(event, "content", "") or ""
                blocks = getattr(event, "code_blocks", None)
                yield buffer, _event(
                    "code_generated",
                    "🧩",
                    "Code generated",
                    msg,
                )

            elif isinstance(event, CodeExecutionEvent):
                result = getattr(event, "result", None)
                output = None
                exit_code = None
                if result is not None:
                    output = getattr(result, "output", None)
                    exit_code = getattr(result, "exit_code", None)
                yield buffer, _event(
                    "code_executed",
                    "▶️",
                    "Code executed",
                    (output[:500] + "…") if isinstance(output, str) and len(output) > 500 else (output or ""),
                )

            elif isinstance(event, (ToolCallSummaryMessage, MultiModalMessage, HandoffMessage, StopMessage)):
                # General message-like events
                summary_content = getattr(event, "content", "") or ""
                yield buffer, _event(
                    event.__class__.__name__,
                    "ℹ️",
                    event.__class__.__name__,
                    summary_content,
                )

            elif isinstance(event, TextMessage):
                source = getattr(event, "source", "") or ""
                content = getattr(event, "content", "") or ""
                icon = "👤" if source == "user" else "🤖"
                title = f"{source.title()} message" if source else "Message"
                yield buffer, _event(
                    "text_message",
                    icon,
                    title,
                    "",  # Keep message empty for concise logs
                )

            elif hasattr(event, "messages") and hasattr(event, "stop_reason"):
                # Final TaskResult
                yield buffer, _event(
                    "run_complete",
                    "🏁",
                    "Run complete",
                    f"Stop reason: {getattr(event, 'stop_reason', '')}",
                )

            else:
                # Catch-all for unknown events
                yield buffer + ' <span class="thinking-animation">● ● ●</span>', _event(
                    "unknown_event",
                    "❓",
                    f"Unknown: {event_name}",
                    "Unhandled event type encountered.",
                )
        
        # Final yield to ensure thinking animation is removed
        if buffer:
            yield buffer, None


    async def chat(self, message: str, user_id: str) -> str:
        """Simple chat interface that processes a message and returns the final response.
        
        Args:
            message: User's input message
            user_id: User identifier for context isolation
            
        Returns:
            str: Final assistant response
        """
        ctx = self._get_or_create_user_ctx(user_id)
        task_result = await ctx.agent.run(task=message)
        
        # Extract the final message content from the task result
        messages = getattr(task_result, 'messages', [])
        if messages:
            final_message = messages[-1]
            if hasattr(final_message, 'content') and isinstance(final_message.content, str):
                return final_message.content
        
        return "I apologize, but I couldn't generate a response."

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
        ctx = self._get_or_create_user_ctx(user_id)
        model_context = ctx.agent.model_context
        
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
        return list(self._user_ctx_cache.keys())

    def user_exists(self, user_id: str) -> bool:
        """Check if a user context exists in the cache.
        
        Args:
            user_id: User identifier to check
            
        Returns:
            bool: True if user context exists, False otherwise
        """
        return user_id in self._user_ctx_cache

    def reset_user_memory(self, user_id: str) -> None:
        """Reset a user's memory by removing their cached context.
        
        This clears the user's insight bank and forces recreation of
        a fresh memory controller on next interaction.
        
        Args:
            user_id: User identifier whose memory should be reset
        """
        if user_id in self._user_ctx_cache:
            self._user_ctx_cache.pop(user_id, None)
    