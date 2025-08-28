# import suppress_warnings  # Must be first to suppress warnings
import warnings
warnings.filterwarnings("ignore")
import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from autogen_agentchat.agents import AssistantAgent
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
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.memory.mem0 import Mem0Memory
from tavily import TavilyClient

from config import AppConfig
from context.redis_chat_completion_context import RedisChatCompletionContext


# Constants
DEFAULT_BUFFER_SIZE = 10
DEFAULT_MAX_SEARCH_RESULTS = 8


@dataclass
class UserCtx:
    """User-specific context containing Mem0 memory and agent instances.
    
    Attributes:
        memory: Mem0 memory instance for user-specific memory management
        agent: Main assistant agent with tools and memory integration
    """
    memory: Mem0Memory
    agent: AssistantAgent


class TravelAgent:
    """Travel planning agent with Mem0-powered personalized memory capabilities.
    
    This agent provides personalized travel planning services by maintaining
    separate Mem0 memory contexts for each user. Each user gets their own
    Mem0 memory instance and supervisor agent that are cached for performance.
    
    Features:
        - Per-user memory isolation using Mem0 with Redis backend
        - Semantic memory search and retrieval via Mem0
        - Web search integration for current travel information
        - Chat history management with configurable buffer sizes
        - Automatic memory extraction and personalized recommendations
    
    Attributes:
        config: Application configuration containing API keys and model settings
        tavily_client: Web search client for travel information
        agent_model: OpenAI client for the main travel agent
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
        print(f"🔧 Initializing Tavily client with API key: {config.tavily_api_key[:10]}...", flush=True)
        try:
            self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
            print("✅ Tavily client initialized successfully", flush=True)
        except Exception as e:
            print(f"❌ Failed to initialize Tavily client: {e}", flush=True)
            self.tavily_client = None
            
        print(f"🔧 Initializing OpenAI client with model: {config.travel_agent_model_name}", flush=True)
        try:
            self.agent_model = OpenAIChatCompletionClient(
                model=config.travel_agent_model_name, 
                parallel_tool_calls=False
            )
            print("✅ OpenAI client initialized successfully", flush=True)
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI client: {e}", flush=True)
            raise

        # Initialize user context cache
        self._user_ctx_cache = {}
    
    async def initialize_seed_data(self) -> None:
        """Initialize seed users with their memories. Call this after creating the agent."""
        await self._init_seed_users()

    # ------------------------------
    # User Context Management
    # ------------------------------
    
    def _create_memory(self, user_id: str) -> Mem0Memory:
        """Create Mem0 memory instance for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Mem0Memory instance configured for the user
            
        Raises:
            RuntimeError: If Mem0 configuration fails
        """
        try:
            return Mem0Memory(
                user_id=user_id,
                is_cloud=False,
                config={
                    "vector_store": {
                        "provider": "redis",
                        "config": {
                            "collection_name": f"memory:{user_id}",  # Per-user namespace
                            "embedding_model_dims": 1536,  # Default for OpenAI embeddings
                            "redis_url": self.config.redis_url,
                        }
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4.1-mini",     # <-- pick your OpenAI model
                            "temperature": 0.1,
                            "api_key": self.config.openai_api_key,
                        }
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": self.config.mem0_embedding_model,
                            "api_key": self.config.openai_api_key,
                        }
                    },
                    "custom_fact_extraction_prompt": """
                    You extract durable traveler details and preferences for a travel concierge.

                    Return JSON only (no prose, no code fences), exactly in this form:
                    {"facts": ["<fact-1>", "<fact-2>", "..."]}
                    If no durable facts are present, return: {"facts": []}

                    Extract only durable, user-specific items helpful across trips:
                    - Airline & seat/class; hotel brands/style; loyalty programs/IDs
                    - Dietary restrictions/allergies; cuisine likes/dislikes
                    - Budget range; preferred airports or arrival windows; accessibility needs
                    - User interests and biographical details that impact planning

                    Constraints:
                    - At most 3–4 concise facts per turn.
                    - One fact per array element, short and declarative.
                    - Exclude greetings, moods, generic chit-chat, and one-off/temporary details unless explicitly time-bound.
                    """,
                    "version": "v1.1"
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Mem0 memory for user {user_id}: {e}")
    
    def _get_or_create_user_ctx(self, user_id: str) -> UserCtx:
        """Get or create user-specific context with Mem0 memory and agent components.
        
        Creates and caches a complete user context including Mem0 memory,
        chat history management, and supervisor agent.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            UserCtx: Complete user context with Mem0 memory initialized
        """
        if user_ctx := self._user_ctx_cache.get(user_id):
            return user_ctx

        print(f"🧠 Creating memory bank for user: {user_id}")
        
        # Create Mem0 memory instance
        mem0_memory = self._create_memory(user_id)
        
        # Initialize chat history management
        model_context = RedisChatCompletionContext(
            redis_url=self.config.redis_url,
            user_id=user_id,
            buffer_size=DEFAULT_BUFFER_SIZE
        )
        
        # Create supervisor agent with Mem0 memory
        agent = self._create_agent(
            model_context=model_context,
            memory=mem0_memory
        )
        
        # Cache and return user context
        self._user_ctx_cache[user_id] = UserCtx(
            memory=mem0_memory,
            agent=agent
        )
        return self._user_ctx_cache[user_id]

    def _load_seed_data(self) -> Dict[str, Any]:
        """Load seed data from JSON file."""
        seed_file = Path(__file__).parent / "context" / "seed.json"
        try:
            with open(seed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    async def _init_seed_users(self) -> None:
        """Initialize seed users with memories from seed.json."""
        seed_data = self._load_seed_data()
        user_memories = seed_data.get("user_memories", {})
        
        for user_id, memories in user_memories.items():
            try:
                ctx = self._get_or_create_user_ctx(str(user_id))
                print(f"🌱 Seeding memory for user: {user_id}")
                
                for memo in memories:
                    # Add memory content to Mem0
                    await ctx.memory.add(MemoryContent(
                        content=memo["insight"],
                        mime_type=MemoryMimeType.TEXT
                    ))
                
                print(f"✅ Seeded {len(memories)} memories to Redis for user: {user_id}")
                
            except Exception as e:
                print(f"❌ Failed to seed memory for user {user_id}: {e}")
                continue

    def _create_agent(
        self,
        model_context: RedisChatCompletionContext,
        memory: Mem0Memory
    ) -> AssistantAgent:
        """Create supervisor agent with Mem0 memory integration and tools.
        
        Args:
            model_context: Redis-backed chat completion context for history
            memory: Mem0 memory instance for long-term memory
            
        Returns:
            AssistantAgent: Configured supervisor with memory and tools
        """
        print("🤖 Creating AssistantAgent with tools...", flush=True)
        
        tools = self._get_tools()
        print(f"   📋 Registered {len(tools)} tools:", flush=True)
        for i, tool in enumerate(tools, 1):
            tool_name = getattr(tool, 'name', 'unknown')
            print(f"      {i}. {tool_name}", flush=True)
        
        try:
            agent = AssistantAgent(
                name="agent",
                model_client=self.agent_model,
                model_context=model_context,  # Chat history management
                memory=[memory],         # Long term memory management
                tools=tools,
                system_message=self._get_system_message(),
                max_tool_iterations=self.config.max_tool_iterations,
                model_client_stream=True,     # Enable token streaming
            )
            print("✅ AssistantAgent created successfully", flush=True)
            return agent
        except Exception as e:
            print(f"❌ Failed to create AssistantAgent: {e}", flush=True)
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}", flush=True)
            raise
    
    def _get_tools(self) -> List[FunctionTool]:
        """Get the list of tools for the travel agent.
        
        Returns:
            List[FunctionTool]: List of tools available to the agent
        """
        print("🔧 Creating FunctionTool instances...", flush=True)
        
        tools = []
        
        try:
            logistics_tool = FunctionTool(
                func=self.search_logistics,
                description=(
                    "Time-aware logistics search ONLY: flights, hotels, and intercity/local transport. "
                    "Use for availability, schedules, prices, carriers/properties, or routes. "
                    "Arguments: query (required), start_date (optional, YYYY-MM-DD), end_date (optional, YYYY-MM-DD). "
                    "Always include dates when the user mentions a travel window; if ambiguous, ask for dates before booking guidance. "
                    "NEVER use this for activities, attractions, neighborhoods, or dining. "
                    "Results are restricted to reputable flight/hotel/transport sources; top URLs are deeply extracted."
                )
            )
            tools.append(logistics_tool)
            print("   ✅ search_logistics tool created", flush=True)
        except Exception as e:
            print(f"   ❌ Failed to create search_logistics tool: {e}", flush=True)
            
        try:
            general_tool = FunctionTool(
                func=self.search_general,
                description=(
                    "Time-aware destination research: activities, attractions, neighborhoods, dining, events, local tips. "
                    "Use for up-to-date things to do, cultural context, and planning inspiration. "
                    "Arguments: query (required) "
                    "Scope searches to the relevant season/year when possible and prefer recent sources. "
                    "NEVER use this for flights, hotels, or transport logistics. "
                    "Example: 'things to do in Lisbon in June 2026'."
                )
            )
            tools.append(general_tool)
            print("   ✅ search_general tool created", flush=True)
        except Exception as e:
            print(f"   ❌ Failed to create search_general tool: {e}", flush=True)
        
        print(f"🏁 Tool creation complete. {len(tools)} tools ready.", flush=True)
        return tools
    
    def _get_system_message(self) -> str:
        """Get the system message for the travel agent supervisor.
        
        Returns:
            str: Complete system message with role, responsibilities, and workflow
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return (
            f"You are an expert, time-aware, friendly Travel Concierge AI. Today is {today} (UTC). "
            "Assume your built in knowledge may be outdated; for anything time-sensitive, verify with tools.\n\n"
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
            "- Consider any appended Important insights (long-term memory) before answering and adapt to them.\n"
            "- Memory system: Mem0 with semantic search and automatic memory extraction"
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
        print(f"🔧 LOGISTICS: {query} | {start_date} to {end_date}", flush=True)
        
        try:
            # Validate Tavily client
            if not self.tavily_client:
                error_msg = "❌ Tavily client not initialized"
                print(error_msg, flush=True)
                return {"error": error_msg, "results": [], "extractions": []}
            
            # Augment query with dates if provided
            enhanced_query = query
            if start_date:
                enhanced_query += f" from {start_date}"
            if end_date and end_date != start_date:
                enhanced_query += f" to {end_date}"
            
            include_domains = [
                # Flights / OTAs
                "expedia.com", "kayak.com", "travel.google.com",
                # Hotels / stays
                "booking.com", "hotels.com",
            ]

            search_kwargs = {
                "query": enhanced_query,
                "topic": "general",
                "search_depth": "advanced",
                "include_domains": include_domains,
                "max_results": DEFAULT_MAX_SEARCH_RESULTS,
            }

            results = self.tavily_client.search(**search_kwargs)
            
            if not results:
                print(f"⚠️ Empty results from Tavily", flush=True)
                return {"results": [], "extractions": []}

            # Sort by score descending and filter out low-quality results
            all_results = results.get("results", [])
            sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
            trimmed = [r for r in sorted_results if r.get("score", 0) > 0.2]
            print(f"📊 Found {len(trimmed)}/{len(all_results)} quality results", flush=True)
            
            results["results"] = trimmed

            # Extract top 2 URLs for deeper context
            top_urls = [r.get("url") for r in trimmed[:2] if r.get("url")]
            extractions: List[Dict[str, Any]] = []
            
            if top_urls:
                try:
                    extracted = self.tavily_client.extract(urls=top_urls)
                    if isinstance(extracted, dict) and extracted.get("results"):
                        extractions = extracted.get("results", [])
                    elif isinstance(extracted, list):
                        extractions = extracted
                    print(f"📄 Extracted {len(extractions)} content blocks", flush=True)
                except Exception as extract_e:
                    print(f"⚠️ URL extraction failed: {extract_e}", flush=True)

            results["extractions"] = extractions
            print(f"✅ LOGISTICS COMPLETE: {len(trimmed)} results + {len(extractions)} extractions", flush=True)
            return results
            
        except Exception as e:
            error_msg = f"❌ LOGISTICS ERROR: {str(e)}"
            print(error_msg, flush=True)
            return {"error": error_msg, "results": [], "extractions": []}

    def search_general(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """📍 General destination research: activities, attractions, neighborhoods, dining, events.

        What it is for
        - Up-to-date things to do, local highlights, neighborhoods to stay, dining ideas, and cultural context.

        How to use
        - Provide a destination/time-focused query, e.g., "things to do in Lisbon in June",
          "Barcelona food tours", "best neighborhoods to stay in Tokyo".

        Behavior
        - Runs an open web search (no logistics domains restriction) with raw content for context.
        """
        print(f"\n🔧 SEARCH_GENERAL CALLED", flush=True)
        print(f"   Query: {query}", flush=True)
        
        try:
            # Validate Tavily client
            if not self.tavily_client:
                error_msg = "❌ Tavily client not initialized"
                print(error_msg, flush=True)
                return {"error": error_msg, "results": []}
            
            search_kwargs = {
                "query": query,
                "topic": "general",
                "search_depth": "advanced",
                "include_raw_content": True,
                "max_results": DEFAULT_MAX_SEARCH_RESULTS,
            }
            
            print(f"   🔍 Calling Tavily search with kwargs: {search_kwargs}", flush=True)
            results = self.tavily_client.search(**search_kwargs)
            print(f"   ✅ Tavily search completed. Raw results type: {type(results)}", flush=True)
            
            if not results:
                print(f"   ⚠️ Empty results from Tavily", flush=True)
                return {"results": []}
            
            # Filter results by score
            all_results = results.get("results", [])
            print(f"   📊 Found {len(all_results)} raw results", flush=True)
            
            filtered_results = [r for r in all_results if r.get("score", 0) > 0.2]
            print(f"   🎯 After filtering (score > 0.2): {len(filtered_results)} results", flush=True)
            
            results["results"] = filtered_results

            print(f"   🏁 GENERAL SEARCH COMPLETE. Returning {len(filtered_results)} results", flush=True)
            return results
            
        except Exception as e:
            error_msg = f"❌ GENERAL SEARCH ERROR: {str(e)}"
            print(error_msg, flush=True)
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}", flush=True)
            return {"error": error_msg, "results": []}

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
        
        # Note: User message will be stored async after response completes
        
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


    # -----------------
    # Utility Methods
    # -----------------

    async def store_memory(self, user_id: str, user_message: str) -> None:
        """Store user message in memory asynchronously.
        
        This method is called after the response is complete to avoid blocking
        the initial request processing.
        
        Args:
            user_id: User identifier for context isolation
            user_message: The user's input message
        """
        try:
            ctx = self._get_or_create_user_ctx(user_id)
            
            # Store user message
            await ctx.memory.add(MemoryContent(
                content=user_message,
                mime_type=MemoryMimeType.TEXT
            ))
            
        except Exception as e:
            print(f"⚠️ Failed to store conversation memory for user {user_id}: {e}")

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
        """Reset a user's Mem0 memory by removing their cached context.
        
        This clears the user's cached context and forces recreation of
        a fresh Mem0 memory instance on next interaction.
        
        Args:
            user_id: User identifier whose memory should be reset
        """
        if user_id in self._user_ctx_cache:
            print(f"🗑️  Resetting Mem0 memory for user: {user_id}")
            self._user_ctx_cache.pop(user_id, None)
    