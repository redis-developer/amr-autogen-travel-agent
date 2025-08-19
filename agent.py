import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from tavily import TavilyClient

from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.experimental.task_centric_memory import MemoryController
from autogen_ext.experimental.task_centric_memory.utils import Teachability, PageLogger
from autogen_agentchat.ui import Console

from config import AppConfig

from memory.redis_chat_completion_context import RedisChatCompletionContext

@dataclass
class UserCtx:
    controller: MemoryController
    teachability: Teachability
    supervisor: AssistantAgent
    logger: PageLogger


class TravelAgent:
    """
    Travel planning agent using a per-user Task-Centric Memory (TCM).
    Each user_id gets its own MemoryController + Teachability + Supervisor (cached).
    """

    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            from config import get_config
            config = get_config()
        self.config = config

        # Env for SDKs that read from env
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key

        # Shared clients
        self.tavily_client = TavilyClient(api_key=config.tavily_api_key)

        # App uses two model deployments (can be the same if you prefer)
        self.agent_model = OpenAIChatCompletionClient(
            model=config.travel_agent_model_name, parallel_tool_calls=False
        )
        self.memory_model = OpenAIChatCompletionClient(model=config.memory_model_name)

        # In-memory registry of user-specific contexts
        self._users: Dict[str, UserCtx] = {}

    # ------------------------------
    # User-scoped context management
    # ------------------------------
    def _get_or_create_user_ctx(self, user_id: str) -> UserCtx:
        if user_id in self._users:
            return self._users[user_id]

        # Per-user logger (nice to show "insights applied" in the demo logs)
        logger = PageLogger(config={"level": "INFO", "path": f"./memory/logs/{user_id}"})

        # Per-user memory controller with a per-user bank path (persists across sessions)
        controller = MemoryController(
            reset=False,               # keep insights alive for this user
            client=self.memory_model,  # model used by the controller for generalize/validate
            logger=logger,
            config={
                "generalize_task": True,
                "generate_topics": True,
                "validate_memos": True,
                "max_memos_to_retrieve": 4,        # keep prompt small/predictable
                "MemoryBank": {"path": f"./memory/users/{user_id}"},
            },
        )

        # Teachability adapter (auto learn + auto inject relevant insights)
        teachability = Teachability(controller, name=f"tcm_{user_id}")

        model_context = RedisChatCompletionContext(redis_url=self.config.redis_url, user_id=user_id)

        # A supervisor agent bound to THIS user's Teachability memory
        supervisor = self._create_supervisor_agent(model_context=model_context, memory_adapter=teachability)

        ctx = UserCtx(controller=controller, teachability=teachability, supervisor=supervisor, logger=logger)
        self._users[user_id] = ctx
        return ctx

    def _create_supervisor_agent(self, model_context: RedisChatCompletionContext, memory_adapter: Teachability) -> AssistantAgent:
        """Create a supervisor bound to a given Teachability adapter."""
        return AssistantAgent(
            name="supervisor",
            model_client=self.agent_model,
            model_context=model_context, # Chat history handling
            memory=[memory_adapter],  # Long term memory handling
            tools=[
                FunctionTool(
                    func=self.search_web,
                    description="Search the web for current travel information: destinations, flights, hotels, activities, events, transit, and prices."
                ),
                # If/when you re-enable calendar export, add it here as a FunctionTool
            ],
            system_message=(
                "You are an expert Travel Concierge AI that helps users plan and book complete trips from start to finish with the ability to recall user preferences across sessions.\n\n"
            
                "ROLE & RESPONSIBILITIES:\n"
                "- Help users discover destinations, plan itineraries, find accommodations, and organize travel logistics\n"
                "- Gather essential trip details: dates, budget, group size, preferences, and special requirements\n"
                "- Research current travel options, prices, availability, and recommendations\n"
                "- Create detailed day-by-day itineraries with specific times, locations, and booking information\n"
                "- Save learned user preferences and insights for personalized future recommendations\n\n"
                
                "WORKFLOW:\n"
                "1. DISCOVERY: Ask clarifying questions when necessary to get more useful details.\n"
                "2. RESEARCH: Use web search to find up-to-date options for flights, hotels, activities, restaurants, and transportation.\n"
                "3. PLANNING: Create a comprehensive itinerary with specific recommendations, times, and logistics\n"

                "Before answering, consider any 'Important insights' from the user that may be appended by long term memory and use them."
            ),
            max_tool_iterations=self.config.max_tool_iterations,
        )

    # -----------------
    # Tools
    # -----------------

    def search_web(self, query: str) -> Dict[str, Any]:
        "Search the web for up to date travel options, details, bookings, and events for the user."
        return self.tavily_client.search(
            query=query, topic="general", search_depth="basic", max_results=5
        )

    # -----------------
    # Chat + Insights UI
    # -----------------
    async def _relevant_insights_for_task(self, controller: MemoryController, task_text: str, limit: int = 4) -> list[str]:
        """Mirror what Teachability retrieves; show a few bullets for transparency."""
        try:
            memos = await controller.retrieve_relevant_memos(task=task_text)
            uniq, seen = [], set()
            for m in memos:
                t = (m.insight or "").strip()
                if t and t not in seen:
                    uniq.append(t); seen.add(t)
                if len(uniq) >= limit:
                    break
            return uniq
        except Exception as e:
            print("Issue during memory retrieval", str(e), flush=True)
            return []

    async def chat(self, user_message: str, user_id: Optional[str] = None) -> str:
        """Stream one chat turn for a given user_id; surface applied insights after."""
        uid = user_id or "demo_user"
        ctx = self._get_or_create_user_ctx(uid)

        stream = ctx.supervisor.run_stream(task=user_message)
        result = await Console(stream)

        # Surface insights (the same class of items Teachability injected)
        if insights := await self._relevant_insights_for_task(ctx.controller, user_message, limit=4):
            print("\n\n--- INSIGHTS APPLIED (user:", uid, ") ---", flush=True)
            for i in insights:
                print("•", i)
            print("--- END INSIGHTS ---\n", flush=True)
        else:
            print("--- NO INSIGHTS EXTRACTED ---", flush=True)

        if hasattr(result, "messages") and result.messages:
            last = result.messages[-1]
            if hasattr(last, "content"):
                return last.content
        return "I'm sorry, I couldn't process that. Please try again."
    
    # -----------------
    # Optional helpers
    # -----------------

    async def get_chat_history(self, user_id: str, n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieve chat history for a user from Redis.
        
        Args:
            user_id: The user ID to get history for
            n: Number of messages to retrieve. If None, uses buffer_size. If -1, gets all messages.
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys for Gradio chat interface
        """
        if user_id not in self._users:
            return []
        
        ctx = self._users[user_id]
        # Get the model context (RedisChatCompletionContext)
        model_context = ctx.supervisor.model_context
        
        try:
            # Get messages from Redis
            messages = await model_context.get_messages(n=n)
            
            # Convert LLMMessage objects to Gradio-compatible format
            gradio_messages = []
            for msg in messages:
                # Only show user messages and assistant messages with actual text content
                if msg.__class__.__name__ == 'UserMessage':
                    if hasattr(msg, 'content') and msg.content and msg.content.strip():
                        gradio_messages.append({
                            'role': 'user',
                            'content': msg.content
                        })
                elif msg.__class__.__name__ == 'AssistantMessage':
                    # Only show assistant messages that have actual text content (not just tool calls)
                    if hasattr(msg, 'content') and msg.content and msg.content.strip():
                        gradio_messages.append({
                            'role': 'assistant', 
                            'content': msg.content
                        })
                # Skip everything else: SystemMessage, FunctionExecutionResultMessage, tool calls, etc.
            
            return gradio_messages
            
        except Exception as e:
            print(f"Error retrieving chat history for user {user_id}: {e}")
            return []

    def get_user_list(self) -> List[str]:
        """Get list of all user IDs that have been created."""
        return list(self._users.keys())

    def user_exists(self, user_id: str) -> bool:
        """Check if a user ID exists."""
        return user_id in self._users

    def reset_user_memory(self, user_id: str):
        """Clear this user's insight bank (fresh demo)."""
        if user_id in self._users:
            # wipe on-disk bank and re-create a fresh controller
            self._users.pop(user_id, None)
    