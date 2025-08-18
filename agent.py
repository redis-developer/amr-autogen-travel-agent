import os
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from redis import Redis

from tavily import TavilyClient

from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.redis import RedisMemory, RedisMemoryConfig
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console

from redisvl.redis.utils import convert_bytes
from config import AppConfig


class TravelAgent:
    """
    A comprehensive travel planning agent that uses AutoGen with memory and tools.
    Helps users plan trips, save preferences, search for travel info, and export itineraries.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the travel agent with all necessary components.
        
        Args:
            config: Application configuration (if None, loads from environment)
        """
        # Load configuration
        if config is None:
            from config import get_config
            config = get_config()
        
        self.config = config
        
        # Set environment variables for libraries that expect them
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key
            
        # Initialize Tavily client
        self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
        
        # Initialize Redis memory
        self.memory = RedisMemory(
            config=RedisMemoryConfig(
                redis_url=config.redis_url,
                index_name=config.redis_index_name,
                prefix=config.redis_prefix,
            )
        )
        
        # Initialize model client
        self.model_client = OpenAIChatCompletionClient(
            model=config.model_name, 
            parallel_tool_calls=False
        )
        
        # Create the supervisor agent
        self.supervisor = self._create_supervisor_agent()
    
    async def save_user_preference(self, user_id: str | None, preference: str) -> str:
        """Save a user preference to long-term memory."""
        await self.memory.add(
            MemoryContent(
                content=preference,
                mime_type=MemoryMimeType.TEXT,
                metadata={"user_id": user_id or "none"},
            )
        )
        return "ok"

    def search_web(self, query: str, topic: str = "general") -> Dict[str, Any]:
        """Search the web for travel-related information and extract content from top results."""
        # Search for results
        search_results = self.tavily_client.search(
            query=query,
            topic=topic,
            search_depth="basic",
            max_results=5
        )
        
        # Extract content from top 3 URLs
        # top_urls = [r["url"] for r in search_results["results"][:3]]
        # extracted_pages = [self.tavily_client.extract(url) for url in top_urls]
        
        return search_results

    
    # def export_calendar(self, itinerary: List[Dict], out_path: str = "itinerary.ics") -> str:
    #     """Export the itinerary plan into a calendar file format."""
    #     cal = Calendar()
    #     cal.add('prodid', '-//Redis Travel Agent//AMR Demo//')
    #     cal.add('version', '2.0')
        
    #     for stop in itinerary:
    #         ev = Event()
    #         ev.add('summary', stop['title'])
    #         ev.add('dtstart', stop['start_dt'])
    #         ev.add('dtend', stop['end_dt'])
    #         if stop.get('notes'):
    #             ev.add('description', stop['notes'])
    #         cal.add_component(ev)
        
    #     Path(out_path).write_bytes(cal.to_ical())
    #     return out_path
    
    def _create_supervisor_agent(self) -> AssistantAgent:
        """Create the main supervisor agent with tools and system message."""
        return AssistantAgent(
            name="supervisor",
            model_client=self.model_client,
            memory=[self.memory],
            tools=[
                FunctionTool(
                    func=self.save_user_preference, 
                    description="Save extracted, individualized user preferences (budget, dietary restrictions, accommodation style, activities, etc.) to use for future trip planning. These are discrete facts we learn about the user and their needs along the way."
                ),
                FunctionTool(
                    func=self.search_web, 
                    description="Search the web for current travel information including destinations, flights, hotels, activities, events, transportation options, prices, and reviews."
                ),
                # FunctionTool(
                #     func=self.export_calendar, 
                #     description="Export the final travel itinerary as a calendar file (.ics format) with dates, times, and locations for easy import into calendar apps."
                # )
            ],
            system_message=(
                "You are an expert Travel Concierge AI that helps users plan and book complete trips from start to finish.\n\n"
                
                "ROLE & RESPONSIBILITIES:\n"
                "- Help users discover destinations, plan itineraries, find accommodations, and organize travel logistics\n"
                "- Gather essential trip details: dates, budget, group size, preferences, and special requirements\n"
                "- Research current travel options, prices, availability, and recommendations\n"
                "- Create detailed day-by-day itineraries with specific times, locations, and booking information\n"
                "- Save learned user preferences for personalized future recommendations\n\n"
                
                "WORKFLOW:\n"
                "1. DISCOVERY: Ask clarifying questions when necessary to get more useful details.\n"
                "2. RESEARCH: Use web search to find up-to-date options for flights, hotels, activities, restaurants, and transportation.\n"
                "3. PLANNING: Create a comprehensive itinerary with specific recommendations, times, and logistics\n"

                "Always be proactive and focus on creating actionable travel plans with specific recommendations the user can book."
            ),
            max_tool_iterations=self.config.max_tool_iterations,
        )
    
    async def chat(self, user_message: str, user_id: Optional[str] = None) -> str:
        """
        Process a user message and return the agent's response.
        
        Args:
            user_message: The user's input message
            user_id: Optional user identifier for personalization
            
        Returns:
            The agent's response as a string
        """
        # Run the agent with the user's message
        stream = self.supervisor.run_stream(task=user_message)
        
        # Process the stream and return the result
        result = await Console(stream)
        
        # Extract the final message content from the result
        if hasattr(result, 'messages') and result.messages:
            # Get the last message from the supervisor
            last_message = result.messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
        
        return "I apologize, but I couldn't process your request properly. Please try again."
    
    async def get_user_preferences(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve stored user preferences from Redis memory.
        
        Args:
            user_id: User identifier to filter preferences (if None, gets all)
            
        Returns:
            List of preference dictionaries with content, timestamp, and metadata
        """
        try:
            # Get the Redis client from the memory instance
            redis_client = self.memory.message_history._index._redis_client
            
            # Get all preference keys
            preference_keys = redis_client.keys("user_preferences:*")

            with redis_client.pipeline(transaction=False) as pipe:
                for key in preference_keys:
                    # Get the hash data
                    pipe.hgetall(key)
                preferences = pipe.execute()

            preferences = [
                {
                    'content': hash_data.get('content', ''),
                    'timestamp': hash_data.get('timestamp', ''),
                } for hash_data in convert_bytes(preferences)
            ]
                
            # Sort by timestamp (newest first)
            preferences.sort(key=lambda x: x['timestamp'], reverse=True)
            return preferences
            
        except Exception as e:
            print(f"Error retrieving preferences: {e}")
            return []
    
    async def get_conversation_history(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieve conversation history for a user (placeholder for future implementation).
        
        Args:
            user_id: User identifier
            
        Returns:
            List of conversation messages
        """
        # This is a placeholder - in a full implementation, you'd store and retrieve
        # conversation history from Redis or another persistence layer
        return []
