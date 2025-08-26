from autogen_ext.experimental.task_centric_memory._prompter import Prompter

from typing import List, Union
from autogen_ext.experimental.task_centric_memory.memory_controller import MemoryController
from autogen_ext.experimental.task_centric_memory._prompter import Prompter
from autogen_ext.experimental.task_centric_memory.utils import Teachability, PageLogger
from autogen_ext.models.openai import OpenAIChatCompletionClient


class TravelPrompter(Prompter):
    """
    Narrow advice extraction to durable, time-agnostic travel preferences and normalize topics/tags for retrieval.
    """
    async def extract_advice(self, text: str) -> str | None:
        """
        Return ONE short, durable preference insight if present, else None.
        Example return: "[pref.airline]=Delta; prefers aisle seats"
        """
        sys_message = (
            "You extract durable, user-specific travel preferences only. Ignore time-sensitive facts (prices, availability, temporary events). "
            "If none exist, reply exactly: NONE. "
            "If found, output ONE short sentence, declarative, no hedging. "
            "Prefer normalized tags like [pref.airline],[pref.seat],[pref.hotel],[pref.dietary],"
            "[pref.airport],[pref.time],[pref.budget],[pref.activities]."
        )
        user_message = [
            "From the text below, extract ONE durable travel preference or constraint "
            "that will remain useful across trips (airline/hotel loyalty, seat/dietary, budget, time-of-day, airport, "
            "accessibility, favorite activities). If nothing qualifies, reply: NONE."
        ]
        user_message.append("\n# Text to analyze")
        user_message.append(text)
        self._clear_history()
        response = await self.call_model(
            summary="Extract travel preferences from text", 
            system_message_content=sys_message, 
            user_content=user_message
        )
        return response if response != "NONE" else None

    async def validate_insight(self, insight: str, task: str) -> bool:
        """
        Use a quick LLM gate to keep only preference-like insights that help travel planning.
        """
        sys_message = (
            "You validate whether a short text is a durable, time-agnostic user travel preference "
            "useful for future trip planning. Return exactly YES or NO."
        )
        user_message = [
            "Criteria for YES: user-specific, durable (not one-off), time-agnostic, and relevant to travel logistics "
            "(airline/hotel loyalty, seat/dietary, budget, time-of-day, airport, accessibility, favorite activities). "
            "Otherwise NO."
        ]
        user_message.append(f"\n# Insight to validate")
        user_message.append(f"INSIGHT: {insight}")
        user_message.append(f"TASK: {task}")
        user_message.append("Answer: YES or NO.")
        self._clear_history()
        response = await self.call_model(
            summary="Validate travel preference insight",
            system_message_content=sys_message,
            user_content=user_message
        )
        return response.strip().upper().startswith("Y")
