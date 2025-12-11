"""Shared LLM utilities for multi-turn generation."""

import json
from typing import TYPE_CHECKING

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    Model,
)

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage


def parse_json_response(content: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:500]}...")
        raise ValueError(f"Failed to parse model response as JSON: {e}") from e


async def multi_turn_generate(
    messages: list["ChatMessage"],
    model: Model,
    follow_ups: list[str] | None = None,
    config: GenerateConfig | None = None,
) -> str:
    """
    Run a multi-turn conversation: generate, then continue with follow-up prompts.

    Args:
        messages: Initial messages (typically system + user prompt)
        model: The model to use for generation
        follow_ups: List of follow-up user prompts to send after each response (default: none)
        config: Optional generation config

    Returns:
        The final completion string
    """
    messages = list(messages)  # Don't mutate original
    follow_ups = follow_ups or []

    # Initial generation
    response = await model.generate(messages, config=config)
    messages.append(ChatMessageAssistant(content=response.completion))

    # Follow-up turns
    for prompt in follow_ups:
        messages.append(ChatMessageUser(content=prompt))
        response = await model.generate(messages, config=config)
        messages.append(ChatMessageAssistant(content=response.completion))

    return response.completion
