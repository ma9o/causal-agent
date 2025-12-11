"""Orchestrator agents using Inspect AI with OpenRouter."""

from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)

from causal_agent.utils.config import get_config
from causal_agent.utils.llm import multi_turn_generate, parse_json_response
from .prompts import (
    STRUCTURE_PROPOSER_SYSTEM,
    STRUCTURE_PROPOSER_USER,
    STRUCTURE_REVIEW_REQUEST,
)
from .schemas import DSEMStructure

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


async def propose_structure_async(
    question: str,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Use the orchestrator LLM to propose a causal model structure.

    Two-stage process:
    1. Initial proposal: Generate structure from question and data
    2. Self-review: Double-check measurement_dtype, aggregation, and how_to_measure

    Args:
        question: The causal research question (natural language)
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset (size, timespan, etc.)

    Returns:
        DSEMStructure as a dictionary
    """
    model_name = get_config().stage1_structure_proposal.model
    model = get_model(model_name)

    # Format the chunks for the prompt
    chunks_text = "\n".join(data_sample)

    # Build initial messages
    messages = [
        ChatMessageSystem(content=STRUCTURE_PROPOSER_SYSTEM),
        ChatMessageUser(
            content=STRUCTURE_PROPOSER_USER.format(
                question=question,
                dataset_summary=dataset_summary or "Not provided",
                chunks=chunks_text,
            )
        ),
    ]

    # Run multi-turn: initial proposal + self-review
    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        follow_ups=[STRUCTURE_REVIEW_REQUEST],
    )

    # Parse and validate final result
    reviewed_data = parse_json_response(completion)
    reviewed_structure = DSEMStructure.model_validate(reviewed_data)

    return reviewed_structure.model_dump(by_alias=True)


def propose_structure(
    question: str,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Synchronous wrapper for propose_structure_async.

    Args:
        question: The causal research question
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset (size, timespan, etc.)

    Returns:
        DSEMStructure as a dictionary
    """
    import asyncio

    return asyncio.run(propose_structure_async(question, data_sample, dataset_summary))
