"""Worker agents using Inspect AI with OpenRouter."""

import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)

from causal_agent.utils.config import get_config
from .prompts import WORKER_SYSTEM, WORKER_USER
from .schemas import WorkerOutput

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


async def process_chunk_async(
    chunk: str,
    chunk_id: str,
    dag: dict,
) -> dict:
    """
    Process a single data chunk against the candidate DAG.

    Args:
        chunk: The data chunk to process
        chunk_id: Unique identifier for this chunk
        dag: The candidate DAG from the orchestrator (DSEMStructure as dict)

    Returns:
        WorkerOutput as a dictionary
    """
    model_name = get_config().stage2_workers.model
    model = get_model(model_name)

    # Format the DAG for the prompt
    dag_json = json.dumps(dag, indent=2)

    messages = [
        ChatMessageSystem(content=WORKER_SYSTEM),
        ChatMessageUser(
            content=WORKER_USER.format(
                dag_json=dag_json,
                chunk_id=chunk_id,
                chunk=chunk,
            )
        ),
    ]

    response = await model.generate(messages)
    content = response.completion

    # Parse and validate the response
    # Handle markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:500]}...")
        raise ValueError(f"Failed to parse worker response as JSON: {e}") from e

    # Ensure chunk_id is set
    data["chunk_id"] = chunk_id

    output = WorkerOutput.model_validate(data)

    return output.model_dump()


def process_chunk(
    chunk: str,
    chunk_id: str,
    dag: dict,
) -> dict:
    """
    Synchronous wrapper for process_chunk_async.

    Args:
        chunk: The data chunk to process
        chunk_id: Unique identifier for this chunk
        dag: The candidate DAG from the orchestrator

    Returns:
        WorkerOutput as a dictionary
    """
    import asyncio

    return asyncio.run(process_chunk_async(chunk, chunk_id, dag))


async def process_chunks_async(
    chunks: list[str],
    dag: dict,
) -> list[dict]:
    """
    Process multiple chunks in parallel.

    Args:
        chunks: List of data chunks to process
        dag: The candidate DAG from the orchestrator

    Returns:
        List of WorkerOutput dictionaries
    """
    import asyncio

    tasks = [
        process_chunk_async(chunk, f"chunk_{i:04d}", dag)
        for i, chunk in enumerate(chunks)
    ]

    return await asyncio.gather(*tasks)


def process_chunks(
    chunks: list[str],
    dag: dict,
) -> list[dict]:
    """
    Synchronous wrapper for process_chunks_async.

    Args:
        chunks: List of data chunks to process
        dag: The candidate DAG from the orchestrator

    Returns:
        List of WorkerOutput dictionaries
    """
    import asyncio

    return asyncio.run(process_chunks_async(chunks, dag))
