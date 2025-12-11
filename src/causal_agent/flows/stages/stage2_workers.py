"""Stage 2: Dimension Population (Workers).

Workers process chunks in parallel to extract dimension values.
Each worker returns a validated Polars dataframe of extractions.
"""

from pathlib import Path

from prefect import task
from prefect.cache_policies import INPUTS

from causal_agent.utils.data import (
    load_text_chunks as load_text_chunks_util,
    get_worker_chunk_size,
)
from causal_agent.workers.agents import process_chunk, WorkerResult


@task(cache_policy=INPUTS)
def load_worker_chunks(input_path: Path) -> list[str]:
    """Load chunks sized for workers (stage 2)."""
    return load_text_chunks_util(input_path, chunk_size=get_worker_chunk_size())


@task(
    retries=2,
    retry_delay_seconds=10,
)
def populate_dimensions(chunk: str, question: str, schema: dict) -> WorkerResult:
    """Worker extracts dimension values from a chunk.

    Returns:
        WorkerResult containing:
        - output: Validated WorkerOutput with extractions and proposed dimensions
        - dataframe: Polars DataFrame with columns (dimension, value, timestamp)
    """
    return process_chunk(chunk, question, schema)
