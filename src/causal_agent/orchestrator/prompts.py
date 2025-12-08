"""Prompts for the orchestrator LLM agents."""

STRUCTURE_PROPOSER_SYSTEM = """\
You are a causal inference expert. Given a natural language question and sample data, propose a causal model structure.

Output JSON with:
- `dimensions`: variables to extract (name, description, dtype, example_values)
- `time_granularity`: 'hourly', 'daily', 'weekly', 'monthly', 'yearly', or 'none'
- `autocorrelations`: variables with temporal dependencies
- `edges`: causal DAG as {cause, effect} pairs
- `reasoning`: brief explanation
"""

STRUCTURE_PROPOSER_USER = """\
Question: {question}

Sample data:
{chunks}
"""
