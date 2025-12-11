"""Schemas for worker LLM outputs."""

from enum import Enum

from pydantic import BaseModel, Field


class EdgeSuggestionType(str, Enum):
    """Type of edge modification suggested by a worker."""

    ADD = "add"  # Suggest adding a new edge
    REMOVE = "remove"  # Suggest removing an existing edge
    REVERSE = "reverse"  # Suggest reversing edge direction


class EdgeSuggestion(BaseModel):
    """A suggested modification to the causal graph."""

    type: EdgeSuggestionType = Field(description="Type of modification")
    cause: str = Field(description="Name of cause variable")
    effect: str = Field(description="Name of effect variable")
    reasoning: str = Field(description="Evidence from this chunk supporting the suggestion")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in suggestion (0-1) based on evidence strength",
    )


class DimensionSuggestion(BaseModel):
    """A suggested new dimension (confounder) found in local data."""

    name: str = Field(description="Variable name")
    description: str = Field(description="What this variable represents")
    reasoning: str = Field(description="Evidence from this chunk for why this matters causally")
    affects: list[str] = Field(description="Which existing dimensions this would affect")


class DimensionExtraction(BaseModel):
    """Extracted data for a single dimension from a chunk."""

    dimension: str = Field(description="Name of the dimension")
    values: list[str] = Field(
        description="Extracted values/observations from the chunk for this dimension"
    )
    timestamps: list[str] = Field(
        default_factory=list,
        description="Associated timestamps if identifiable (ISO format preferred)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in extraction quality (0-1)",
    )
    notes: str = Field(
        default="",
        description="Any caveats or notes about the extraction",
    )


class WorkerOutput(BaseModel):
    """Complete output from a worker processing a single chunk."""

    chunk_id: str = Field(description="Identifier for the processed chunk")
    extractions: list[DimensionExtraction] = Field(
        default_factory=list,
        description="Extracted data for each dimension",
    )
    edge_suggestions: list[EdgeSuggestion] = Field(
        default_factory=list,
        description="Suggested modifications to the causal graph",
    )
    dimension_suggestions: list[DimensionSuggestion] = Field(
        default_factory=list,
        description="Suggested new dimensions (confounders) found in local data",
    )
    chunk_summary: str = Field(
        description="Brief summary of what this chunk contains temporally and topically",
    )
