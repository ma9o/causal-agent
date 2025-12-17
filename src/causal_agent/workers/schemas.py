"""Schemas for worker LLM outputs."""

from typing import Any

import polars as pl
from pydantic import BaseModel, Field


class Extraction(BaseModel):
    """A single extracted observation for a dimension."""

    dimension: str = Field(description="Name of the dimension")
    value: int | float | bool | str | None = Field(
        description="Extracted value of the correct datatype"
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp if identifiable",
    )


class ProposedDimension(BaseModel):
    """A suggested new dimension found in local data."""

    name: str = Field(description="Variable name")
    description: str = Field(description="What this variable represents")
    evidence: str = Field(description="What was seen in this chunk")
    relevant_because: str = Field(description="How it connects to the causal question")
    not_already_in_dimensions_because: str = Field(
        description="Why it needs to be added and why existing dimensions don't capture it"
    )


class WorkerOutput(BaseModel):
    """Complete output from a worker processing a single chunk."""

    extractions: list[Extraction] = Field(
        default_factory=list,
        description="Extracted observations for dimensions",
    )
    proposed_dimensions: list[ProposedDimension] | None = Field(
        default=None,
        description="Suggested new dimensions if something important is missing",
    )

    def to_dataframe(self) -> pl.DataFrame:
        """Convert extractions to a Polars DataFrame.

        Returns:
            DataFrame with columns: dimension, value, timestamp
            Value column uses pl.Object to preserve mixed types.
        """
        if not self.extractions:
            return pl.DataFrame(
                schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8}
            )

        return pl.DataFrame(
            [
                {
                    "dimension": e.dimension,
                    "value": e.value,
                    "timestamp": e.timestamp,
                }
                for e in self.extractions
            ],
            schema={"dimension": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
        )


def _check_dtype_match(value: Any, expected_dtype: str) -> bool:
    """Check if a value matches the expected measurement_dtype."""
    if value is None:
        return True  # None is always acceptable

    dtype_checks = {
        "continuous": lambda v: isinstance(v, (int, float)),
        "binary": lambda v: isinstance(v, bool) or v in (0, 1, "0", "1", "true", "false", "True", "False"),
        "count": lambda v: isinstance(v, int) or (isinstance(v, float) and v == int(v) and v >= 0),
        "ordinal": lambda v: isinstance(v, (int, float, str)),  # Flexible - can be numeric or string
        "categorical": lambda v: isinstance(v, str),
    }

    check = dtype_checks.get(expected_dtype)
    if check is None:
        return True  # Unknown dtype, don't fail
    return check(value)


def validate_worker_output(
    data: dict,
    schema: dict,
) -> tuple[WorkerOutput | None, list[str]]:
    """Validate worker output dict, collecting ALL errors instead of failing on first.

    Args:
        data: Dictionary to validate as WorkerOutput
        schema: The DSEM schema dict (with dimensions) to validate against

    Returns:
        Tuple of (validated output or None, list of error messages)
    """
    errors = []

    # Basic structure checks
    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    extractions = data.get("extractions", [])
    proposed_dimensions = data.get("proposed_dimensions")

    if not isinstance(extractions, list):
        errors.append("'extractions' must be a list")
        extractions = []

    if proposed_dimensions is not None and not isinstance(proposed_dimensions, list):
        errors.append("'proposed_dimensions' must be a list or null")
        proposed_dimensions = None

    # Build set of valid observed dimension names and their dtypes
    dimensions = schema.get("dimensions", [])
    observed_dims = {
        dim.get("name"): dim.get("measurement_dtype")
        for dim in dimensions
        if dim.get("observability") == "observed"
    }

    # Validate each extraction
    valid_extractions = []
    for i, ext_data in enumerate(extractions):
        if not isinstance(ext_data, dict):
            errors.append(f"extractions[{i}]: must be a dictionary")
            continue

        dim_name = ext_data.get("dimension", "<missing>")
        value = ext_data.get("value")

        # Check dimension exists and is observed
        if dim_name not in observed_dims:
            valid_dim_names = ", ".join(sorted(observed_dims.keys()))
            errors.append(
                f"extractions[{i}]: dimension '{dim_name}' not in observed dimensions. "
                f"Valid dimensions: {valid_dim_names}"
            )
            continue

        # Check dtype match
        expected_dtype = observed_dims[dim_name]
        if not _check_dtype_match(value, expected_dtype):
            errors.append(
                f"extractions[{i}]: value {value!r} for '{dim_name}' doesn't match "
                f"expected dtype '{expected_dtype}'"
            )
            continue

        # Validate via Pydantic
        try:
            ext = Extraction.model_validate(ext_data)
            valid_extractions.append(ext)
        except Exception as e:
            errors.append(f"extractions[{i}] ({dim_name}): {e}")

    # Validate proposed dimensions if present
    valid_proposed = None
    if proposed_dimensions is not None:
        valid_proposed = []
        for i, prop_data in enumerate(proposed_dimensions):
            if not isinstance(prop_data, dict):
                errors.append(f"proposed_dimensions[{i}]: must be a dictionary")
                continue

            name = prop_data.get("name", "<missing>")

            # Check not already in schema
            all_dim_names = {dim.get("name") for dim in dimensions}
            if name in all_dim_names:
                errors.append(
                    f"proposed_dimensions[{i}]: '{name}' already exists in schema"
                )
                continue

            try:
                prop = ProposedDimension.model_validate(prop_data)
                valid_proposed.append(prop)
            except Exception as e:
                errors.append(f"proposed_dimensions[{i}] ({name}): {e}")

    # If no errors, build and return the output
    if not errors:
        try:
            output = WorkerOutput(
                extractions=valid_extractions,
                proposed_dimensions=valid_proposed if valid_proposed else None,
            )
            return output, []
        except Exception as e:
            errors.append(f"Final validation failed: {e}")

    return None, errors
