from pydantic import BaseModel, Field


class DimensionDescription(BaseModel):
    """Description of a dimension with time granularity."""

    text: str = Field(description="What this variable represents")
    time_granularity: str = Field(description="'hourly', 'daily', 'weekly', 'monthly', 'yearly', or 'none'")


class Dimension(BaseModel):
    """A candidate dimension/variable for the causal model."""

    name: str = Field(description="Variable name (eg sleep_quality)")
    description: DimensionDescription = Field(description="What this variable represents and its time granularity")
    dtype: str = Field(description="Data type: 'continuous', 'categorical', 'binary', 'ordinal'")
    is_autocorrelated: bool = Field(description="Whether this variable has temporal autocorrelation")


class CausalEdge(BaseModel):
    """A directed causal edge, optionally lagged."""

    cause: str = Field(description="The cause variable")
    effect: str = Field(description="The effect variable")
    lag: int = Field(default=0, description="Time lag in hours (0=contemporaneous, 24=cause at t-24h affects effect at t)")


class ProposedStructure(BaseModel):
    """Output schema for the structure proposal stage."""

    dimensions: list[Dimension] = Field(
        description="Candidate variables/dimensions to extract from data"
    )
    edges: list[CausalEdge] = Field(
        description="Causal edges (lag=0 for contemporaneous, lag>0 for cross-lagged)"
    )

    def to_networkx(self):
        """Convert to NetworkX DiGraph (compatible with DoWhy)."""
        import networkx as nx

        G = nx.DiGraph()
        # Add all dimension nodes
        for dim in self.dimensions:
            G.add_node(dim.name, **dim.model_dump())
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.cause, edge.effect)
        return G

    def to_edge_list(self) -> list[tuple[str, str]]:
        """Convert to edge list format."""
        return [(e.cause, e.effect) for e in self.edges]
