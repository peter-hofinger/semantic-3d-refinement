"""Semantic-guided refinement of feed-forward 3D reconstruction."""

from beartype.claw import beartype_packages

# Runtime type checks for the refinement code, which annotates with real imports.
# The other stages import their annotation types under TYPE_CHECKING only, which
# beartype cannot resolve, so they are left out.
beartype_packages(
    (
        "src.analysis",
        "src.refinement.monocular",
        "src.refinement.multiview",
        "src.refinement.normals",
        "src.refinement.inputs",
        "src.evaluation.depth",
        "src.loaders.eth3d",
        "src.loaders.synthetic",
        "src.loaders.scannetpp.scan",
    )
)


def main() -> None:
    """Print a greeting (placeholder entry point)."""
    print("Hello from semantic-3d-refinement!")  # noqa: T201
