"""Semantic-guided refinement of feed-forward 3D reconstructions."""

from beartype.claw import beartype_this_package

beartype_this_package()


def main() -> None:
    """Run monocular refinement from a config file."""
    from semantic_3d_refinement.refine import main as refine_main  # noqa: PLC0415

    refine_main()
