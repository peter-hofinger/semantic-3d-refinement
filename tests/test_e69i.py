"""E69i's ranking scores, plane reference and receptive radius on known answers."""

import numpy as np

from experiments.scripts.e69i import (
    average_precision,
    plane_share,
    receptive_radius,
    roc_auc,
)


def test_roc_auc_known_values() -> None:
    """Perfect ranking gives 1, reversed 0, a tie 0.5, and one swap of 2 x 2 0.75."""
    label = np.array([False, False, True, True])
    assert roc_auc(np.array([0.0, 1, 2, 3]), label) == 1.0
    assert roc_auc(np.array([3.0, 2, 1, 0]), label) == 0.0
    assert roc_auc(np.zeros(4), label) == 0.5
    assert roc_auc(np.array([0.0, 2, 1, 3]), label) == 0.75


def test_average_precision_known_values() -> None:
    """Positives ranked 1st and 3rd: precisions 1 and 2/3, mean 5/6; perfect gives 1."""
    label = np.array([True, False, True, False])
    assert np.isclose(average_precision(np.array([4.0, 3, 2, 1]), label), 5 / 6)
    assert average_precision(np.array([4.0, 1, 3, 2]), label) == 1.0


def test_plane_share_bounds() -> None:
    """A plane cuts at least one 8^3 child and at most a few layers' worth of them."""
    share = plane_share(4, np.random.default_rng(0), n=500)
    assert 1 / 512 <= share <= 3 / 8


def test_receptive_radius_three_levels() -> None:
    """Widths of three levels: convs 2+4+8, pools 1+2, decoder 4+8 cells: 29."""
    assert receptive_radius(3) == 29
    assert receptive_radius(1) == 2
