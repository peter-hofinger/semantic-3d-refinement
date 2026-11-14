from collections.abc import Callable

import torch

from src.backbone.solv3r import (
    MAX_CALLS,
    MIN_CALLS,
    Solv3RBackbone,
    relative_change,
    run_solver,
)


def halve_towards(target: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda z: (z + target) / 2


def test_zero_calls_return_the_start_state() -> None:
    z = torch.randn(3, 5)
    out, used = run_solver(halve_towards(torch.ones(3, 5)), z, 0, None)
    assert used == 0
    assert torch.equal(out, z)


def test_fixed_calls_reach_the_known_solution() -> None:
    target, z = torch.ones(4), torch.zeros(4)
    out, used = run_solver(halve_towards(target), z, 10, None)
    assert used == 10
    torch.testing.assert_close(out, target * (1 - 0.5**10))


def test_adaptive_solve_stops_at_the_first_small_change() -> None:
    target, z = torch.ones(4), torch.zeros(4)
    out, used = run_solver(halve_towards(target), z, 16, 0.02)
    # change at call n is 0.5**n / (1 - 0.5**(n - 1)): 0.033 at n = 5, 0.016 at n = 6
    assert used == 6
    torch.testing.assert_close(out, target * (1 - 0.5**6))


def test_adaptive_solve_keeps_its_call_bounds() -> None:
    assert run_solver(lambda z: z, torch.ones(2), 16, 0.02)[1] == MIN_CALLS
    assert run_solver(lambda z: 2 * z, torch.ones(2), 16, 0.02)[1] == MAX_CALLS


def test_relative_change_is_scale_invariant() -> None:
    z, z_next = torch.randn(10), torch.randn(10)
    assert abs(relative_change(z, z_next) - relative_change(3 * z, 3 * z_next)) < 1e-6


def test_cache_tag_names_the_solve() -> None:
    assert Solv3RBackbone().tag == "solv3r_c16"
    assert Solv3RBackbone(adaptive=0.02).tag == "solv3r_a0.02"
