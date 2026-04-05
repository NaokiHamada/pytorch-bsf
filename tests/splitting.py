"""Tests for torch_bsf.splitting."""

import pytest
import torch

import torch_bsf as tb
import torch_bsf.bezier_simplex as tbbs
from torch_bsf.splitting import (
    longest_edge_criterion,
    max_error_criterion,
    reparametrize,
    split,
    split_by_criterion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_curve(n_values: int = 1) -> tb.BezierSimplex:
    """Degree-1 Bézier curve B(t) = t_1  (for each value dimension)."""
    return tb.BezierSimplex(
        {
            (1, 0): [0.0] * n_values,
            (0, 1): [1.0] * n_values,
        }
    )


def _identity_curve_degree2(n_values: int = 1) -> tb.BezierSimplex:
    """Degree-2 Bézier curve B(t) = t_1  (identity, n_params=2)."""
    return tb.BezierSimplex(
        {
            (2, 0): [0.0] * n_values,
            (1, 1): [0.5] * n_values,
            (0, 2): [1.0] * n_values,
        }
    )


# ---------------------------------------------------------------------------
# split(): validation
# ---------------------------------------------------------------------------


def test_split_wrong_n_params():
    bs = tbbs.rand(n_params=1, n_values=2, degree=3)
    with pytest.raises(ValueError, match="n_params"):
        split(bs, 0, 1)


def test_split_bad_edge_indices():
    bs = tbbs.rand(n_params=3, n_values=2, degree=2)
    with pytest.raises(ValueError, match="i="):
        split(bs, 0, 0)
    with pytest.raises(ValueError, match="i="):
        split(bs, -1, 1)
    with pytest.raises(ValueError, match="i="):
        split(bs, 0, 5)


def test_split_bad_s():
    bs = tbbs.rand(n_params=2, n_values=2, degree=2)
    with pytest.raises(ValueError, match="s="):
        split(bs, 0, 1, s=0.0)
    with pytest.raises(ValueError, match="s="):
        split(bs, 0, 1, s=1.0)
    with pytest.raises(ValueError, match="s="):
        split(bs, 0, 1, s=1.5)


# ---------------------------------------------------------------------------
# split(): structural properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_params", [2, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("n_values", [1, 3])
def test_split_preserves_shape(n_params, degree, n_values):
    """Sub-simplices have the same n_params, degree, and n_values."""
    bs = tbbs.rand(n_params=n_params, n_values=n_values, degree=degree)
    bs_A, bs_B = split(bs, i=0, j=1, s=0.5)
    assert bs_A.n_params == bs.n_params
    assert bs_A.degree == bs.degree
    assert bs_A.n_values == bs.n_values
    assert bs_B.n_params == bs.n_params
    assert bs_B.degree == bs.degree
    assert bs_B.n_values == bs.n_values


@pytest.mark.parametrize("n_params", [2, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_split_preserves_smoothness_weight(n_params, degree):
    bs = tbbs.rand(n_params=n_params, n_values=2, degree=degree)
    # smoothness_weight is a plain float attribute, not used for adjacency
    # unless > 0; use 0.0 to keep the test simple.
    bs.smoothness_weight = 0.0
    bs_A, bs_B = split(bs, i=0, j=n_params - 1, s=0.3)
    assert bs_A.smoothness_weight == bs.smoothness_weight
    assert bs_B.smoothness_weight == bs.smoothness_weight


# ---------------------------------------------------------------------------
# split(): correctness for degree-1 Bézier curve (identity)
# ---------------------------------------------------------------------------


def test_split_degree1_control_points_a():
    """bs_A control points are correct for the identity degree-1 curve."""
    bs = _identity_curve()
    bs_A, _ = split(bs, i=0, j=1, s=0.5)
    assert torch.allclose(bs_A.control_points[(1, 0)], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(bs_A.control_points[(0, 1)], torch.tensor([0.5]), atol=1e-6)


def test_split_degree1_control_points_b():
    """bs_B control points are correct for the identity degree-1 curve."""
    _, bs_B = split(_identity_curve(), i=0, j=1, s=0.5)
    assert torch.allclose(bs_B.control_points[(1, 0)], torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(bs_B.control_points[(0, 1)], torch.tensor([1.0]), atol=1e-6)


def test_split_degree2_control_points_a():
    """bs_A control points are correct for the identity degree-2 curve."""
    bs_A, _ = split(_identity_curve_degree2(), i=0, j=1, s=0.5)
    assert torch.allclose(bs_A.control_points[(2, 0)], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(bs_A.control_points[(1, 1)], torch.tensor([0.25]), atol=1e-6)
    assert torch.allclose(bs_A.control_points[(0, 2)], torch.tensor([0.5]), atol=1e-6)


def test_split_degree2_control_points_b():
    """bs_B control points are correct for the identity degree-2 curve."""
    _, bs_B = split(_identity_curve_degree2(), i=0, j=1, s=0.5)
    assert torch.allclose(bs_B.control_points[(2, 0)], torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(bs_B.control_points[(1, 1)], torch.tensor([0.75]), atol=1e-6)
    assert torch.allclose(bs_B.control_points[(0, 2)], torch.tensor([1.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# split(): continuity at the split point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_params", [2, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("s", [0.25, 0.5, 0.75])
def test_split_continuity_at_new_vertex(n_params, degree, s):
    """Both sub-simplices evaluate to the same value at the new vertex."""
    bs = tbbs.rand(n_params=n_params, n_values=3, degree=degree)
    i, j = 0, 1
    bs_A, bs_B = split(bs, i=i, j=j, s=s)

    # New vertex in A: e_j  (all weight at j in A's local coords)
    # New vertex in B: e_i  (all weight at i in B's local coords)
    t_new_A = torch.zeros(1, n_params)
    t_new_A[0, j] = 1.0
    t_new_B = torch.zeros(1, n_params)
    t_new_B[0, i] = 1.0

    with torch.no_grad():
        val_A = bs_A(t_new_A)
        val_B = bs_B(t_new_B)

    assert torch.allclose(val_A, val_B, atol=1e-5), (
        f"Split point mismatch: {val_A} != {val_B}"
    )


# ---------------------------------------------------------------------------
# split(): reproduces original on each sub-domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("s", [0.3, 0.5, 0.7])
def test_split_reproduces_original_curve(degree, s):
    """For a Bézier curve, sub-simplices reproduce the original on their domain."""
    bs = tbbs.rand(n_params=2, n_values=3, degree=degree)
    i, j = 0, 1
    bs_A, bs_B = split(bs, i=i, j=j, s=s)

    # Sample 20 points in sub-domain A  (t_j / (t_i + t_j) <= s → t_j <= s * t_i/(1-s)... simplified: s * t_i >= (1-s) * t_j)
    alpha_vals = torch.linspace(0.0, s, 20)  # t_j = alpha, t_i = 1 - alpha
    params_A = torch.stack([1.0 - alpha_vals, alpha_vals], dim=1)  # t_j/(t_i+t_j) = alpha <= s

    u_A, _ = reparametrize(params_A, i, j, s, "A")
    with torch.no_grad():
        orig_vals = bs(params_A)
        split_vals = bs_A(u_A)

    assert torch.allclose(orig_vals, split_vals, atol=1e-5), (
        f"Sub-simplex A does not reproduce original: max diff = "
        f"{(orig_vals - split_vals).abs().max().item()}"
    )

    # Sample 20 points in sub-domain B  (t_j >= s)
    alpha_vals = torch.linspace(s, 1.0, 20)
    params_B = torch.stack([1.0 - alpha_vals, alpha_vals], dim=1)

    u_B, _ = reparametrize(params_B, i, j, s, "B")
    with torch.no_grad():
        orig_vals = bs(params_B)
        split_vals = bs_B(u_B)

    assert torch.allclose(orig_vals, split_vals, atol=1e-5), (
        f"Sub-simplex B does not reproduce original: max diff = "
        f"{(orig_vals - split_vals).abs().max().item()}"
    )


@pytest.mark.parametrize("n_params", [3, 4])
@pytest.mark.parametrize("degree", [1, 2])
def test_split_reproduces_original_higher_dim(n_params, degree):
    """Sub-simplices reproduce the original for higher-dimensional parameter spaces."""
    bs = tbbs.rand(n_params=n_params, n_values=2, degree=degree)
    i, j, s = 0, 1, 0.5
    bs_A, bs_B = split(bs, i=i, j=j, s=s)

    from torch_bsf.sampling import simplex_random

    torch.manual_seed(42)
    params_all = simplex_random(n_params, n_samples=50)

    ti = params_all[:, i]
    tj = params_all[:, j]
    denom = ti + tj

    # Sub-domain A: t_j / (t_i + t_j) <= s  (including t_i+t_j=0)
    mask_A = (denom == 0) | (tj <= s * denom)
    if mask_A.any():
        u_A, _ = reparametrize(params_all[mask_A], i, j, s, "A")
        with torch.no_grad():
            orig = bs(params_all[mask_A])
            approx = bs_A(u_A)
        assert torch.allclose(orig, approx, atol=1e-5)

    # Sub-domain B: t_j / (t_i + t_j) >= s
    mask_B = (denom == 0) | (tj >= s * denom)
    if mask_B.any():
        u_B, _ = reparametrize(params_all[mask_B], i, j, s, "B")
        with torch.no_grad():
            orig = bs(params_all[mask_B])
            approx = bs_B(u_B)
        assert torch.allclose(orig, approx, atol=1e-5)


# ---------------------------------------------------------------------------
# reparametrize()
# ---------------------------------------------------------------------------


def test_reparametrize_t_not_2d():
    """reparametrize raises ValueError when t is not a 2-D tensor."""
    t_1d = torch.tensor([0.5, 0.5])
    with pytest.raises(ValueError, match="2-D"):
        reparametrize(t_1d, i=0, j=1, s=0.5, subsimplex="A")
    t_3d = torch.tensor([[[0.5, 0.5]]])
    with pytest.raises(ValueError, match="2-D"):
        reparametrize(t_3d, i=0, j=1, s=0.5, subsimplex="A")


def test_reparametrize_t_too_few_columns():
    """reparametrize raises ValueError when t has fewer than 2 columns."""
    t = torch.tensor([[0.5]])
    with pytest.raises(ValueError, match="n_params >= 2"):
        reparametrize(t, i=0, j=1, s=0.5, subsimplex="A")


def test_reparametrize_t_not_floating():
    """reparametrize raises ValueError when t is not a floating-point tensor."""
    t_int = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    with pytest.raises(ValueError, match="floating-point"):
        reparametrize(t_int, i=0, j=1, s=0.5, subsimplex="A")
    t_bool = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="floating-point"):
        reparametrize(t_bool, i=0, j=1, s=0.5, subsimplex="A")


def test_reparametrize_bad_s():
    """reparametrize raises ValueError when s is not in (0, 1)."""
    t = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="s="):
        reparametrize(t, i=0, j=1, s=0.0, subsimplex="A")
    with pytest.raises(ValueError, match="s="):
        reparametrize(t, i=0, j=1, s=1.0, subsimplex="A")
    with pytest.raises(ValueError, match="s="):
        reparametrize(t, i=0, j=1, s=-0.1, subsimplex="B")


def test_reparametrize_bad_edge_indices():
    """reparametrize raises ValueError for invalid edge indices."""
    t = torch.tensor([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError, match="i="):
        reparametrize(t, i=0, j=0, s=0.5, subsimplex="A")
    with pytest.raises(ValueError, match="i="):
        reparametrize(t, i=-1, j=1, s=0.5, subsimplex="A")
    with pytest.raises(ValueError, match="i="):
        reparametrize(t, i=0, j=5, s=0.5, subsimplex="A")


def test_reparametrize_bad_subsimplex():
    """reparametrize raises ValueError for invalid subsimplex argument."""
    t = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="subsimplex"):
        reparametrize(t, i=0, j=1, s=0.5, subsimplex="C")


def test_reparametrize_a_domain():
    """Points in sub-domain A get a valid mask and remapped coordinates."""
    # t = (0.8, 0.2) → t_j/(t_i+t_j) = 0.2 ≤ 0.5  → belongs to A
    t = torch.tensor([[0.8, 0.2]])
    u, mask = reparametrize(t, i=0, j=1, s=0.5, subsimplex="A")
    assert mask[0].item()
    assert torch.allclose(u[0, 1], torch.tensor(0.4), atol=1e-6)  # t_j/s = 0.4
    assert torch.allclose(u[0, 0], torch.tensor(0.6), atol=1e-6)  # 0.8 - 0.5*0.4


def test_reparametrize_b_domain():
    """Points in sub-domain B get a valid mask and remapped coordinates."""
    # t = (0.3, 0.7) → t_j/(t_i+t_j) = 0.7 ≥ 0.5  → belongs to B
    t = torch.tensor([[0.3, 0.7]])
    u, mask = reparametrize(t, i=0, j=1, s=0.5, subsimplex="B")
    assert mask[0].item()
    assert torch.allclose(u[0, 0], torch.tensor(0.6), atol=1e-6)  # t_i/(1-s) = 0.6
    assert torch.allclose(u[0, 1], torch.tensor(0.4), atol=1e-6)  # 0.7 - 0.5*0.6


def test_reparametrize_zero_face_belongs_to_both():
    """A point on the opposite face (t_i = t_j = 0) belongs to both sub-domains."""
    t = torch.tensor([[0.0, 0.0, 1.0]])
    _, mask_A = reparametrize(t, i=0, j=1, s=0.5, subsimplex="A")
    _, mask_B = reparametrize(t, i=0, j=1, s=0.5, subsimplex="B")
    assert mask_A[0].item()
    assert mask_B[0].item()


def test_reparametrize_coordinates_sum_to_one():
    """Re-parameterised coordinates sum to 1 (up to floating-point error)."""
    torch.manual_seed(7)
    from torch_bsf.sampling import simplex_random

    t = simplex_random(n_params=4, n_samples=50)
    for sub in ("A", "B"):
        u, mask = reparametrize(t, i=0, j=1, s=0.5, subsimplex=sub)
        if mask.any():
            row_sums = u[mask].sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# longest_edge_criterion()
# ---------------------------------------------------------------------------


def test_longest_edge_criterion_wrong_n_params():
    bs = tbbs.rand(n_params=1, n_values=2, degree=1)
    with pytest.raises(ValueError, match="n_params"):
        longest_edge_criterion(bs)


def test_longest_edge_criterion_returns_valid_edge():
    bs = tbbs.rand(n_params=4, n_values=3, degree=2)
    i, j, s = longest_edge_criterion(bs)
    assert 0 <= i < j < bs.n_params
    assert 0.0 < s < 1.0


def test_longest_edge_criterion_identity_curve():
    """For the identity curve (edge length = 1), picks edge (0, 1)."""
    bs = _identity_curve(n_values=1)
    i, j, s = longest_edge_criterion(bs)
    assert i == 0 and j == 1
    assert s == 0.5


def test_longest_edge_criterion_custom_s():
    bs = tbbs.rand(n_params=3, n_values=2, degree=1)
    _, _, s = longest_edge_criterion(bs, s=0.3)
    assert s == 0.3


# ---------------------------------------------------------------------------
# max_error_criterion()
# ---------------------------------------------------------------------------


def test_max_error_criterion_returns_valid_edge():
    torch.manual_seed(0)
    bs = tbbs.rand(n_params=2, n_values=1, degree=1)
    params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    values = torch.tensor([[0.0], [0.5], [1.0]])
    criterion = max_error_criterion(params, values, grid_size=5)
    i, j, s = criterion(bs)
    assert 0 <= i < bs.n_params
    assert 0 <= j < bs.n_params
    assert i != j
    assert 0.0 < s < 1.0


def test_max_error_criterion_bad_grid_size():
    params = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="grid_size must be >= 1"):
        max_error_criterion(params, values, grid_size=0)


def test_max_error_criterion_wrong_n_params():
    bs = tbbs.rand(n_params=1, n_values=2, degree=1)
    params = torch.tensor([[1.0]])
    values = torch.tensor([[0.0]])
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match="n_params"):
        criterion(bs)


def test_max_error_criterion_params_not_2d():
    bs = tbbs.rand(n_params=2, n_values=1, degree=1)
    params = torch.tensor([1.0, 0.0, 0.5])  # 1-D
    values = torch.tensor([[0.0], [1.0], [0.5]])
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match=r"`params` must be a 2D tensor"):
        criterion(bs)


def test_max_error_criterion_values_not_2d():
    bs = tbbs.rand(n_params=2, n_values=1, degree=1)
    params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    values = torch.tensor([0.0, 0.5, 1.0])  # 1-D
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match=r"`values` must be a 2D tensor"):
        criterion(bs)


def test_max_error_criterion_mismatched_n():
    bs = tbbs.rand(n_params=2, n_values=1, degree=1)
    params = torch.tensor([[1.0, 0.0], [0.5, 0.5]])  # N=2
    values = torch.tensor([[0.0], [0.5], [1.0]])  # N=3
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match="same number of samples"):
        criterion(bs)


def test_max_error_criterion_wrong_params_shape():
    bs = tbbs.rand(n_params=2, n_values=1, degree=1)
    params = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.3, 0.2]])  # wrong n_params=3
    values = torch.tensor([[0.0], [0.5]])
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match=r"`params` must have shape"):
        criterion(bs)


def test_max_error_criterion_wrong_values_shape():
    bs = tbbs.rand(n_params=2, n_values=1, degree=1)
    params = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    values = torch.tensor([[0.0, 1.0], [0.5, 0.6]])  # wrong n_values=2
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match=r"`values` must have shape"):
        criterion(bs)


# ---------------------------------------------------------------------------
# split_by_criterion()
# ---------------------------------------------------------------------------


def test_split_by_criterion_longest_edge():
    bs = tbbs.rand(n_params=3, n_values=2, degree=2)
    bs_A, bs_B = split_by_criterion(bs, longest_edge_criterion)
    assert bs_A.n_params == bs.n_params
    assert bs_B.n_params == bs.n_params


def test_split_by_criterion_max_error():
    torch.manual_seed(1)
    from torch_bsf.sampling import simplex_random

    n_params, n_values = 3, 2
    params = simplex_random(n_params, n_samples=20)
    bs = tbbs.rand(n_params=n_params, n_values=n_values, degree=2)
    values = bs(params).detach()
    criterion = max_error_criterion(params, values, grid_size=3)
    bs_A, bs_B = split_by_criterion(bs, criterion)
    assert bs_A.degree == bs.degree
    assert bs_B.degree == bs.degree


def test_splitting_module_public_api():
    import torch_bsf.splitting as splitting

    assert hasattr(splitting, "split")
    assert hasattr(splitting, "reparametrize")
    assert hasattr(splitting, "SplitCriterion")
    assert hasattr(splitting, "longest_edge_criterion")
    assert hasattr(splitting, "max_error_criterion")
    assert hasattr(splitting, "split_by_criterion")
