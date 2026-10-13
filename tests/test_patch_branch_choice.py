"""
Branch choice of ``approx="patch"`` off the positive real axis (2026-10-12).

"patch" evaluates Li_s(z), s >= 2, either by the power series Σ z^k/k^s (truncation error ~ |z|^N)
or by the expansion in μ = log z (error ~ t^N, t = |μ|/(2π)). It used the power series whenever
|z| < 1 and t >= 0.2322, a threshold that solves |z| = t on the positive real axis only. For
arg z ≠ 0 the threshold says nothing about |z|: at z = -|z|, t >= 1/2 for every |z|, so the power
series was used up to z -> -1, where N terms leave an error ~ 1/(2 N^s). kahlerjax evaluates its
instanton sums at z < 0 for every curve with e^{2πi q·b} = -1; on the dS examples of arXiv:2406.13751
Li_2 was off by up to 5.4e-3 near z = -1 with p_range = 10. Off the positive real axis the power
series is now used only where its crude error estimate |z|^N is the smaller one, |z| <= t (default
pval); on the positive real axis the rule is exactly the old one.

Reference: mpmath, for values and for the derivative d Li_s / dz = Li_{s-1}(z)/z (JVP rule,
``_Li_over_z``). The points ``Z`` are inside the unit disk; at each of them except z = -0.6 the
old rule took the power series and missed the tolerance; the new rule takes the expansion in
log z there. The points ``Z_SERIES`` lie off the axis where the power series must be used (at the
small ones the expansion in log z diverges). On the positive real axis the rule is exactly the old
one, by construction (2026-10-13: before, it agreed only in exact arithmetic; for pval < 0.2322
rounding could flip the branch, e.g. pval = 0.14 at z = 0.41492979447371536).
"""

import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from mpmath import mp, polylog as mp_polylog

from jaxpolylog.polylogs import jax_polylog, jax_polylog_vmap, _PVAL_OPTIMAL

P_RANGE = 60
Z = np.array([-0.6, -0.9, -0.99, -0.996, 0.95j, 0.9*np.exp(3j*np.pi/4), 0.98*np.exp(2j*np.pi/3)])
Z_SERIES = np.array([-0.05, -0.4, 0.3j, 1e-3*np.exp(1j), 1e-3*np.exp(2.5j), 0.2*np.exp(2j)])


def ref(s, z):
    mp.dps = 40
    return np.array([complex(mp_polylog(s, mp.mpc(complex(x)))) for x in z])


@pytest.mark.parametrize("s", [2, 3, 4])
def test_patch_value_off_positive_axis(s):
    got = np.asarray(jax_polylog_vmap(jnp.asarray(Z), s, P_RANGE, approx="patch"))
    R = ref(s, Z)
    assert np.max(np.abs(got - R) / np.abs(R)) < 1e-12


@pytest.mark.parametrize("s", [2, 3, 4])
def test_patch_value_where_the_power_series_is_needed(s):
    from jaxpolylog.polylogs import _patch_use_inf
    assert np.all(np.asarray(_patch_use_inf(jnp.asarray(Z_SERIES), _PVAL_OPTIMAL)))
    got = np.asarray(jax_polylog_vmap(jnp.asarray(Z_SERIES), s, P_RANGE, approx="patch"))
    R = ref(s, Z_SERIES)
    assert np.max(np.abs(got - R) / np.abs(R)) < 1e-12


@pytest.mark.parametrize("s", [3, 4])
def test_patch_derivative_off_positive_axis(s):
    """For s >= 3 the derivative Li_{s-1}(z)/z goes through the same branch choice."""
    z = jnp.asarray(Z)
    _, dz = jax.jvp(lambda x: jax_polylog_vmap(x, s, P_RANGE, approx="patch"), (z,), (jnp.ones_like(z),))
    R = ref(s - 1, Z) / Z
    assert np.max(np.abs(np.asarray(dz) - R) / np.abs(R)) < 1e-12


@pytest.mark.parametrize("pval", [0.10, 0.14, _PVAL_OPTIMAL, 0.40])
def test_positive_axis_unchanged(pval):
    """On 0 < z < 1 the branch is still "inf" iff t >= pval (bitwise the same numbers), including a
    point where rounding made the off-axis rule disagree (t = 0.14 exactly, pval = 0.14)."""
    for s in (2, 3):
        for x in (0.01, 0.1, 0.2, 0.25, 0.4, 0.7, 0.9, 0.99, 0.41492979447371536):
            z = jnp.asarray(x + 0j)
            t = abs(np.log(x)) / (2*np.pi)
            branch = "inf" if t >= pval else "zero"
            got = complex(jax_polylog(z, s, P_RANGE, "patch", pval))
            assert got == complex(jax_polylog(z, s, P_RANGE, branch, pval)), (s, x, pval)


def test_custom_pval_off_axis():
    """Off the axis a ``pval`` below the default moves points to the power series where
    |z| <= e^{-2π pval} (and t >= pval), as documented: z = 0.6 e^{0.5i} has t ≈ 0.114."""
    z = jnp.asarray(0.6 * np.exp(0.5j))
    t = abs(np.log(complex(z))) / (2 * np.pi)
    assert 0.05 <= t < 0.6 <= np.exp(-2 * np.pi * 0.05)
    for s in (2, 3):
        assert complex(jax_polylog(z, s, P_RANGE, "patch", 0.05)) == complex(jax_polylog(z, s, P_RANGE, "inf", 0.05))
        assert complex(jax_polylog(z, s, P_RANGE, "patch")) == complex(jax_polylog(z, s, P_RANGE, "zero"))
