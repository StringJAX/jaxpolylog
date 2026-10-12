"""
Real (float64) arguments with ``approx="patch"`` (2026-10-09).

The "safe substitute" constants of the double-where dispatch were built as complex literals with
``dtype=z.dtype`` (``jnp.asarray(0.5 + 0.0j, dtype=z.dtype)``). For a real ``z`` that cast raises
``TypeError`` in current JAX (measured with jax 0.10.0), so ``jax_polylog_vmap(z, s, ...,
approx="patch")`` failed for s = 2, 3 and real z. kahlerjax calls Li_3 and Li_2 that way, with
real arguments whenever the worldsheet-instanton prefactors are real (e.g. no B-field).

Reference: the same call with the complex argument z + 0j, whose accuracy the other tests establish;
for 0 < z < 1, Li_s(z) is real, and so are its derivatives along the real axis. Values are also
checked against mpmath.
"""

import pytest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from mpmath import mp, polylog as mp_polylog

from jaxpolylog.polylogs import jax_polylog_vmap

Z = np.array([1e-12, 1e-6, 0.01, 0.1, 0.3, 0.6, 0.9])
P_RANGE = 60


@pytest.mark.parametrize("s", [3, 2, 1])
def test_real_argument_patch_value(s):
    v_real = np.asarray(jax_polylog_vmap(jnp.asarray(Z), s, P_RANGE, approx="patch"))
    v_cplx = np.asarray(jax_polylog_vmap(jnp.asarray(Z, dtype=complex), s, P_RANGE, approx="patch"))
    assert v_real.dtype == np.float64
    assert np.allclose(v_real, v_cplx.real, rtol=1e-14, atol=0)
    mp.dps = 40
    ref = np.array([float(mp_polylog(s, mp.mpf(float(z)))) for z in Z])
    assert np.max(np.abs(v_real / ref - 1)) < 1e-13


@pytest.mark.parametrize("s", [3, 2])
def test_real_argument_patch_derivatives(s):
    """First and second derivatives along the real axis, through the JVP rule (``_Li_over_z``)."""
    f_real = lambda x: jnp.sum(jax_polylog_vmap(x, s, P_RANGE, approx="patch"))
    f_cplx = lambda x: jnp.sum(jax_polylog_vmap(x.astype(complex), s, P_RANGE, approx="patch")).real
    x = jnp.asarray(Z[2:])
    g_real, g_cplx = np.asarray(jax.grad(f_real)(x)), np.asarray(jax.grad(f_cplx)(x))
    assert np.allclose(g_real, g_cplx, rtol=1e-13, atol=0)
    h_real = np.diag(np.asarray(jax.hessian(f_real)(x)))
    h_cplx = np.diag(np.asarray(jax.hessian(f_cplx)(x)))
    assert np.allclose(h_real, h_cplx, rtol=1e-12, atol=0)
