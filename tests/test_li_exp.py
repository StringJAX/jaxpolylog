"""
``li_exp(k, w) = Li_k(e^{2πiw})`` (2026-10-12) against mpmath.

Reference: ``mpmath.polylog(k, exp(2πiw))`` with w taken exactly from its float value, at 40 digits
plus the number of digits by which x = e^{2πiw} is small (forming 1 - x at 40 digits loses a tiny
x; mpmath's own Li_1 is wrong there otherwise).

Error measure on the grids: |li_exp − ref| / max(|ref|, |w|·|dLi_k/dw|), a backward-error measure:
the second term is the change of Li_k when w is perturbed by one unit in its last place. It allows
for the conditioning of x = e^{2πiw} at large |Im w| (a relative error of about 2π|Im w|·1e-16 that
no algorithm receiving w in floating point can avoid) and near the irrational zeros of Li_{-3}.
Where w is exact relative to a special point, plain relative error is required: on the physical line
Re w = ±1/2 (half-integral B-field, x near -1, where Li_{-2} vanishes), see
``test_forward_accuracy_at_half_integral_re_w``.

Tolerances: 1e-13 for k = 2, 3 and 1e-14 for k <= 1 (plan P1 acceptance), on grids over
Re w ∈ {0, 0.13, 1/4, 1/2, 0.71, -0.37, 0.042} × Im w ∈ ±[1e-15, 50], the conifold regime
|w| ∈ [1e-15, 1e-2] at all phases, the region switches, and the branch cut.
"""

import math

import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from mpmath import mp
import mpmath

from jaxpolylog import li_exp

ORDERS = list(range(-3, 4))
TOL = {k: (1e-13 if k >= 2 else 1e-14) for k in ORDERS}
RE = [0.0, 0.13, 0.25, 0.5, 0.71, -0.37, 0.042]


def _mp_li(k, w):
    w = complex(w)
    mp.dps = 40 + max(0, math.ceil(2 * math.pi * w.imag / math.log(10)))
    x = mpmath.exp(2j * mp.pi * mp.mpc(w.real, w.imag))
    return mpmath.polylog(k, x)


def ref(k, ws):
    return np.array([complex(_mp_li(k, w)) for w in ws])


def scale(k, ws, r):
    d = np.array([abs(complex(2 * mp.pi * _mp_li(k - 1, w))) for w in ws])
    return np.maximum(np.abs(r), np.abs(np.asarray(ws)) * d)


def max_err(k, ws):
    ws = np.asarray(ws, dtype=complex)
    got = np.asarray(li_exp(k, jnp.asarray(ws)))
    r = ref(k, ws)
    e = np.abs(got - r) / scale(k, ws, r)
    i = int(np.argmax(e))
    return e[i], ws[i]


@pytest.mark.parametrize("k", ORDERS)
def test_grid_upper_half_plane(k):
    ws = [complex(a, b) for a in RE for b in np.logspace(-15, np.log10(50), 25)]
    e, w = max_err(k, ws)
    assert e < TOL[k], (e, w)


@pytest.mark.parametrize("k", ORDERS)
def test_grid_lower_half_plane(k):
    """Im w < 0 (|x| > 1): for k = 2, 3 the expansion (|w| <= 1/2) and the inversion relation
    (|w| > 1/2); for k <= 1 the closed forms in x (|x| <= 2) and in 1/x (|x| > 2)."""
    ws = [complex(a, -b) for a in RE for b in np.logspace(-15, np.log10(50), 25)]
    e, w = max_err(k, ws)
    assert e < TOL[k], (e, w)


@pytest.mark.parametrize("k", ORDERS)
def test_conifold_regime(k):
    """|w| in [1e-15, 1e-2] at phases all around, including w = ±i|w| and w real."""
    phases = np.linspace(-np.pi, np.pi, 17)
    ws = [r * np.exp(1j * p) for r in np.logspace(-15, -2, 14) for p in phases]
    e, w = max_err(k, ws)
    assert e < TOL[k], (e, w)


def _boundary_points():
    """Points on both sides of the region switches: |x| = |w| (upper half plane), |w| = 1/2 below."""
    pts = []
    for a in (0.0, 0.1, 0.25, 0.4, 0.5):
        lo, hi = 1e-6, 1.0                       # |x| - |w| changes sign on (lo, hi) along Im w
        f = lambda y: math.exp(-2 * math.pi * y) - abs(complex(a, y))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        y = 0.5 * (lo + hi)
        pts.append((complex(a, y * (1 - 1e-12)), complex(a, y * (1 + 1e-12))))
    for a in (0.0, 0.2, 0.45):
        y = -math.sqrt(0.25 - a * a)
        pts.append((complex(a, y * (1 - 1e-12)), complex(a, y * (1 + 1e-12))))
    return pts


@pytest.mark.parametrize("k", ORDERS)
def test_region_switches(k):
    """Accurate on both sides, and no jump: f(w+) - f(w-) equals f'·(w+ - w-) to 1e-14 of the scale.
    k = 2, 3: |x| = |w| above the axis and |w| = 1/2 below; k <= 1: |x| = 2 (Im w = -ln 2/(2π))."""
    if k >= 2:
        pts = _boundary_points()
    else:
        y = -math.log(2.0) / (2 * math.pi)
        pts = [(complex(a, y * (1 - 1e-12)), complex(a, y * (1 + 1e-12))) for a in (0.0, 0.13, 0.25, 0.5, -0.37)]
    ws = [w for pair in pts for w in pair]
    e, w = max_err(k, ws)
    assert e < TOL[k], (e, w)
    for wm, wp in pts:
        fm, fp = complex(li_exp(k, wm)), complex(li_exp(k, wp))
        df = complex(2j * mp.pi * _mp_li(k - 1, 0.5 * (wm + wp)))      # mpmath: also covers k = -3
        r = ref(k, [wm])
        jump = abs(fp - fm - df * (wp - wm)) / scale(k, [wm], r)[0]
        assert jump < 1e-14, (wm, jump)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_branch_cut_convention(k):
    """On Re w = 0, Im w < 0 (x > 1 real) the value from below the cut, as mpmath gives, i.e. the
    limit Re w -> 0-; from above it differs by 2πi (ln x)^{k-1}/(k-1)!."""
    for y in (1e-6, 0.05, 0.3, 0.7, 2.0):
        w = complex(0.0, -y)
        eps = 1e-13 * y                     # off the cut by a relative 1e-13
        on, below, above = (complex(li_exp(k, v)) for v in (w, complex(-eps, -y), complex(eps, -y)))
        ln_x = 2 * np.pi * y
        jump = 2j * np.pi * ln_x**(k - 1) / math.factorial(k - 1)
        assert abs(on - complex(_mp_li(k, w))) < 1e-13 * max(1.0, abs(on))
        assert abs(on - below) < 1e-9 * max(1.0, abs(on))
        assert abs(above - below - jump) < 1e-9 * max(1.0, abs(jump))


@pytest.mark.parametrize("k", ORDERS)
def test_forward_accuracy_at_half_integral_re_w(k):
    """Re w = ±1/2 (x = -|x|) and next to it, Im w ∈ [1e-15, 1]: plain relative error. Li_{-2} vanishes
    at x = -1; before 2026-10-13, 1 + x was formed by addition and lost all digits as Im w -> 0 (2e-7 at
    1e-10). Next to ±1/2 the shift w ∓ 1/2 must be exact (as 2πi(w ∓ 1/2), not 2πiw ∓ iπ)."""
    res = (0.5, -0.5, 1.5, 0.5 - 1e-9, -0.5 + 1e-12, 0.5 - 1e-5)
    ws = [complex(a, b) for a in res for b in np.logspace(-15, 0, 16)]
    got = np.asarray(li_exp(k, jnp.asarray(ws)))
    r = ref(k, ws)
    e = np.abs(got - r) / np.abs(r)
    assert e.max() < TOL[k], (e.max(), ws[int(np.argmax(e))])


@pytest.mark.parametrize("k", ORDERS)
def test_far_below_the_axis(k):
    """Im w down to -100 (|x| up to e^{628}): k <= 1 used to overflow in powers of x and 1 - x
    (Li_{-3} wrong from Im w ≈ -28, NaN from -38; Li_0 NaN and Li_1 infinite from -114)."""
    ws = [complex(a, -b) for a in (0.13, 0.5, -0.37, 0.0) for b in (29.0, 38.0, 57.0, 100.0)]
    e, w = max_err(k, ws)
    assert e < TOL[k], (e, w)


def test_infinite_imaginary_part():
    """Im w = +inf: x = 0, so Li_k = 0 (a complex product 2πi·w gave 0·inf = NaN)."""
    for k in ORDERS:
        v = complex(li_exp(k, complex(0.3, np.inf)))
        assert v == 0, (k, v)


def test_output_dtype():
    """Integer and float64 input give complex128 (x64 on); float32 gives complex64."""
    for w in (0, np.int64(0), jnp.asarray(0), 0.0, np.float64(0.1), 0.1 + 0.2j):
        assert li_exp(2, w).dtype == jnp.complex128, type(w)
    assert abs(complex(li_exp(2, 0)) - np.pi**2 / 6) < 1e-15
    assert li_exp(2, np.float32(0.1)).dtype == jnp.complex64


def test_zero():
    assert abs(complex(li_exp(2, 0.0)) - np.pi**2 / 6) < 1e-15
    assert abs(complex(li_exp(3, 0j)) - float(mpmath.zeta(3))) < 1e-15
    d = complex(jax.jvp(lambda w: li_exp(3, w), (jnp.asarray(0j),), (jnp.asarray(1 + 0j),))[1])
    assert abs(d - 2j * np.pi * np.pi**2 / 6) < 1e-13


@pytest.mark.parametrize("k", [3, 2, 1, 0, -1, -2])
def test_derivative(k):
    """d/dw Li_k(e^{2πiw}) = 2πi Li_{k-1}(e^{2πiw}) against mpmath, and the second derivative."""
    ws = [0.13 + 0.07j, -0.37 + 0.6j, 0.3 - 0.2j, 1e-6 + 2e-6j, 0.5 + 1e-3j]
    for w in ws:
        w0 = jnp.asarray(complex(w))
        f1 = lambda v: jax.jvp(lambda u: li_exp(k, u), (v,), (jnp.ones_like(v),))[1]
        d1 = complex(f1(w0))
        r1 = complex(2j * mp.pi * _mp_li(k - 1, w))
        assert abs(d1 - r1) < 1e-13 * max(abs(r1), 1e-300), (k, w)
        if k - 1 >= -3 and k >= -1:
            d2 = complex(jax.jvp(f1, (w0,), (jnp.ones_like(w0),))[1])
            r2 = complex((2j * mp.pi)**2 * _mp_li(k - 2, w))
            assert abs(d2 - r2) < 1e-12 * abs(r2), (k, w)


def test_derivative_of_lowest_order_raises():
    with pytest.raises(NotImplementedError):
        jax.jvp(lambda u: li_exp(-3, u), (jnp.asarray(0.1 + 0.2j),), (jnp.asarray(1 + 0j),))


def test_grad_through_real_parameters():
    """jax.grad of a real function of real parameters: d/da Re Li_2(e^{2πi(a + ib)}) = Re(2πi Li_1)."""
    g = jax.grad(lambda a: jnp.real(li_exp(2, a + 0.3j)))(0.17)
    assert abs(float(g) - float(np.real(2j * np.pi * complex(li_exp(1, 0.17 + 0.3j))))) < 1e-14


@pytest.mark.parametrize("k", ORDERS)
def test_jit_vmap_and_shapes(k):
    ws = jnp.asarray([0.13 + 0.07j, -0.37 + 0.6j, 0.3 - 0.2j, 1e-9j, 0.71 + 3j, 0.5 - 2j])
    direct = np.asarray(li_exp(k, ws))
    assert np.array_equal(np.asarray(jax.jit(lambda w: li_exp(k, w))(ws)), direct)
    assert np.array_equal(np.asarray(jax.vmap(lambda w: li_exp(k, w))(ws)), direct)
    # other shapes compile to other code: equal to rounding (measured <= 3.8e-16), not bitwise
    assert np.allclose(np.asarray(li_exp(k, ws.reshape(2, 3))), direct.reshape(2, 3), rtol=1e-15, atol=0)
    assert np.allclose(np.asarray([complex(li_exp(k, w)) for w in ws]), direct, rtol=1e-15, atol=0)


@pytest.mark.parametrize("k", ORDERS)
def test_nan_propagation(k):
    out = np.asarray(li_exp(k, jnp.asarray([np.nan + 0j, 0.1 + 0.2j, complex(0.2, np.nan)])))
    assert np.isnan(out[0]) and np.isnan(out[2]) and np.isfinite(out[1])


@pytest.mark.parametrize("k", ORDERS)
def test_periodicity(k):
    """Folding is exact: shifts of Re w by integers (dyadic Re w, so w + n is exact) change nothing."""
    for w in (0.125 + 0.1j, -0.375 + 0.002j, 0.25 - 0.3j):
        base = complex(li_exp(k, w))
        for n in (-3, 1, 7):
            assert complex(li_exp(k, w + n)) == base


def test_real_input():
    assert complex(li_exp(2, 0.3)) == complex(li_exp(2, 0.3 + 0j))


@pytest.mark.parametrize("k", [4, -4, 2.0, True, "2"])
def test_invalid_order(k):
    with pytest.raises(ValueError):
        li_exp(k, 0.1j)
