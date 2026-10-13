# Copyright 2022-2026 Andreas Schachner
#
# This file is part of jaxpolylog.
#
# jaxpolylog is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# jaxpolylog is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with jaxpolylog. If not, see <https://www.gnu.org/licenses/>.

r"""
Polylogarithms of an exponential argument, :math:`\mathrm{Li}_k(e^{2\pi i w})` (added 2026-10-12).

Instanton sums evaluate :math:`\mathrm{Li}_k(e^{2\pi i\, q\cdot z})` with the exponent known to full
precision. Passing :math:`x = e^{2\pi i w}` to :func:`~jaxpolylog.polylogs.jax_polylog` throws part of
it away: near :math:`x = 1` (a vanishing curve, :math:`w \to 0`) the low digits of :math:`w` are lost
when :math:`x` is formed, and :math:`\mathrm{Li}_{k\le1}` is singular there. :func:`li_exp` takes
:math:`w` itself.
"""

from functools import partial
import math

import numpy as np
import jax
import jax.numpy as jnp

from .polylogs import _zero_branch_coeffs

__all__ = ["li_exp"]

# Terms of both series for k = 2, 3. Error analysis (see :func:`li_exp`): the power series is used
# where |x| < |w|, the expansion in mu elsewhere above the axis and for |w| <= 1/2 below it, so both
# converge at least like 0.511^n; 64 terms leave tails below 1e-21 (measured 2026-10-13).
_N_TERMS = 64
_ORDERS = range(-3, 4)
# Expansion around mu = 0: sum_n zeta(k-n)/n! mu^n (the n = k-1 term is the explicit logarithm),
# highest power first for jnp.polyval.
_NEAR_COEFFS = {k: np.array(_zero_branch_coeffs(k, _N_TERMS)[::-1]) for k in (2, 3)}
# Power series x * sum_{n=1}^{N} x^(n-1)/n^k, highest power first.
_FAR_COEFFS = {k: np.array([1.0 / n**k for n in range(_N_TERMS, 0, -1)]) for k in (2, 3)}
_HARMONIC = {2: 1.0, 3: 1.5}
# Below the real axis, points with |w| above this use the inversion relation (k = 2, 3).
_INVERSION_RADIUS = 0.5
# Below the real axis, k <= 1 is evaluated in y = 1/x where |x| > 2, i.e. Im w < -ln(2)/(2π):
# the powers of x and 1 - x overflow long before x itself (Li_{-3} from Im w < -28).
_IM_W_INVERT_CLOSED = -math.log(2.0) / (2 * math.pi)


def _check_order(k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k not in _ORDERS:
        raise ValueError(f"li_exp: the order k must be an integer in [-3, 3], got {k!r}.")


def _mu(w):
    r"""μ = 2πiw, formed from Re w and Im w (a complex product would give 0·∞ = NaN for infinite w)."""
    return jax.lax.complex(-2 * jnp.pi * w.imag, 2 * jnp.pi * w.real)


def _log_minus_x(w):
    r"""
    The principal ln(-x), x = e^{2πiw}, for folded w: 2πi(w + 1/2) for Re w <= 0 and 2πi(w - 1/2) for
    Re w > 0. Its imaginary part lies in [-π, π] (π on the cut Re w = 0, the value from below).
    Near x = -1 (Re w = ±1/2) the shift by 1/2 is exact.
    """
    return _mu(jnp.where(w.real > 0, w - 0.5, w + 0.5))


def _on_cut(w):
    r"""Points on the branch cut :math:`x > 1` (Re w = 0, Im w < 0) after folding."""
    return (w.real == 0) & (w.imag < 0)


def _log_cut_below(z, cut):
    r"""Principal log, except on ``cut``, where the value from below the cut (+iπ) is taken."""
    return jnp.where(cut, jnp.log(jnp.abs(z)) + 1j * jnp.pi, jnp.log(z))


def _closed_direct(k, w):
    r"""Li_k(x) for k <= 1 and |x| <= 2, with 1 - x = -expm1(μ) and, for k = -2, 1 + x = -expm1(ln(-x))."""
    mu = _mu(w)
    x = jnp.exp(mu)
    one_minus_x = -jnp.expm1(mu)
    if k == 1:
        small = jnp.abs(x) <= 0.5
        via_log1p = -jnp.log1p(-jnp.where(small, x, 0.0))
        via_log = -_log_cut_below(one_minus_x, _on_cut(w))
        return jnp.where(small, via_log1p, via_log)
    if k == 0:
        return x / one_minus_x
    if k == -1:
        return x / one_minus_x**2
    if k == -2:
        one_plus_x = -jnp.expm1(_log_minus_x(w))          # accurate at the zero x = -1
        return x * one_plus_x / one_minus_x**3
    return x * (1 + 4 * x + x**2) / one_minus_x**4          # k == -3


def _closed_inverted(k, w):
    r"""
    Li_k(x) for k <= 1 and |x| > 2, in y = 1/x = e^{-μ}: Li_1(x) = -ln(-x) - log1p(-y),
    Li_0(x) = -1 - Li_0(y), Li_{-n}(x) = (-1)^{n+1} Li_{-n}(y) for n >= 1.
    """
    mu = _mu(w)
    y = jnp.exp(-mu)
    one_minus_y = -jnp.expm1(-mu)
    if k == 1:
        return -_log_minus_x(w) - jnp.log1p(-y)
    if k == 0:
        return -1 - y / one_minus_y
    if k == -1:
        return y / one_minus_y**2
    if k == -2:
        return -y * (1 + y) / one_minus_y**3
    return y * (1 + 4 * y + y**2) / one_minus_y**4          # k == -3


def _closed_form(k, w):
    r"""Li_k(e^{2πiw}) for k <= 1 (folded w)."""
    invert = w.imag < _IM_W_INVERT_CLOSED
    v_inv = _closed_inverted(k, jnp.where(invert, w, -0.5j))   # safe points lie in the other region
    v_dir = _closed_direct(k, jnp.where(invert, 0.25j, w))
    return jnp.where(invert, v_inv, v_dir)


def _near(k, w):
    r"""Expansion in μ = 2πiw (convergent for |μ| < 2π); exact value ζ(k) at w = 0."""
    mu = _mu(w)
    zero = mu == 0
    mu_safe = jnp.where(zero, 1.0, mu)
    log_term = mu_safe**(k - 1) / math.factorial(k - 1) * (_HARMONIC[k] - _log_cut_below(-mu_safe, _on_cut(w)))
    return jnp.where(zero, 0.0, log_term) + jnp.polyval(jnp.asarray(_NEAR_COEFFS[k], dtype=mu.dtype), mu)


def _far(k, w):
    r"""Power series in x = e^{2πiw}, used where |x| < |w| (so |x| <= ~0.511; |w| is unbounded)."""
    x = jnp.exp(_mu(w))
    return x * jnp.polyval(jnp.asarray(_FAR_COEFFS[k], dtype=x.dtype), x)


def _upper(k, w):
    r"""k = 2, 3 for folded w with Im w >= 0, or below the axis with |w| <= 1/2."""
    t = jnp.abs(w)
    use_far = jnp.exp(-2 * jnp.pi * w.imag) < t
    v_far = _far(k, jnp.where(use_far, w, 0.25j))       # safe points lie in the other region
    v_near = _near(k, jnp.where(use_far, 0.1j, w))
    return jnp.where(use_far, v_far, v_near)


def _inverted(k, w):
    r"""
    Li_k(x) = -(-1)^k Li_k(1/x) - (2πi)^k/k! B_k(1/2 + ln(-x)/(2πi)), x = e^{2πiw}, x ∉ (0, 1].

    With the principal ln(-x) of :func:`_log_minus_x`, the Bernoulli argument is w + 1 for Re w <= 0
    and w for Re w > 0. On the cut (Re w = 0) this is the value from below, as mpmath gives.
    """
    u = jnp.where(w.real <= 0, w + 1, w)
    bernoulli = u * u - u + 1.0 / 6 if k == 2 else u**3 - 1.5 * u * u + 0.5 * u
    return (-(-1)**k * _upper(k, -w)
            - (2j * jnp.pi)**k / math.factorial(k) * bernoulli)


@partial(jax.jit, static_argnums=0)
def _li_exp(k, w):
    w = jnp.asarray(w)
    w = w.astype(jnp.result_type(w, 1j))                # int -> complex128 under x64; float32 -> complex64
    w = w - jnp.round(w.real)                           # x is periodic in Re w; exact
    if k <= 1:
        return _closed_form(k, w)
    use_inv = (w.imag < 0) & (jnp.abs(w) > _INVERSION_RADIUS)
    v_inv = _inverted(k, jnp.where(use_inv, w, -0.75j))
    v_up = _upper(k, jnp.where(use_inv, 0.1j, w))
    return jnp.where(use_inv, v_inv, v_up)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def li_exp(k: int, w):
    r"""
    **Description:**
    The polylogarithm of an exponential argument, :math:`\mathrm{Li}_k(e^{2\pi i w})`, for integer
    :math:`-3 \le k \le 3`. Elementwise, jit- and vmap-compatible, differentiable in :math:`w`.

    .. admonition:: Details
        :class: dropdown

        :math:`w` is first folded, :math:`w \to w - \mathrm{round}(\mathrm{Re}\,w)` (exact, since
        :math:`e^{2\pi i w}` has period 1). With :math:`\mu = 2\pi i w` and :math:`x = e^{\mu}`:

        - :math:`k \le 1`, :math:`|x| \le 2`: closed forms in :math:`x`, with
          :math:`1 - x = -\mathrm{expm1}(\mu)`, :math:`\mathrm{Li}_1 = -\mathrm{log1p}(-x)` for
          :math:`|x| \le 1/2`, and for :math:`\mathrm{Li}_{-2}`, whose zero :math:`x = -1` lies at
          :math:`\mathrm{Re}\,w = \pm 1/2`, :math:`1 + x = -\mathrm{expm1}(\ln(-x))` with the exact
          :math:`\ln(-x) = 2\pi i (w \mp 1/2)`;
        - :math:`k \le 1`, :math:`|x| > 2` (:math:`\mathrm{Im}\,w < -\ln 2/(2\pi)`): the same closed
          forms in :math:`y = 1/x = e^{-\mu}`, :math:`\mathrm{Li}_1(x) = -\ln(-x) - \mathrm{log1p}(-y)`,
          :math:`\mathrm{Li}_0(x) = -1 - \mathrm{Li}_0(y)` and
          :math:`\mathrm{Li}_{-n}(x) = (-1)^{n+1}\mathrm{Li}_{-n}(y)`, so that no power of :math:`x`
          overflows;
        - :math:`k = 2, 3`, the expansion around :math:`\mu = 0`,

          .. math::
              \mathrm{Li}_k(e^\mu) = \frac{\mu^{k-1}}{(k-1)!}\left(H_{k-1} - \log(-\mu)\right)
              + \sum_{n \ne k-1} \zeta(k-n)\, \frac{\mu^n}{n!} \,,

          convergent for :math:`|\mu| < 2\pi`, its error after :math:`N` terms roughly
          :math:`t^N`, :math:`t = |\mu|/(2\pi) = |w|`; or the power series
          :math:`\sum_{n \ge 1} x^n/n^k`, roughly :math:`|x|^N`. The power series is used where
          :math:`|x| < t`, the smaller of these crude estimates. For folded :math:`w` with
          :math:`\mathrm{Im}\,w \ge 0`, or :math:`|w| \le 1/2`, the ratio of the series used is then
          at most about 0.511; 64 terms are used.
        - :math:`k = 2, 3` with :math:`\mathrm{Im}\,w < 0` and :math:`|w| > 1/2` (:math:`|x| > 1`
          outside the expansion's useful range): the inversion relation

          .. math::
              \mathrm{Li}_k(x) = -(-1)^k\, \mathrm{Li}_k(1/x)
              - \frac{(2\pi i)^k}{k!}\, B_k\!\left(\frac12 + \frac{\ln(-x)}{2\pi i}\right) ,
              \qquad x \notin (0, 1] ,

          with :math:`1/x = e^{-2\pi i w}` and the Bernoulli polynomial :math:`B_k`.
        - :math:`w = 0`: :math:`\zeta(2)`, :math:`\zeta(3)` for :math:`k = 2, 3` (the vanishing term
          :math:`\mu^{k-1}\log(-\mu)` is removed by a double ``where``; so :math:`\mathrm{Li}_2(1)` and
          the derivative of :math:`\mathrm{Li}_3` there are finite, while the derivative of
          :math:`\mathrm{Li}_2` diverges). For :math:`k \le 1`, :math:`w = 0` is a pole; the phase of
          the infinite value returned there is meaningless.

        **Branch cut.** For :math:`k \ge 1`, :math:`x > 1` (:math:`\mathrm{Re}\,w = 0`,
        :math:`\mathrm{Im}\,w < 0`) is the cut. There the value from below the cut is returned, the
        limit :math:`\mathrm{Re}\,w \to 0^-`, as mpmath's ``polylog`` does for real :math:`x > 1`.

        **Derivatives** (``custom_jvp``):
        :math:`\frac{d}{dw}\mathrm{Li}_k(e^{2\pi i w}) = 2\pi i\,\mathrm{Li}_{k-1}(e^{2\pi i w})`,
        evaluated by :func:`li_exp` itself; nested derivatives are available down to order
        :math:`k = -3`, and differentiating :math:`\mathrm{Li}_{-3}` raises ``NotImplementedError``.

        **Precision.** The result is complex128 for float64, complex128 or integer input and
        complex64 for float32 or complex64 input (JAX type promotion). Importing jaxpolylog enables
        ``jax_enable_x64``; if it is disabled afterwards, everything is complex64. Values smaller than
        the smallest normal float underflow to 0 (e.g. :math:`\mathrm{Li}_{-1}` for
        :math:`\mathrm{Im}\,w \lesssim -113`). Measured against mpmath, see the CHANGELOG.

    Args:
        k (int): The order, an integer in [-3, 3]. Static.
        w (complex or array): The exponent: the argument is :math:`e^{2\pi i w}`. Real input is
            treated as complex.

    Returns:
        complex or array: :math:`\mathrm{Li}_k(e^{2\pi i w})`, with the shape of ``w``.

    Raises:
        ValueError: If ``k`` is not an integer in [-3, 3].

    Example:
        ```python
        import jax
        jax.config.update("jax_enable_x64", True)
        from jaxpolylog import li_exp
        li_exp(2, 0.0)          # zeta(2) = pi^2/6
        li_exp(3, 1e-12 + 0.3j)
        ```
    """
    _check_order(k)
    return _li_exp(k, w)


@li_exp.defjvp
def _li_exp_jvp(k, primals, tangents):
    (w,), (dw,) = primals, tangents
    if k <= -3:
        raise NotImplementedError("li_exp: derivatives of Li_{-3} (order k = -4) are not implemented.")
    return li_exp(k, w), 2j * jnp.pi * li_exp(k - 1, w) * dw
