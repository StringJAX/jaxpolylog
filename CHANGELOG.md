# Changelog

Notable changes to jaxpolylog, newest first.

## [Unreleased] — 0.4.1 (prepared 2026-10-28; not yet tagged)

Packaging, CI and documentation only; no change to any computed value.

### Fixed
- The API pages of `jax_polylog` and `li_exp`. autodoc classifies a `jax.custom_jvp` object as
  data, not as a function, so both pages showed `jax_polylog = <jax._src.custom_derivatives.custom_jvp
  object>` with no signature, and napoleon read the docstrings' first line, `**Description:**`, as
  an attribute's `type: description` (a stray `**` and a `:type: **Description` field). The stub pages
  now use `.. autofunction::` (template `_templates/autosummary/custom_jvp.rst`) and show the
  signatures; a hook in `conf.py` parses these docstrings as a function's in the summary tables.
- The `|z| < 1` in `jax_polylog`'s docstring is now maths; docutils had read `|z|` as an undefined
  substitution.
- `conf.py` no longer suppresses the `autodoc` and `docutils` warning classes; they had hidden the
  defects above (four warnings and one error).
- The quickstart notebook's cells carry ids (nbformat 4.5 requires them; a future nbformat makes
  their absence an error).
- `documentation/source/_static/` is tracked (`.gitkeep`). `html_static_path` names it, and an
  empty directory does not survive a clone, so a clean checkout warned.

### Changed
- Licence classifier: "GNU General Public License v3 or later (GPLv3+)", matching the
  `GPL-3.0-or-later` licence field and the source headers. Python 3.13 classifier added.
- Read the Docs: `fail_on_warning: true`.

### Added
- CI (`.github/workflows/ci.yml`): the test suite on Python 3.12 and 3.13, and docs builds (html and
  singlehtml, the builders Read the Docs runs with `-W`) with `-W`, on every push and pull request to `main`
  and weekly.

## 0.4.0 — 2026-10-15

### Fixed
- `jaxpolylog.__version__` matches the package version. The 0.3.0 upload on PyPI reports
  `__version__ == '0.2.1'` while its metadata says 0.3.0: `setup.py` had been bumped and
  `jaxpolylog/__init__.py` had not. `tests/test_version.py` now checks that the two agree.
- Branch choice of `approx="patch"` off the positive real axis (2026-10-12). For s ≥ 2 the method
  uses the power series Σ z^k/k^s (truncation error ~ |z|^N) or the expansion in μ = log z (error
  ~ t^N, t = |μ|/(2π)). It took the power series whenever |z| < 1 and t ≥ `pval`; the default
  `pval` ≈ 0.2322 solves |z| = t on the positive real axis only. Off that axis t ≥ `pval` says
  nothing about |z|: at z = -|z|, t ≥ 1/2 for every |z|, so the power series was used up to
  z → -1, where N terms leave an error of about 1/(2N^s). Measured on the instanton arguments of the
  dS examples of arXiv:2406.13751 (z from -0.974 to -0.998): Li_2 off by up to 5.4e-3 with
  `p_range` = 10 and 4.3e-5 with 100. Off the positive real axis the power series now also requires
  |z| ≤ max(t, e^{-2π pval}) (helper `_patch_use_inf`, used by the values and by the derivative
  rule `_Li_over_z`); with the default `pval` this is |z| ≤ t, the branch with the smaller of two
  crude error estimates (|z|^N, t^N; prefactors ignored). It is not the better branch at every point
  (for s = 3, `p_range` = 10, up to 150 times worse near z = -0.52), but the largest error over the
  unit disk falls at every `p_range` (s = 2, `p_range` = 10: from 7.3e-3 to 8.5e-5). On the positive
  real axis the rule is exactly t ≥ `pval`, so results there are bit-for-bit unchanged for every
  `pval` (2026-10-13: by construction; the first version agreed only in exact arithmetic, and for
  `pval` < 0.2322 rounding could flip the branch, e.g. `pval` = 0.14 at z = 0.41492979447371536).
  Off the axis, results change wherever the old rule chose the power series with |z| > t. kahlerjax
  evaluates z < 0 for every curve with e^{2πi q·b} = -1; on the dS examples, V moved by up to 1.4e-2
  and now reproduces the published values to ≤ 2.3e-10 (with ≥ 50 terms). jaxvacua's fast tests of
  the files `test_periods`, `test_css`, `test_coniLCS_limit`, `test_conifold_basis`,
  `test_conifold_bulk_eft`, `test_lcs` and `test_critical_points` (`"patch"` is called in
  `periods.py`, `css.py` and `conifold/`) pass unchanged: 149 passed, 2 skipped.
- Real (float64) arguments with `approx="patch"` (2026-10-09). The safe substitute values of the
  double-where dispatch were complex literals cast to `z.dtype`, which raises `TypeError` for real
  `z` in current JAX (measured with jax 0.10.0): `Li_3` and `Li_2` and their derivatives failed for
  any real input. They are now real literals, which cast to real or complex `z` alike; results for
  complex `z` are unchanged. kahlerjax calls `Li_3` and `Li_2` this way, with real arguments
  whenever the worldsheet-instanton prefactors are real.
- `jax_polylog(z, 1, ...)` now evaluates $\mathrm{Li}_1(z)$ as `-log1p(-z)` instead of `-log(1-z)`
  (2026-10-08). Forming `1-z` lost the low digits of a small `z`, so the relative error was about
  machine epsilon divided by `|z|`: 1e-6 at `|z| = 1e-10`. Worldsheet-instanton sums evaluate
  $\mathrm{Li}_1$ at such arguments, with large integer prefactors.

### Added
- `li_exp(k, w)` = Li_k(e^{2πiw}) for integer -3 ≤ k ≤ 3 (new module `jaxpolylog/exponent_argument.py`,
  exported at top level; 2026-10-12, review fixes 2026-10-13). It takes the exponent w, so nothing is
  lost forming x = e^{2πiw} near x = 1. k ≤ 1 by closed forms with `expm1`/`log1p`, in x for |x| ≤ 2
  and in 1/x for |x| > 2; k = 2, 3 by the expansion in μ = 2πiw or the power series in x, whichever has
  the smaller crude error estimate (64 terms), and the inversion relation for Im w < 0, |w| > 1/2. On
  the cut x > 1 (Re w = 0, Im w < 0) it returns the value from below, as mpmath does. `custom_jvp`
  with d/dw = 2πi Li_{k-1}, nested down to order -3. Integer and float64 input give complex128.
  Measured against mpmath (`worklog/kklt_jax/measurements/li_exp/check_li_exp.log`; grid maxima, not
  bounds) over Re w ∈ [-1/2, 1/2] × Im w ∈ ±[1e-15, 50], |w| ≥ 1e-15, and the line Re w = ±1/2:
  relative to max(|Li_k|, |w Li_k'|) (backward error) ≤ 1.5e-15 for k = 2, 3 and ≤ 5.9e-15 for
  k ≤ 1; plain relative error ≤ 1.0e-14 (the largest values at |Im w| ≈ 50, where the conditioning of
  e^{2πiw} reaches that level), and on Re w = ±1/2 ≤ 5.5e-16 for k ≤ 1. On the upper-half-plane
  points `patch` is off by 4.1e-5 (k = 2) and 6.6e-5 (k = 3) at p = 10, 4.0e-15 and 5.6e-15 at
  p = 100. Time for 10^5 points: 3.1–4.2 ms against 4.6–5.5 ms (`patch`, p = 10) and 46–106 ms
  (p = 100) over two runs; at 609 points the two are within timing noise. Docs: API page and stubs.
  - Found in the review of 2026-10-13 and fixed before release: for k ≤ 1 and Im w < 0 the powers of
    x and 1 - x overflowed (Li_{-3} wrong from Im w ≈ -28, NaN from -38; Li_0 NaN and Li_1 infinite
    from -114), now evaluated in 1/x; Li_{-2} lost all digits near its zero x = -1 (relative error
    2e-7 at w = 1/2 + 1e-10 i), now 1 + x = -expm1(ln(-x)) with the exact ln(-x) = 2πi(w ∓ 1/2);
    integer input gave complex64; w with Im w = +inf gave NaN (now 0); several docstring statements.
- `tests/test_li_exp.py` (83 tests): the grids above (the lower half plane now to Im w = -50, as
  stated; until 2026-10-13 to -5), forward accuracy on Re w = ±1/2 and next to it, Im w down to -100,
  the region switches of k = 2, 3 and of k ≤ 1 (accurate on both sides, no jump beyond 1e-14), the
  cut convention, w = 0, Im w = +inf, output dtypes, derivatives against mpmath, jit/vmap/shapes,
  NaN, periodicity, invalid orders. The grids use the backward-error measure above, which allows for
  the conditioning of e^{2πiw}; the Re w = ±1/2 test uses plain relative error. Mutations caught
  (`worklog/kklt_jax/measurements/review_2026-10-13/`): the Bernoulli argument, the old branch rule,
  the cut convention, 20 terms, Li_1 without `log1p`, the JVP factor, no folding, no w = 0 guard, no
  inversion for k ≤ 1, 1 + x by addition, ln(-x) as 2πiw ∓ iπ, the old dtype rule, μ by a complex
  product.
- `tests/test_patch_branch_choice.py` (13 tests, 2026-10-12/13): `"patch"` values of Li_2, Li_3, Li_4
  and the derivatives of Li_3, Li_4 against mpmath at seven points inside the unit disk off the
  positive real axis (z = -0.6 … -0.996, 0.95i, two others) where the expansion in log z must be
  used, and at six where the power series must be (z = -0.05, -0.4, 0.3i, 1e-3 e^{i}, 1e-3 e^{2.5i},
  0.2 e^{2i}); on the positive real axis, bit-for-bit the branch the old rule chose, for `pval` 0.10,
  0.14, the default and 0.40, including the rounding case above; off the axis, the branch for a
  `pval` below the default (z = 0.6 e^{0.5i}, `pval` = 0.05). The five off-axis tests fail before the
  fix; the positive-axis test with `pval` = 0.14 fails on the first version of the fix; replacing
  max(t, e^{-2π pval}) by t, dropping t ≥ `pval`, or using the expansion everywhere off the axis
  fails at least one test (2026-10-13).
- `tests/test_real_arguments.py` (5 tests): real arguments with `approx="patch"`, values against
  mpmath and against the complex path, first and second derivatives (2026-10-09; 4 fail before
  the fix).
- `tests/test_version.py`: `setup.py` and `jaxpolylog.__version__` give the same version.
- `TestClosedFormCases.test_li1_small_modulus` in `tests/test_polylog_accuracy.py`:
  $\mathrm{Li}_1$ against mpmath for `|z|` from 1e-3 to 1e-15, relative tolerance 1e-14.
