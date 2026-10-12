# Changelog

Notable changes to jaxpolylog, newest first.

## Unreleased

### Fixed
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
- `tests/test_real_arguments.py` (5 tests): real arguments with `approx="patch"`, values against
  mpmath and against the complex path, first and second derivatives (2026-10-09; 4 fail before
  the fix).
- `TestClosedFormCases.test_li1_small_modulus` in `tests/test_polylog_accuracy.py`:
  $\mathrm{Li}_1$ against mpmath for `|z|` from 1e-3 to 1e-15, relative tolerance 1e-14.
