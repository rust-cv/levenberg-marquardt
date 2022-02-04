# levenberg-marquardt

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo] ![ci][bci]

[ci]: https://img.shields.io/crates/v/levenberg-marquardt.svg
[cl]: https://crates.io/crates/levenberg-marquardt/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/levenberg-marquardt/badge.svg
[dl]: https://docs.rs/levenberg-marquardt/

[lo]: https://tokei.rs/b1/github/rust-cv/levenberg-marquardt?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[bci]: https://github.com/rust-cv/levenberg-marquardt/workflows/ci/badge.svg

Solver for nonlinear least squares problems

The implementation is a port of the classic MINPACK implementation of the
Levenberg-Marquardt (LM) algorithm. This version of the algorithm is sometimes referred
to as _exact_ LM.

All current unit tests indicate that we achieved _identical_ output (on a floating-point level)
to the MINPACK implementation, especially for rank deficient unstable problems.
This was mainly useful for testing.
The Fortran algorithm was extended with `NaN` and `inf` handling, similar to what [lmfit][lmfit] does.

The crate offers a feature called `minpack-compat` which sets floating-point constants
to the ones used by MINPACK and removes the termination criterion of "zero residuals".
This is necessary for identical output to MINPACK but generally not recommended.

# Usage

See the [docs](https://docs.rs/levenberg-marquardt/) for detailed information.

```rust
impl LeastSquaresProblem<f64> for Problem {
    // define this trait for the problem you want to solve
}
let problem = Problem::new(initial_params);
let (problem, report) = LevenbergMarquardt::new().minimize(problem);
assert!(report.termination.was_successful());
```

# References

Sofware:

- The [MINPACK](https://www.netlib.org/minpack/) Fortran implementation.
- A C version/update, [lmfit][lmfit].
- A Python implementation in [pwkit](https://github.com/pkgw/pwkit/blob/master/pwkit/lmmin.py).

One original reference for the algorithm seems to be

> Moré J.J. (1978) The Levenberg-Marquardt algorithm: Implementation and theory. In: Watson G.A. (eds) Numerical Analysis. Lecture Notes in Mathematics, vol 630. Springer, Berlin, Heidelberg.

by one of the authors of MINPACK.

The algorihm is also described in the form as
implemented by this crate in the [book "Numerical Optimization"](https://link.springer.com/book/10.1007%2F978-0-387-40065-5) by Nocedal and Wright, chapters 4 and 10.

[lmfit]: https://github.com/pkgw/pwkit/blob/master/pwkit/lmmin.py