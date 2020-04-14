# levenberg-marquardt

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![Tests][btl] ![Lints][bll] ![no_std][bnl]

[ci]: https://img.shields.io/crates/v/levenberg-marquardt.svg
[cl]: https://crates.io/crates/levenberg-marquardt/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/levenberg-marquardt/badge.svg
[dl]: https://docs.rs/levenberg-marquardt/

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[btl]: https://github.com/rust-cv/levenberg-marquardt/workflows/unit%20tests/badge.svg
[bll]: https://github.com/rust-cv/levenberg-marquardt/workflows/lints/badge.svg
[bnl]: https://github.com/rust-cv/levenberg-marquardt/workflows/no-std/badge.svg

Solver for non-linear least-squares problems.

The implementation is a port of the classic MINPACK implementation of the
Levenberg-Marquardt (LM) algorithm. This versions is sometimes referred to as _exact_ LM.

# Usage

See the [docs](https://docs.rs/levenberg-marquardt/) for detailed information.

```rust
impl LeastSquaresProblem<f64> for Problem {
    // define this trait for the problem you want to solve
}
let problem = Problem::new();
let (problem, report) = LevenbergMarquardt::new().minimize(initial_params, problem);
assert!(report.termination.was_succes());
```

# References

Sofware:

- The [MINPACK](https://www.netlib.org/minpack/) Fortran implementation.
- A C version/update, [lmfit](https://jugit.fz-juelich.de/mlz/lmfit).

One original reference for the algorithm seems to be

> Moré J.J. (1978) The Levenberg-Marquardt algorithm: Implementation and theory. In: Watson G.A. (eds) Numerical Analysis. Lecture Notes in Mathematics, vol 630. Springer, Berlin, Heidelberg.

by one of the authors of MINPACK.

The algorihm is also described in the form as
implemented by this crate in the [book "Numerical Optimization"](https://link.springer.com/book/10.1007%2F978-0-387-40065-5) by Nocedal and Wright, chapters 4 and 10.