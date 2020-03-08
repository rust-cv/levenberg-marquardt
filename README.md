# levenberg-marquardt

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo] ![Tests][btl] ![Lints][bll] ![no_std][bnl]

[ci]: https://img.shields.io/crates/v/levenberg-marquardt.svg
[cl]: https://crates.io/crates/levenberg-marquardt/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/levenberg-marquardt/badge.svg
[dl]: https://docs.rs/levenberg-marquardt/

[lo]: https://tokei.rs/b1/github/rust-cv/levenberg-marquardt?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[btl]: https://github.com/rust-cv/levenberg-marquardt/workflows/unit%20tests/badge.svg
[bll]: https://github.com/rust-cv/levenberg-marquardt/workflows/lints/badge.svg
[bnl]: https://github.com/rust-cv/levenberg-marquardt/workflows/no-std/badge.svg

Provides abstractions to run Levenberg-Marquardt optimization

To add it, install `cargo-edit` (`cargo install cargo-edit`) and run `cargo add levenberg-marquardt`.

Usage (see the [docs](https://docs.rs/levenberg-marquardt/) for more detailed information):

```rust
levenberg_marquardt::optimize(
    // The max number of iterations before terminating
    50,
    // The max number of times it can fail to find a better solution in a row before terminating.
    10,
    // A lambda parameter of `0.0` is Gauss-Newton and a high lambda is gradient descent.
    50.0,
    // If lambda * lambda_converge performs better than the current lambda, that solution is used.
    0.8,
    // If lambda and lambda * lambda_converge fail to find a better solution, lambda is multiplied by this.
    2.0,
    // If the average of residuals squared falls below this value, the algorithm terminates.
    0.0,
    // The initial model.
    initial_model,
    |v| // Normalize your model here if it can become degenerate (rotations, normal vectors).,
    |v| // Compute your residuals here. Multiply these by a constant to scale the speed of convergence.,
    |v| // Compute the Jacobian for each column of the residual matrix here.,
)
```
