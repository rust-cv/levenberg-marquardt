//! Implementation of the [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
//! optimization algorithm using [nalgebra](https://nalgebra.org).
//!
//! This algorithm tries to solve the least-squares optimization problem
//! ```math
//! \min_{\vec{x}\in\R^n}f(\vec{x})\quad\text{where}\quad\begin{cases}\begin{aligned}
//!   \ f\!:\R^n &\to \R \\
//!  \vec{x} &\mapsto \frac{1}{2}\sum_{i=1}^m \bigl(r_i(\vec{x})\bigr)^2,
//! \end{aligned}\end{cases}
//! ```
//! for differentiable _residual functions_ `$r_i\!:\R^n\to\R$`.
//!
//! # Inputs
//!
//! The problem has `$n$` parameters `$\vec{x}\in\R^n$` and `$m\geq n$` residual
//! functions `$r_i\!:\R^n\to\R$`.
//!
//! You must provide an implementation of
//!
//! - the residual vector `$\vec{x} \mapsto (r_1(\vec{x}), \ldots, r_m(\vec{x}))^\top\in\R^m$`
//! - and its Jacobian `$\mathbf{J} \in \R^{m\times n}$`, defined as
//!   ```math
//!   \mathbf{J} \coloneqq
//!   \begin{pmatrix}
//!   \frac{\partial r_1}{\partial x_1} & \cdots & \frac{\partial r_1}{\partial x_n} \\
//!   \vdots & \ddots & \vdots \\
//!   \frac{\partial r_m}{\partial x_1} & \cdots & \frac{\partial r_m}{\partial x_n}
//!   \end{pmatrix}.
//!   ```
//!
//! Finally, you have to provide an initial guess for `$\vec{x}$`. This can
//! be a constant value, but typically the optimization result _crucially_ depends
//! on a good initial value.
//!
//! The algorithm also has a number of hyperparameters which are documented
//! at [`LevenbergMarquardt`](struct.LevenbergMarquardt.html) along with
//! implementation details.
//!
//! # Usage Example
//!
//! We use `$f(x, y) \coloneqq \frac{1}{2}[(x^2 + y - 11)^2 + (x + y^2 - 7)^2]$` as a [test function](https://en.wikipedia.org/wiki/Himmelblau%27s_function)
//! for this example.
//! In this case we have `$n = 2$` and `$m = 2$` with
//!
//! ```math
//!   r_1(\vec{x}) \coloneqq x_1^2 + x_2 - 11\quad\text{and}\quad
//!   r_2(\vec{x}) \coloneqq x_1 + x_2^2 - 7.
//! ```
//!
//! ```
//! # use nalgebra::*;
//! # use nalgebra::storage::Owned;
//! # use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
//! struct ExampleProblem {
//!     // holds current value of the n parameters
//!     p: Vector2<f32>,
//! }
//!
//! // We must implement a trait for every problem we want to solve
//! impl LeastSquaresProblem<f32, U2, U2> for ExampleProblem {
//!     type ParameterStorage = Owned<f32, U2>;
//!     type ResidualStorage = Owned<f32, U2>;
//!     type JacobianStorage = Owned<f32, U2, U2>;
//!     
//!     fn set_params(&mut self, p: &VectorN<f32, U2>) {
//!         self.p.copy_from(p);
//!         // do common calculations for residuals and the Jacobian here
//!     }
//!     
//!     fn residuals(&self) -> Option<Vector2<f32>> {
//!         Some(Vector2::new(
//!             self.p.x * self.p.x + self.p.y - 11.0,
//!             self.p.x + self.p.y * self.p.y - 7.0,
//!         ))
//!     }
//!     
//!     fn jacobian(&self) -> Option<Matrix2<f32>> {
//!         Some(Matrix2::new(
//!             2.0 * self.p.x, 1.0,
//!             1.0, 2.0 * self.p.y,
//!         ))
//!     }
//! }
//!
//! let problem = ExampleProblem {
//!     p: Vector2::zeros(),
//! };
//! let (_result, report) = LevenbergMarquardt::new()
//!     .minimize(Vector2::new(1., 1.), problem);
//! assert!(report.failure.is_none());
//! assert!(report.objective_function.abs() < 1e-10);
//! ```
//!
//! # Derivative checking
//!
//! **TODO** [`differentiate_numerically`](fn.differentiate_numerically.html).
//!
//! # Vector-valued residuals `$\vec{r}_i$`
//!
//! For some problems it is natural to group the residuals into
//! vector-valued residuals `$\vec{r}_i$`. **TODO**
#![no_std]

mod lm;
mod problem;
mod qr;
mod trust_region;
mod utils;

pub use lm::{Failure, LevenbergMarquardt, MinimizationReport};
pub use problem::LeastSquaresProblem;

pub use utils::differentiate_numerically;
