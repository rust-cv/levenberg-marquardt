//! Implementation of the [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
//! optimization algorithm using [nalgebra](https://nalgebra.org).
//!
//! This algorithm tries to solve the least squares optimization problem
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
//! The problem has `$n$` parameters `$\vec{x}\in\R^n$` and `$m$` residual
//! functions `$r_i\!:\R^n\to\R$`.
//!
//! You must provide an implementation of
//!
//! - the residual vector `$\vec{x} \mapsto (r_1(\vec{x}), \ldots, r_m(\vec{x}))^\top\in\R^m$`
//! - and its Jacobian `$\mathbf{J} \in \R^{m\times n}$`, defined as
//!   ```math
//!   \mathbf{J} \coloneqq
//!   \def\arraystretch{1.5}
//!   \begin{pmatrix}
//!   \frac{\partial r_1}{\partial x_1} & \cdots & \frac{\partial r_1}{\partial x_n} \\
//!   \frac{\partial r_2}{\partial x_1} & \cdots & \frac{\partial r_2}{\partial x_n} \\
//!   \vdots & \ddots & \vdots \\
//!   \frac{\partial r_m}{\partial x_1} & \cdots & \frac{\partial r_m}{\partial x_n}
//!   \end{pmatrix}.
//!   ```
//!
//! Finally, you have to provide an initial guess for `$\vec{x}$`. This can
//! be a constant value, but typically the optimization result crucially depends
//! on a good initial value.
//!
//! The algorithm also has a number of hyperparameters which are documented
//! at [`LevenbergMarquardt`](struct.LevenbergMarquardt.html).
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
//!     p: Vector2<f64>,
//! }
//!
//! // We implement a trait for every problem we want to solve
//! impl LeastSquaresProblem<f64, U2, U2> for ExampleProblem {
//!     type ParameterStorage = Owned<f64, U2>;
//!     type ResidualStorage = Owned<f64, U2>;
//!     type JacobianStorage = Owned<f64, U2, U2>;
//!     
//!     fn set_params(&mut self, p: &VectorN<f64, U2>) {
//!         self.p.copy_from(p);
//!         // do common calculations for residuals and the Jacobian here
//!     }
//!     
//!     fn params(&self) -> VectorN<f64, U2> { self.p }
//!     
//!     fn residuals(&self) -> Option<Vector2<f64>> {
//!         let [x, y] = [self.p.x, self.p.y];
//!         // vector containing residuals $r_1(\vec{x})$ and $r_2(\vec{x})$
//!         Some(Vector2::new(x*x + y - 11., x + y*y - 7.))
//!     }
//!     
//!     fn jacobian(&self) -> Option<Matrix2<f64>> {
//!         let [x, y] = [self.p.x, self.p.y];
//!         
//!         // first row of Jacobian, derivatives of first residual
//!         let d1_x = 2. * x; // $\frac{\partial}{\partial x_1}r_1(\vec{x}) = \frac{\partial}{\partial x} (x^2 + y - 11) = 2x$
//!         let d1_y = 1.;     // $\frac{\partial}{\partial x_2}r_1(\vec{x}) = \frac{\partial}{\partial y} (x^2 + y - 11) = 1$
//!         
//!         // second row of Jacobian, derivatives of second residual
//!         let d2_x = 1.;     // $\frac{\partial}{\partial x_1}r_2(\vec{x}) = \frac{\partial}{\partial x} (x + y^2 - 7) = 1$
//!         let d2_y = 2. * y; // $\frac{\partial}{\partial x_2}r_2(\vec{x}) = \frac{\partial}{\partial y} (x + y^2 - 7) = 2y$
//!
//!         Some(Matrix2::new(
//!             d1_x, d1_y,
//!             d2_x, d2_y,
//!         ))
//!     }
//! }
//!
//! let problem = ExampleProblem {
//!     // the initial guess for $\vec{x}$
//!     p: Vector2::new(1., 1.),
//! };
//! let (_result, report) = LevenbergMarquardt::new().minimize(problem);
//! assert!(report.termination.was_successful());
//! assert!(report.objective_function.abs() < 1e-10);
//! ```
//!
//! # Derivative checking
//!
//! You should try using [`differentiate_numerically`](fn.differentiate_numerically.html)
//! in a unit test to verify that your Jacobian implementation matches the residuals.
#![no_std]
#![cfg_attr(feature = "RUSTC_IS_NIGHTLY", core_intrinsics)]

extern crate alloc;

mod lm;
mod problem;
mod qr;
mod trust_region;
pub(crate) mod utils;

pub use lm::TerminationReason;
pub use problem::LeastSquaresProblem;

pub use utils::{differentiate_holomorphic_numerically, differentiate_numerically};

cfg_if::cfg_if! {
    if #[cfg(feature="minpack-compat")] {
        pub type LevenbergMarquardt = lm::LevenbergMarquardt<f64>;
        pub type MinimizationReport = lm::MinimizationReport<f64>;
    } else {
        pub use lm::{LevenbergMarquardt, MinimizationReport};
    }
}
