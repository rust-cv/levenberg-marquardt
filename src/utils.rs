use crate::LeastSquaresProblem;
use nalgebra::{Dim, RealField, Vector};

/// Compute a [numerical approximation](https://en.wikipedia.org/wiki/Numerical_differentiation)
/// to the Jacobian for testing.
///
/// The function is intended to be used for debugging or testing.
/// You can try to check your derivative implementation of an
/// [`LeastSquaresProblem`](trait.LeastSquaresProblem.html) with this.
///
/// Computing the derivatives numerically is unstable: You can construct
/// functions where the computed result is catastrophically wrong. If you
/// observe large differences between the derivative computed by this function
/// and your implementation the reason _might_ be due to instabilty.
///
/// The achieved precision by this function
/// is lower than the floating point precision in general. So the error is bigger
/// bigger than `$10^{-15}$` for `f64` and bigger than `$10^{-7}$` for `f32`. See the example
/// below for what that means in your tests. If possible use `f64` for the testing.
///
/// # Example
///
/// **TODO**
pub fn differentiate_numerically<F, N, M, O>(x: Vector<F, N, O::ParameterStorage>, problem: O)
where
    F: RealField,
    N: Dim,
    M: Dim,
    O: LeastSquaresProblem<F, N, M>,
{
    todo!()
}
