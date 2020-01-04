//! See [`optimize`] for documentation on the Levenberg-Marquardt optimization algorithm.

#![no_std]

use nalgebra::{storage::Storage, Dim, Matrix, Scalar, Vector};

/// Note that the differentials and state vector are represented with column vectors.
/// This is atypical from the normal way it is done in mathematics. This is done because
/// nalgebra is column-major. A nalgebra `Vector` is a column vector.
///
/// Make sure that you create your Jacobian such that it is several fixed length
/// column vectors rather than several row vectors as per normal. If you have already
/// computed it with row vectors, then you can take the transpose.
///
/// It is recommended to make the number of columns dynamic unless you have a small fixed
/// number of data-points.
///
/// `max_iterations` limits the number of times the initial guess will be updated.
///
/// `initial_lambda` defines the initial lambda value. As lambda grows higher,
/// Levenberg-Marquardt approaches gradient descent, which is better at converging to a distant
/// minima. As lambda grows lower, Levenberg-Marquardt approaches the minima of the first-order
/// taylor approximation, which allows faster convergence closer to the minima.
/// A lambda of `0.0` would imply that it is purely based on the first-order taylor approximation.
/// Please do not set lambda to exactly `0.0` or the `lambda_scale` will be unable to
/// increase lambda since it does so through multiplication.
///
/// `lambda_scale` should be set to a value above `1.0`. On each iteration of Levenberg-Marquardt,
/// the lambda is used as-is and divided by `lambda_scale`. If the original lambda or the
/// new lambda is better, that lambda becomes the new lambda. If neither are better that the
/// previous sum-of-squares, then lambda is multiplied by `lambda_scale`.
///
/// `threshold` is the point at which the average-of-squares is low enough that the algorithm can
/// terminate. This exists so that the algorithm can short-circuit and exit early if the
/// solution was easy to find. Set this to `0.0` if you want it to continue for all `max_iterations`.
/// You might do that if you always have a fixed amount of time per optimization, such as when
/// processing live video frames.
///
/// `init` is the initial guess. Make sure to set `init` close to the actual solution.
/// It is recommended to use a sample consensus algorithm to get a close initial approximation.
///
/// `jacobian_and_residuals` is a function that takes in the current guess and produces the Jacobian
/// matrix of the function that is being optimized in respect to the guess and the residuals, which
/// are the difference between the expected output and the output given the current guess. The underlying
/// data is not required by `optimize`. Only the Jacobian and the residuals are required to perform
/// Levenberg-Marquardt optimization. You will need to caputure your inputs and outputs in the closure
/// to compute these, but they are not arguments since they are constants to Levenberg-Marquardt.
pub fn optimize<N: Scalar, R: Dim, C: Dim, VS: Storage<N, R, C>, MS: Storage<N, R>>(
    max_iterations: usize,
    initial_lambda: N,
    lambda_scale: N,
    threshold: N,
    init: Vector<N, R, VS>,
    jacobian_and_residuals: impl Fn(Vector<N, R, VS>) -> (Matrix<N, R, C, MS>, Matrix<N, R, C, MS>),
) -> Vector<N, R, VS> {
    unimplemented!()
}
