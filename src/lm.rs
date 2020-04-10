use nalgebra::{
    allocator::Allocator,
    constraint::{DimEq, ShapeConstraint},
    DefaultAllocator, Dim, DimMin, DimMinimum, DimName, RealField,
};
use num_traits::FromPrimitive;

use crate::LeastSquaresProblem;

/// Levenberg-Marquardt optimization algorithm.
///
/// See the [module documentation](index.html) for a usage example.
///
/// The runtime and termination behavior can be controlled by various hyperparameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LevenbergMarquardt<F> {
    /// Limit for the maximum number of iterations.
    pub max_iterations: usize,
    /// Limit for the number of times that `$\lambda$` can diverge
    /// consecutively from Gauss-Newton due to a failed improvement. Once the
    /// solution is as good as possible, it will begin regressing to gradient descent. This
    /// limit prevents it from wasting the remaining cycles of the algorithm.
    pub consecutive_divergence_limit: usize,
    /// The initial value for `$\lambda$`. As `$\lambda$` grows higher,
    /// Levenberg-Marquardt approaches gradient descent, which is better at converging to a distant
    /// minima. As `$\lambda$` grows lower, Levenberg-Marquardt approaches Gauss-Newton, which allows faster
    /// convergence closer to the minima. A `$\lambda$` of `0.0` would imply that it is purely based on
    /// Gauss-Newton approximation. Please do not set `$\lambda$` to exactly `0.0` or the `lambda_scale` will be unable to
    /// increase `$\lambda$` since it does so through multiplication.
    pub initial_lambda: F,
    /// Must be set to a value below `1.0`. On each iteration of Levenberg-Marquardt,
    /// the lambda is used as-is and multiplied by `lambda_converge`. If the original lambda or the
    /// new lambda is better, that lambda becomes the new lambda. If neither are better than the
    /// previous sum-of-squares, then lambda is multiplied by `lambda_diverge`.
    pub lambda_converge: F,
    /// Must be set to a value above `1.0` and highly recommended to set it **above**
    /// `lambda_converge^-1` (it will re-test an already-used lambda otherwise). On each iteration,
    /// if the sum-of-squares regresses, then lambda is multiplied by `lambda_diverge` to move closer
    /// to gradient descent in hopes that it will cause it to converge.
    pub lambda_diverge: F,
    /// The threshold at which the average-of-squares is low enough that the algorithm can
    /// terminate. This exists so that the algorithm can short-circuit and exit early if the
    /// solution was easy to find. Set this to `0.0` if you want it to continue for all `max_iterations`.
    /// You might do that if you always have a fixed amount of time per optimization, such as when
    /// processing live video frames.
    pub threshold: F,
}

impl<F> Default for LevenbergMarquardt<F>
where
    F: FromPrimitive,
{
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            consecutive_divergence_limit: 5,
            initial_lambda: F::from_f32(50.0)
                .expect("leverberg-marquardt vector and matrix type cant store 50.0"),
            lambda_converge: F::from_f32(0.8)
                .expect("leverberg-marquardt vector and matrix type cant store 0.8"),
            lambda_diverge: F::from_f32(2.0)
                .expect("leverberg-marquardt vector and matrix type cant store 2.0"),
            threshold: F::from_f32(0.0)
                .expect("leverberg-marquardt vector and matrix type cant store 0.0"),
        }
    }
}

impl<F> LevenbergMarquardt<F> {
    /// Try to solve the given least-squares problem.
    ///
    /// # Initial value
    ///
    /// The initial guess for the paramters `$\vec{x}$` must be
    /// already stored in `target`. A good initial guess is usually essential
    /// for the minimization to be successful.
    pub fn minimize<N, M, O>(&self, mut target: O) -> O
    where
        F: RealField,
        N: DimMin<N> + DimName,
        M: Dim,
        O: LeastSquaresProblem<F, N, M>,
        DefaultAllocator: Allocator<F, N>,
        DefaultAllocator: Allocator<F, N, N>,
        DefaultAllocator: Allocator<F, N, Buffer = O::ParameterStorage>,
        ShapeConstraint: DimEq<DimMinimum<N, N>, N>,
    {
        let mut lambda = self.initial_lambda;
        let mut res = target.residuals();
        let mut sum_of_squares = res.norm_squared();
        let mut consecutive_divergences = 0;
        let total = F::from_usize(res.len())
            .expect("there were more items in the vector than could be represented by the type");

        for _ in 0..self.max_iterations {
            let (hessian, gradients) = {
                let jacobian = target.jacobian();
                (jacobian.tr_mul(&jacobian), jacobian.tr_mul(&res))
            };

            // Get a tuple of the lambda, guess, residual, and sum-of-squares.
            // Returns an option because it may not be possible to solve the inverse.
            let lam_tar_res_sum = |lam| {
                // Compute JJᵀ + λ*diag(JJᵀ).
                let mut hessian_lambda_diag = hessian.clone();
                let new_diag = hessian_lambda_diag.map_diagonal(|n| n * (lam + F::one()));
                hessian_lambda_diag.set_diagonal(&new_diag);

                // Invert JᵀJ + λ*diag(JᵀJ) and solve for delta.
                let delta = hessian_lambda_diag
                    .try_inverse()
                    .map(|inv_jjl| -inv_jjl * &gradients);
                // Compute the new guess, residuals, and sum-of-squares.
                let vars = delta.map(|delta| {
                    let mut target = target.clone();
                    target.apply_parameter_step(&delta);
                    let res = target.residuals();
                    let sum = res.norm_squared();
                    (lam, target, res, sum)
                });
                // If the sum-of-squares is infinite or NaN it shouldn't be allowed through.
                vars.filter(|vars| vars.3.is_finite())
            };

            // Select the vars that minimize the sum-of-squares the most.
            let smaller_lambda = lambda * self.lambda_converge;
            let new_vars = match (lam_tar_res_sum(smaller_lambda), lam_tar_res_sum(lambda)) {
                (Some(s_vars), Some(o_vars)) => {
                    Some(if s_vars.3 < o_vars.3 { s_vars } else { o_vars })
                }
                (Some(vars), None) | (None, Some(vars)) => Some(vars),
                (None, None) => None,
            };

            if let Some((n_lam, n_tar, n_res, n_sum)) = new_vars {
                // We didn't see a decrease in the new state.
                if n_sum > sum_of_squares {
                    // Increase lambda twice and go to the next iteration.
                    // Increase twice so that the new two tested lambdas are different than current.
                    lambda *= self.lambda_diverge;
                    consecutive_divergences += 1;
                } else {
                    // There was a decrease, so update everything.
                    lambda = n_lam;
                    target = n_tar;
                    res = n_res;
                    sum_of_squares = n_sum;
                    consecutive_divergences = 0;
                }
            } else {
                // We were unable to take the inverse, so increase lambda in hopes that it may
                // cause the matrix to become invertible.
                lambda *= self.lambda_diverge;
                consecutive_divergences += 1;
            }

            // Terminate early if we hit the consecutive divergence limit.
            if consecutive_divergences == self.consecutive_divergence_limit {
                break;
            }

            // We can terminate early if the sum of squares is below the threshold.
            if sum_of_squares < self.threshold * total {
                break;
            }
        }
        target
    }
}
