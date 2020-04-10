//! Solver for the trust-region sub-problem in the LM algorithm.
use crate::qr::LinearLeastSquaresDiagonalProblem;
use nalgebra::{
    allocator::Allocator, convert, storage::ContiguousStorageMut, DefaultAllocator, Dim, DimName,
    RealField, VectorN,
};

/// Approximately solve the LM trust-region subproblem.
///
/// Given `$\mathbf{F}\in\R^{m\times n}$` and a non-singular diagonal matrix `$\mathbf{D}$`
/// this routine approximately solves the problem
/// ```math
///   \min_{\vec{p}\in\R^n}\|\mathbf{J}\vec{p} - \vec{r}\|^2\text{ subject to }\|\mathbf{D}\vec{p}\|\leq\Delta.
/// ```
///
/// It can be shown that `$\vec{p}$` with `$\|\mathbf{D}\vec{p}\|\leq\Delta$` is
/// a solution if and only if there exists `$\lambda\geq 0$` such that
/// ```math
/// \begin{aligned}
/// (\mathbf{J}^\top\mathbf{J} + \lambda \mathbf{D}\mathbf{D})\vec{p} &= \mathbf{J}^\top\vec{r}, \\
/// \lambda(\Delta - \|\mathbf{D}\vec{p}\|) &= 0.
/// \end{aligned}
/// ```
///
/// # Inputs
///
/// The matrix `$\mathbf{F}$` and vector `$\vec{r}$` correspond to `$\mathbf{A}$` and
/// `$\vec{b}$` of [`LinearLeastSquaresDiagonalProblem`](../qr/struct.LinearLeastSquaresDiagonalProblem.html).
///
/// # Reference
///
/// This method resembles `LMPAR` from `MINPACK`. See the following paper
/// on how it works:
///
/// > Moré J.J. (1978) The Levenberg-Marquardt algorithm: Implementation and theory. In: Watson G.A. (eds) Numerical Analysis. Lecture Notes in Mathematics, vol 630. Springer, Berlin, Heidelberg.
///
/// Chapter 4.3 of "Numerical Optimization" by Nocedal and Wright also contains
/// information about this algorithm but is misses a few details.
pub fn determine_lambda_and_parameter_update<F, M, N, S>(
    mut lls: LinearLeastSquaresDiagonalProblem<F, M, N, S>,
    diag: &VectorN<F, N>,
    delta: F,
    initial_lambda: F,
) -> (VectorN<F, N>, F)
where
    F: RealField,
    M: Dim,
    N: DimName,
    S: ContiguousStorageMut<F, M, N>,
    DefaultAllocator: Allocator<F, N> + Allocator<usize, N>,
{
    let (mut p, _) = lls.solve_with_zero_diagonal();
    let diag_p = p.component_mul(&diag);
    let diag_p_norm = diag_p.norm();
    let fp = diag_p_norm - delta;
    if fp <= delta * convert(0.1f64) {
        // we have a feasible p with lambda = 0
        return (p, F::zero());
    }

    // we now look for lambda > 0 with ||D p|| = delta
    // by using an approximate Newton iteration.

    let lambda_min = if lls.has_full_rank() { 0 } else { 0 };
    todo!()
}
