//! Solver for the trust-region sub-problem in the LM algorithm.
use crate::qr::LinearLeastSquaresDiagonalProblem;
use nalgebra::{
    allocator::Allocator, convert, storage::ContiguousStorageMut, DefaultAllocator, Dim, DimName,
    RealField, VectorN,
};
use num_traits::real::Real;

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
/// information about this algorithm but it misses a few details.
pub fn determine_lambda_and_parameter_update<F, M, N, S>(
    mut lls: LinearLeastSquaresDiagonalProblem<F, M, N, S>,
    diag: &VectorN<F, N>,
    delta: F,
    initial_lambda: F,
) -> (VectorN<F, N>, F)
where
    F: RealField + Real,
    M: Dim,
    N: DimName,
    S: ContiguousStorageMut<F, M, N>,
    DefaultAllocator: Allocator<F, N> + Allocator<usize, N>,
{
    const REL_ERR: f64 = 0.1;
    debug_assert!(delta.is_positive());
    debug_assert!(initial_lambda >= F::zero());
    debug_assert!(!diag.iter().any(F::is_zero));

    let has_full_rank = lls.has_full_rank();
    let (mut p, mut l) = lls.solve_with_zero_diagonal();
    let mut diag_p = p.component_mul(&diag);
    let diag_p_norm = diag_p.norm();
    let mut fp = diag_p_norm - delta;
    if fp <= delta * convert(REL_ERR) {
        // we have a feasible p with lambda = 0
        return (p, F::zero());
    }

    // we now look for lambda > 0 with ||D p|| = delta
    // by using an approximate Newton iteration.

    let mut lambda_lower = if has_full_rank {
        p.cmpy(Real::recip(diag_p_norm), diag, &diag_p, F::zero());
        p = l.solve(p);
        let norm = p.norm();
        fp / delta / norm / norm
    } else {
        F::zero()
    };

    let gnorm;
    let mut lambda_upper = {
        // Upper bound is given by ||(J * D^T)^T r|| / delta, see paper cited above.
        p = l.mul_qt_b(p);
        for j in 0..p.nrows() {
            p[j] /= diag[l.permutation[j]];
        }
        gnorm = p.norm();
        let upper = gnorm / delta;
        if upper.is_zero() {
            F::min_positive_value() / Real::min(delta, convert(REL_ERR))
        } else {
            upper
        }
    };

    let mut lambda = Real::min(Real::max(initial_lambda, lambda_lower), lambda_upper);
    if lambda.is_zero() {
        lambda = gnorm / delta;
    }

    for iteration in 0.. {
        if lambda.is_zero() {
            lambda = Real::max(F::min_positive_value(), lambda * convert(0.001));
        }
        let l_sqrt = Real::sqrt(lambda);
        diag_p.axpy(l_sqrt, diag, F::zero());
        let (p_new, mut l) = lls.solve_with_diagonal(&diag_p, p);
        p = p_new;
        if iteration == 10 {
            break;
        }
        diag_p = p.component_mul(&diag);
        let diag_p_norm = diag_p.norm();
        let fp_old = fp;
        fp = diag_p_norm - delta;
        if Real::abs(fp) <= delta * convert(REL_ERR)
            || (lambda_lower.is_zero() && fp <= fp_old && fp_old.is_negative())
        {
            break;
        }

        let newton_correction = {
            p.cmpy(Real::recip(diag_p_norm), diag, &diag_p, F::zero());
            p = l.solve(p);
            let norm = p.norm();
            fp / delta / norm / norm
        };

        if fp.is_positive() {
            lambda_lower = Real::max(lambda_lower, lambda);
        } else {
            lambda_upper = Real::min(lambda_upper, lambda);
        }
        lambda = Real::max(lambda_lower, lambda + newton_correction);
    }

    (p, lambda)
}

#[cfg(test)]
mod tests {
    use super::determine_lambda_and_parameter_update;
    use crate::qr::*;
    use nalgebra::*;

    #[test]
    fn test_case1() {
        let j = Matrix4x3::from_column_slice(&[
            33., -40., 44., -43., -37., -1., -40., 48., 43., -11., -40., 43.,
        ]);
        let residual = Vector4::new(7., -1., 0., -1.);

        let qr = PivotedQR::new(j).ok().unwrap();
        let lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = Vector3::new(18.2, 18.2, 3.2);
        let (p_o, l_o) = determine_lambda_and_parameter_update(lls, &diag, 0.5, 0.2);

        assert!((l_o - 34.628643558156341f64).abs() < 1e-10);
        let p_r = Vector3::new(0.017591648698939, -0.020395135814051, 0.059285196018896);
        assert!((p_o - p_r).norm() < 1e-10);
    }

    #[test]
    fn test_case2() {
        let j = Matrix4x3::from_column_slice(&[
            -7., 28., -40., 29., 7., -49., -39., 43., -25., -47., -11., 34.,
        ]);
        let residual = Vector4::new(-7., -8., -8., -10.);

        let qr = PivotedQR::new(j).ok().unwrap();
        let lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = Vector3::new(10.2, 13.2, 1.2);
        let (p_o, l_o) = determine_lambda_and_parameter_update(lls, &diag, 0.5, 0.2);

        assert!(l_o == 0.0);
        let p_r = Vector3::new(-0.048474221517806, -0.007207732068190, 0.083138659283539);
        assert!((p_o - p_r).norm() < 1e-10);
    }

    #[test]
    fn test_case3() {
        let j = Matrix4x3::from_column_slice(&[
            8., -42., -34., -31., -30., -15., -36., -1., 27., 22., 44., 6.,
        ]);
        let residual = Vector4::new(1., -5., 2., 7.);

        let qr = PivotedQR::new(j).ok().unwrap();
        let lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = Vector3::new(4.2, 8.2, 11.2);
        let (p_o, l_o) = determine_lambda_and_parameter_update(lls, &diag, 0.5, 0.2);

        assert!((l_o - 0.017646940861467f64).abs() < 1e-10);
        let p_r = Vector3::new(-0.008462374169585, 0.033658082419054, 0.037230479167632);
        assert!((p_o - p_r).norm() < 1e-10);
    }

    #[test]
    fn test_case4() {
        let j = Matrix4x3::from_column_slice(&[
            14., -12., 20., -11., 19., 38., -4., -11., -14., 12., -20., 11.,
        ]);
        let residual = Vector4::new(-5., 3., -2., 7.);

        let qr = PivotedQR::new(j).ok().unwrap();
        let lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = Vector3::new(6.2, 1.2, 0.2);
        let (p_o, l_o) = determine_lambda_and_parameter_update(lls, &diag, 0.5, 0.2);

        assert!(l_o.abs() < 1e-15);
        let p_r = Vector3::new(-0.000277548738904, -0.046232379576219, 0.266724338086713);
        assert!((p_o - p_r).norm() < 1e-10);
    }
}
