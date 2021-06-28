//! Solver for the trust-region sub-problem in the LM algorithm.
#![allow(clippy::excessive_precision)]

use crate::qr::LinearLeastSquaresDiagonalProblem;
use crate::utils::{dwarf, enorm};
use nalgebra::{
    allocator::Allocator, convert, DefaultAllocator, Dim, DimMax, DimMaximum, DimMin, OVector,
    RealField,
};
use num_traits::Float;

pub struct LMParameter<F: RealField, N: Dim>
where
    DefaultAllocator: Allocator<F, N>,
{
    pub step: OVector<F, N>,
    pub lambda: F,
    pub dp_norm: F,
}

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
pub fn determine_lambda_and_parameter_update<F, M, N>(
    lls: &mut LinearLeastSquaresDiagonalProblem<F, M, N>,
    diag: &OVector<F, N>,
    delta: F,
    initial_lambda: F,
) -> LMParameter<F, N>
where
    F: RealField + Float,
    N: Dim,
    M: Dim + DimMin<N> + DimMax<N>,
    DefaultAllocator: Allocator<F, N> + Allocator<F, DimMaximum<M, N>, N> + Allocator<usize, N>,
{
    const P1: f64 = 0.1;
    debug_assert!(delta.is_positive());
    debug_assert!(initial_lambda >= F::zero());
    debug_assert!(!diag.iter().any(F::is_zero));

    let is_non_singular = lls.is_non_singular();
    let (mut p, mut l) = lls.solve_with_zero_diagonal();
    let mut diag_p = p.component_mul(diag);
    let mut diag_p_norm = enorm(&diag_p);
    let mut fp = diag_p_norm - delta;
    if fp <= delta * convert(P1) {
        // we have a feasible p with lambda = 0
        return LMParameter {
            step: p,
            lambda: F::zero(),
            dp_norm: diag_p_norm,
        };
    }

    // we now look for lambda > 0 with ||D p|| = delta
    // by using an approximate Newton iteration.

    let mut lambda_lower = if is_non_singular {
        p.copy_from(&diag_p);
        p /= diag_p_norm;
        for (p, d) in p.iter_mut().zip(diag.iter()) {
            *p *= *d;
        }
        p = l.solve(p);
        let norm = enorm(&p);
        ((fp / delta) / norm) / norm
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
        gnorm = enorm(&p);
        let upper = gnorm / delta;
        if upper.is_zero() {
            dwarf::<F>() / Float::min(delta, convert(P1))
        } else {
            upper
        }
    };

    let mut lambda = Float::min(Float::max(initial_lambda, lambda_lower), lambda_upper);
    if lambda.is_zero() {
        lambda = gnorm / diag_p_norm;
    }

    for iteration in 1.. {
        if lambda.is_zero() {
            lambda = Float::max(dwarf(), lambda_upper * convert(0.001));
        }
        let l_sqrt = Float::sqrt(lambda);
        diag_p.axpy(l_sqrt, diag, F::zero());
        let (p_new, mut l) = lls.solve_with_diagonal(&diag_p, p);
        p = p_new;
        diag_p = p.component_mul(diag);
        diag_p_norm = enorm(&diag_p);
        if iteration == 10 {
            break;
        }
        let fp_old = fp;
        fp = diag_p_norm - delta;
        if Float::abs(fp) <= delta * convert(P1)
            || (lambda_lower.is_zero() && fp <= fp_old && fp_old.is_negative())
        {
            break;
        }

        let newton_correction = {
            p.copy_from(&diag_p);
            p /= diag_p_norm;
            for (p, d) in p.iter_mut().zip(diag.iter()) {
                *p *= *d;
            }
            p = l.solve(p);
            let norm = enorm(&p);
            ((fp / delta) / norm) / norm
        };

        if fp.is_positive() {
            lambda_lower = Float::max(lambda_lower, lambda);
        } else {
            lambda_upper = Float::min(lambda_upper, lambda);
        }
        lambda = Float::max(lambda_lower, lambda + newton_correction);
    }

    LMParameter {
        step: p,
        lambda,
        dp_norm: diag_p_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::determine_lambda_and_parameter_update;
    use crate::qr::*;
    use approx::assert_relative_eq;
    use nalgebra::*;

    #[test]
    fn test_case1() {
        let j = Matrix4x3::from_column_slice(&[
            33., -40., 44., -43., -37., -1., -40., 48., 43., -11., -40., 43.,
        ]);
        let residual = Vector4::new(7., -1., 0., -1.);

        let qr = PivotedQR::new(j);
        let mut lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = Vector3::new(18.2, 18.2, 3.2);
        let param = determine_lambda_and_parameter_update(&mut lls, &diag, 0.5, 0.2);

        assert_relative_eq!(param.lambda, 34.628643558156341f64);
        let p_r = Vector3::new(0.017591648698939, -0.020395135814051, 0.059285196018896);
        assert_relative_eq!(param.step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case2() {
        let j = Matrix4x3::from_column_slice(&[
            -7., 28., -40., 29., 7., -49., -39., 43., -25., -47., -11., 34.,
        ]);
        let residual = Vector4::new(-7., -8., -8., -10.);

        let qr = PivotedQR::new(j);
        let mut lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = Vector3::new(10.2, 13.2, 1.2);
        let param = determine_lambda_and_parameter_update(&mut lls, &diag, 0.5, 0.2f64);

        assert_eq!(param.lambda.classify(), ::core::num::FpCategory::Zero);
        let p_r = Vector3::new(-0.048474221517806, -0.007207732068190, 0.083138659283539);
        assert_relative_eq!(param.step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case3() {
        let j = Matrix4x3::from_column_slice(&[
            8., -42., -34., -31., -30., -15., -36., -1., 27., 22., 44., 6.,
        ]);
        let residual = Vector4::new(1., -5., 2., 7.);

        let qr = PivotedQR::new(j);
        let mut lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = Vector3::new(4.2, 8.2, 11.2);
        let param = determine_lambda_and_parameter_update(&mut lls, &diag, 0.5, 0.2);

        assert_relative_eq!(param.lambda, 0.017646940861467262f64, epsilon = 1e-14);
        let p_r = Vector3::new(-0.008462374169585, 0.033658082419054, 0.037230479167632);
        assert_relative_eq!(param.step, p_r, epsilon = 1e-14);
    }

    #[test]
    fn test_case4() {
        let j = Matrix4x3::from_column_slice(&[
            14., -12., 20., -11., 19., 38., -4., -11., -14., 12., -20., 11.,
        ]);
        let residual = Vector4::new(-5., 3., -2., 7.);

        let qr = PivotedQR::new(j);
        let mut lls = qr.into_least_squares_diagonal_problem(residual);
        let diag = Vector3::new(6.2, 1.2, 0.2);
        let param = determine_lambda_and_parameter_update(&mut lls, &diag, 0.5, 0.2);

        assert_relative_eq!(param.lambda, 0.);
        let p_r = Vector3::new(-0.000277548738904, -0.046232379576219, 0.266724338086713);
        assert_relative_eq!(param.step, p_r, epsilon = 1e-14);
    }
}
