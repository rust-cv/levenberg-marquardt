use crate::qr::PivotedQR;
use crate::trust_region::determine_lambda_and_parameter_update;
use crate::LeastSquaresProblem;
use nalgebra::{
    allocator::Allocator,
    constraint::{DimEq, ShapeConstraint},
    convert,
    storage::Storage,
    DefaultAllocator, Dim, DimMin, DimMinimum, RealField, Vector, VectorN, U1,
};
use num_traits::Float;

#[derive(Debug)]
/// Reasons for failure of the minimization.
pub enum Failure {
    /// The residual or Jacobian computation was not successful.
    User,
    /// Encountered `NaN` or `$\pm\infty$`.
    Numerical,
    /// A parameter update did not change `$f$`.
    NoImprovementPossible,
    /// Maximum number of function evaluations was hit.
    LostPatience,
    /// The number of parameters `$n$` is zero.
    NoParameters,
    /// Indicates that `$m < n$`, which is not allowed.
    NotEnoughResiduals,
}

#[derive(Debug)]
/// Information about the minimization.
///
/// Use this to inspect the minimization process. Most importantly
/// you may want to check if there was a failure.
pub struct MinimizationReport<F: RealField> {
    pub failure: Option<Failure>,
    pub number_of_evaluations: usize,
    /// Contains the value of `$f(\vec{x})$`.
    pub objective_function: F,
}

/// Helper to keep target and report about it together.
struct TargetReport<F, N, M, O>
where
    F: RealField,
    N: Dim,
    M: Dim,
    O: LeastSquaresProblem<F, N, M>,
{
    target: O,
    report: MinimizationReport<F>,
    marker: core::marker::PhantomData<(N, M)>,
}

impl<F, N, M, O> TargetReport<F, N, M, O>
where
    F: RealField,
    N: Dim,
    M: Dim,
    O: LeastSquaresProblem<F, N, M>,
{
    fn failure(self, failure: Failure) -> (O, MinimizationReport<F>) {
        (
            self.target,
            MinimizationReport {
                failure: Some(failure),
                ..self.report
            },
        )
    }

    fn success(self) -> (O, MinimizationReport<F>) {
        (self.target, self.report)
    }

    fn counted_residuals(&mut self) -> Option<Vector<F, M, O::ResidualStorage>> {
        self.report.number_of_evaluations += 1;
        let residuals = self.target.residuals();
        if let Some(residuals) = self.target.residuals() {
            self.report.objective_function = residuals.norm_squared() * convert(0.5);
        }
        residuals
    }
}

/// Levenberg-Marquardt optimization algorithm.
///
/// See the [module documentation](index.html) for a usage example.
///
/// The runtime and termination behavior can be controlled by various hyperparameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LevenbergMarquardt<F> {
    ftol: F,
    xtol: F,
    gtol: F,
    stepbound: F,
    patience: usize,
    scale_diag: bool,
}

impl<F: RealField + Float> LevenbergMarquardt<F> {
    pub fn new() -> Self {
        let user_tol = F::default_epsilon() * convert(30.0);
        Self {
            ftol: user_tol,
            xtol: user_tol,
            gtol: user_tol,
            stepbound: convert(100.0),
            patience: 100,
            scale_diag: true,
        }
    }

    /// Set the relative error desired in the objective function `$f$`.
    ///
    /// Termination occurs when both the actual and
    /// predicted relative reductions for `$f$` are at most `ftol`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{ftol} < 0$`.
    pub fn with_ftol(self, ftol: F) -> Self {
        assert!(!ftol.is_negative(), "ftol must be >= 0");
        Self { ftol, ..self }
    }

    /// Set relative error between last two approximations.
    ///
    /// Termination occurs when the relative error between
    /// two consecutive iterates is at most `xtol`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{xtol} < 0$`.
    pub fn with_xtol(self, xtol: F) -> Self {
        assert!(!xtol.is_negative(), "xtol must be >= 0");
        Self { xtol, ..self }
    }

    /// Set orthogonality desired between the residual vector and its derivative.
    ///
    /// Termination occurs when the cosine of the angle
    /// between the residual vector `$\vec{r}$` and any column of the Jacobian `$\mathbf{J}$` is at
    /// most `gtol` in absolute value.
    ///
    /// With other words, the algorithm will terminate if
    /// ```math
    ///   \max_{i=1,\ldots,n}\frac{|(\mathbf{J}^\top \vec{r})_i|}{\|\mathbf{J}\vec{e}_i\|\|\vec{r}\|} \leq \texttt{gtol}.
    /// ```
    ///
    /// This tests more or less if a _critical point_ was found, i.e., whether
    /// `$\nabla f(\vec{x}) = \mathbf{J}^\top\vec{r} \approx 0$`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{gtol} < 0$`.
    pub fn with_gtol(self, gtol: F) -> Self {
        assert!(!gtol.is_negative(), "gtol must be >= 0");
        Self { gtol, ..self }
    }

    /// Set factor for the initial step bound.
    ///
    /// This bound is set to `$\mathtt{stepbound}\cdot\|\mathbf{D}\vec{x}\|$`
    /// if nonzero, or else to `stepbound` itself. In most cases `stepbound` should lie
    /// in the interval `$[0.1,100]$`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{stepbound} \leq 0$`.
    pub fn with_stepbound(self, stepbound: F) -> Self {
        assert!(stepbound.is_positive(), "stepbound must be > 0");
        Self { stepbound, ..self }
    }

    /// Set the maximal number of function evaluations.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{patience} \leq 0$`.
    pub fn with_patience(self, patience: usize) -> Self {
        assert!(patience > 0, "patience must be > 0");
        Self { patience, ..self }
    }

    /// Enable or disable whether the variables will be rescaled internally.
    pub fn with_scale_diag(self, scale_diag: bool) -> Self {
        Self { scale_diag, ..self }
    }

    /// Try to solve the given least-squares problem.
    pub fn minimize<N, M, O>(
        &self,
        initial_x: Vector<F, N, O::ParameterStorage>,
        target: O,
    ) -> (O, MinimizationReport<F>)
    where
        N: DimMin<M>,
        M: DimMin<N>,
        O: LeastSquaresProblem<F, N, M>,
        DefaultAllocator: Allocator<F, N>
            + Allocator<F, N, N>
            + Allocator<F, M>
            + Allocator<F, N, Buffer = O::ParameterStorage>
            + Allocator<usize, N>,
        ShapeConstraint: DimEq<DimMinimum<N, M>, N> + DimEq<DimMinimum<M, N>, N>,
    {
        const P1: f64 = 0.1;
        const P0001: f64 = 1.0e-4;

        let mut report = TargetReport {
            target,
            report: MinimizationReport {
                failure: None,
                number_of_evaluations: 0,
                objective_function: <F as Float>::nan(),
            },
            marker: core::marker::PhantomData,
        };

        // Evaluate with at start point
        let mut x = initial_x;
        report.target.set_params(&mut x);
        let mut residuals = if let Some(residuals) = report.counted_residuals() {
            residuals
        } else {
            return report.failure(Failure::User);
        };
        // Compute norm
        let mut residuals_norm = report.report.objective_function * convert(2.0);

        // Initialize diagonal
        let n = x.data.shape().0;
        let mut diag = VectorN::<F, N>::from_element_generic(n, U1, F::one());
        // Check n > 0
        if diag.nrows() == 0 {
            return report.failure(Failure::NoParameters);
        }
        // Check m >= n
        if diag.nrows() > residuals.nrows() {
            return report.failure(Failure::NotEnoughResiduals);
        }
        if !residuals_norm.is_finite() {
            return report.failure(Failure::Numerical);
        }
        if residuals_norm <= Float::min_positive_value() {
            // Already zero, nothing to do
            return report.success();
        }

        let mut tmp = VectorN::<F, N>::zeros_generic(n, U1);

        let mut delta = F::zero();
        let mut lambda = F::zero();
        let mut xnorm = F::zero();

        let mut first_outer = true;
        loop {
            // Compute jacobian
            let jacobian = if let Some(jacobian) = report.target.jacobian() {
                jacobian
            } else {
                return report.failure(Failure::User);
            };

            let qr = PivotedQR::new(jacobian).ok().unwrap();
            let mut lls = qr.into_least_squares_diagonal_problem(residuals);

            // Compute norm of scaled gradient and detect degeneracy
            let gnorm = lls.max_a_t_b_scaled() / residuals_norm;
            if gnorm <= self.gtol {
                return report.success();
            }

            if first_outer {
                // Initialize diag and delta
                xnorm = if self.scale_diag {
                    for (d, col_norm) in diag.iter_mut().zip(lls.column_norms.iter()) {
                        *d = if col_norm.is_zero() {
                            F::one()
                        } else {
                            *col_norm
                        };
                    }
                    tmp.cmpy(F::one(), &diag, &x, F::zero());
                    tmp.norm()
                } else {
                    x.norm()
                };
                if !xnorm.is_finite() {
                    return report.failure(Failure::Numerical);
                }
                delta = if xnorm.is_zero() {
                    self.stepbound
                } else {
                    self.stepbound * xnorm
                };
            } else if self.scale_diag {
                // Update diag
                for (d, norm) in diag.iter_mut().zip(lls.column_norms.iter()) {
                    *d = Float::max(*norm, *d);
                }
            }

            let mut first_inner = true;
            residuals = loop {
                let param = determine_lambda_and_parameter_update(&mut lls, &diag, delta, lambda);
                lambda = param.lambda;
                let pnorm = param.dp_norm;
                if !pnorm.is_finite() {
                    return report.failure(Failure::Numerical);
                }
                // at first call, adjust the initial step bound
                if first_outer && first_inner && pnorm < delta {
                    delta = pnorm;
                }

                // These values are needed later. We check now to fail early.
                let temp2 = lambda * Float::powi(pnorm / residuals_norm, 2);
                if !temp2.is_finite() {
                    return report.failure(Failure::Numerical);
                }
                let temp1 = lls.a_x_norm_squared(&param.step) / Float::powi(residuals_norm, 2);
                if !temp1.is_finite() {
                    return report.failure(Failure::Numerical);
                }

                // Compute new parameters: x - p
                tmp.copy_from(&x);
                tmp.axpy(-F::one(), &param.step, F::one());
                // Evaluate
                report.target.set_params(&mut tmp);
                residuals = if let Some(residuals) = report.counted_residuals() {
                    residuals
                } else {
                    return report.failure(Failure::User);
                };
                let new_residuals_norm = report.report.objective_function * convert(2.);

                // Compute predicted and actual reduction
                let actual_reduction = if new_residuals_norm * convert(P1) < residuals_norm {
                    F::one() - Float::powi(new_residuals_norm / residuals_norm, 2)
                } else {
                    -F::one()
                };
                let predicted_reduction = temp1 + temp2 * convert(2.0);

                let ratio = if predicted_reduction.is_zero() {
                    F::zero()
                } else {
                    actual_reduction / predicted_reduction
                };
                let half: F = convert(0.5);
                if ratio <= convert(0.25) {
                    let mut temp = if !actual_reduction.is_negative() {
                        half
                    } else {
                        let dir_der = -temp1 + temp2;
                        half * dir_der / (dir_der + half * actual_reduction)
                    };
                    if new_residuals_norm * convert(P1) >= residuals_norm || temp < convert(P1) {
                        temp = convert(P1);
                    };
                    delta = temp * Float::min(delta, pnorm / convert(P1));
                    lambda /= temp;
                } else if lambda.is_zero() || ratio >= convert(0.75) {
                    delta = pnorm * convert(2.);
                    lambda *= half;
                }

                let inner_success = ratio >= convert(P0001);
                // on sucess, update x, residuals and their norms
                if inner_success {
                    core::mem::swap(&mut x, &mut tmp);
                    xnorm = if self.scale_diag {
                        tmp.cmpy(F::one(), &diag, &x, F::zero());
                        tmp.norm()
                    } else {
                        x.norm()
                    };
                    if !xnorm.is_finite() {
                        return report.failure(Failure::Numerical);
                    }
                    residuals_norm = new_residuals_norm;
                } else {
                    // Reset objective function value
                    report.report.objective_function = residuals_norm * convert(0.5);
                }

                // convergence tests
                if residuals_norm <= F::min_positive_value()
                    || (Float::abs(actual_reduction) <= self.ftol
                        && predicted_reduction <= self.ftol
                        && ratio <= convert(2.))
                    || delta <= self.xtol * xnorm
                {
                    return report.success();
                }

                // termination tests
                if report.report.number_of_evaluations >= self.patience {
                    return report.failure(Failure::LostPatience);
                }
                if (Float::abs(actual_reduction) <= F::default_epsilon()
                    && predicted_reduction <= F::default_epsilon()
                    && ratio <= convert(2.))
                    || delta <= F::default_epsilon() * xnorm
                    || gnorm <= F::default_epsilon()
                {
                    return report.failure(Failure::NoImprovementPossible);
                }

                first_inner = false;
                if inner_success {
                    break residuals;
                }
            };
            first_outer = false;
        }
    }
}

impl<F: RealField + Float> Default for LevenbergMarquardt<F> {
    fn default() -> Self {
        Self::new()
    }
}
