use crate::qr::{LinearLeastSquaresDiagonalProblem, PivotedQR};
use crate::trust_region::{determine_lambda_and_parameter_update, LMParameter};
use crate::utils::{enorm, epsmch};
use crate::LeastSquaresProblem;
use nalgebra::{
    allocator::{Allocator, Reallocator},
    convert, DefaultAllocator, Dim, DimMax, DimMaximum, DimMin, Matrix, OVector, RealField, Vector,
};
use num_traits::Float;

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::redundant_clone
)]
pub(crate) mod test_examples;
#[cfg(test)]
mod test_helpers;
#[cfg(test)]
mod test_init_step;
#[cfg(test)]
#[allow(clippy::float_cmp, clippy::clone_on_copy, clippy::redundant_clone)]
mod test_update_diag;

#[derive(PartialEq, Eq, Debug)]
/// Reasons for terminating the minimization.
pub enum TerminationReason {
    /// The residual or Jacobian computation was not successful, it returned `None`.
    User(&'static str),
    /// Encountered `NaN` or `$\pm\infty$`.
    Numerical(&'static str),
    /// The residuals are literally zero.
    ResidualsZero,
    /// The residuals vector and the Jacobian columns are almost orthogonal.
    ///
    /// This is the `gtol` termination criterion.
    Orthogonal,
    /// The `ftol` or `xtol` criterion was fulfilled.
    Converged { ftol: bool, xtol: bool },
    /// The bound for `ftol`, `xtol` or `gtol` was set so low that the
    /// test passed with the machine epsilon but not with the actual
    /// bound. This means you must increase the bound.
    NoImprovementPossible(&'static str),
    /// Maximum number of function evaluations was hit.
    LostPatience,
    /// The number of parameters `$n$` is zero.
    NoParameters,
    /// The number of residuals `$m$` is zero.
    NoResiduals,
    /// The shape of the computed residuals or Jacobian is not correct.
    WrongDimensions(&'static str),
}

impl TerminationReason {
    /// Compute whether the outcome is considered successful.
    ///
    /// This does not necessarily mean we have a minimizer.
    /// Some termination criteria are approximations for necessary
    /// optimality conditions or check limitations due to
    /// floating point arithmetic.
    pub fn was_successful(&self) -> bool {
        matches!(
            self,
            TerminationReason::ResidualsZero
                | TerminationReason::Orthogonal
                | TerminationReason::Converged { .. }
        )
    }

    /// A fundamental assumptions was not met.
    ///
    /// For example if the number of residuals changed.
    pub fn was_usage_issue(&self) -> bool {
        matches!(
            self,
            TerminationReason::NoParameters
                | TerminationReason::NoResiduals
                | TerminationReason::NoImprovementPossible(_)
                | TerminationReason::WrongDimensions(_)
        )
    }
}

#[derive(Debug)]
/// Information about the minimization.
///
/// Use this to inspect the minimization process. Most importantly
/// you may want to check if there was a failure.
pub struct MinimizationReport<F: RealField> {
    pub termination: TerminationReason,
    /// Number of residuals which were computed.
    pub number_of_evaluations: usize,
    /// Contains the value of `$f(\vec{x})$`.
    pub objective_function: F,
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

impl<F: RealField + Float> Default for LevenbergMarquardt<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: RealField + Float> LevenbergMarquardt<F> {
    pub fn new() -> Self {
        if cfg!(feature = "minpack-compat") {
            let user_tol = convert(1.49012e-08);
            Self {
                ftol: user_tol,
                xtol: user_tol,
                gtol: F::zero(),
                stepbound: convert(100.0),
                patience: 100,
                scale_diag: true,
            }
        } else {
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
    }

    /// Set the relative error desired in the objective function `$f$`.
    ///
    /// Termination occurs when both the actual and
    /// predicted relative reductions for `$f$` are at most `ftol`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{ftol} < 0$`.
    #[must_use]
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
    #[must_use]
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
    ///   \cos\bigl(\sphericalangle (\mathbf{J}\vec{e}_i, \vec{r})\bigr) =
    ///   \frac{|(\mathbf{J}^\top \vec{r})_i|}{\|\mathbf{J}\vec{e}_i\|\|\vec{r}\|} \leq \texttt{gtol}
    ///   \quad\text{for all }i=1,\ldots,n.
    /// ```
    ///
    /// This is based on the fact that those vectors are orthognal near the optimum (gradient is zero).
    /// The angle check is scale invariant, whereas checking that
    /// `$\nabla f(\vec{x})\approx \vec{0}$` is not.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{gtol} < 0$`.
    #[must_use]
    pub fn with_gtol(self, gtol: F) -> Self {
        assert!(!gtol.is_negative(), "gtol must be >= 0");
        Self { gtol, ..self }
    }

    /// Shortcut to set `tol` as in MINPACK `LMDER1`.
    ///
    /// Sets `ftol = xtol = tol` and `gtol = 0`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{tol} \leq 0$`.
    #[must_use]
    pub fn with_tol(self, tol: F) -> Self {
        assert!(tol.is_positive(), "tol must > 0");
        Self {
            ftol: tol,
            xtol: tol,
            gtol: F::zero(),
            ..self
        }
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
    #[must_use]
    pub fn with_stepbound(self, stepbound: F) -> Self {
        assert!(stepbound.is_positive(), "stepbound must be > 0");
        Self { stepbound, ..self }
    }

    /// Set factor for the maximal number of function evaluations.
    ///
    /// The maximal number of function evaluations is set to
    /// `$\texttt{patience}\cdot(n + 1)$`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{patience} \leq 0$`.
    #[must_use]
    pub fn with_patience(self, patience: usize) -> Self {
        assert!(patience > 0, "patience must be > 0");
        Self { patience, ..self }
    }

    /// Enable or disable whether the variables will be rescaled internally.
    #[must_use]
    pub fn with_scale_diag(self, scale_diag: bool) -> Self {
        Self { scale_diag, ..self }
    }

    /// Try to solve the given least squares problem.
    ///
    /// The paramters of the problem which are set when this function is called
    /// are used as the initial guess for `$\vec{x}$`.
    pub fn minimize<N, M, O>(&self, target: O) -> (O, MinimizationReport<F>)
    where
        N: Dim,
        M: DimMin<N> + DimMax<N>,
        O: LeastSquaresProblem<F, M, N>,
        DefaultAllocator:
            Allocator<F, N> + Reallocator<F, M, N, DimMaximum<M, N>, N> + Allocator<usize, N>,
    {
        let (mut lm, mut residuals) = match LM::new(self, target) {
            Err(report) => return report,
            Ok(res) => res,
        };
        let n = lm.x.nrows();
        loop {
            // Build linear least squaress problem used for the trust-region subproblem
            let mut lls = {
                let jacobian = match lm.jacobian() {
                    Err(reason) => return lm.into_report(reason),
                    Ok(jacobian) => jacobian,
                };
                if jacobian.ncols() != n || jacobian.nrows() != lm.m {
                    return lm.into_report(TerminationReason::WrongDimensions("jacobian"));
                }

                let qr = PivotedQR::new(jacobian);
                qr.into_least_squares_diagonal_problem(residuals)
            };

            // Update the diagonal, initialize "delta" in first call
            if let Err(reason) = lm.update_diag(&mut lls) {
                return lm.into_report(reason);
            };

            residuals = loop {
                let param =
                    determine_lambda_and_parameter_update(&mut lls, &lm.diag, lm.delta, lm.lambda);
                let tr_iteration = lm.trust_region_iteration(&mut lls, param);
                match tr_iteration {
                    // successful paramter update, break and recompute Jacobian
                    Ok(Some(residuals)) => break residuals,
                    // terminate (either success or failure)
                    Err(reason) => return lm.into_report(reason),
                    // need another iteration
                    Ok(None) => (),
                }
            };
        }
    }
}

/// Struct which holds the state of the LM algorithm and which implements its individual steps.
struct LM<'a, F, N, M, O>
where
    F: RealField + Copy,
    N: Dim,
    M: DimMin<N> + DimMax<N>,
    O: LeastSquaresProblem<F, M, N>,
    DefaultAllocator: Allocator<F, N> + Allocator<F, DimMaximum<M, N>, N>,
{
    config: &'a LevenbergMarquardt<F>,
    /// Current parameters `$\vec{x}$`
    x: Vector<F, N, O::ParameterStorage>,
    tmp: Vector<F, N, O::ParameterStorage>,
    /// The implementation of `LeastSquaresProblem`
    target: O,
    /// Statistics and termination reasons, used for return value
    report: MinimizationReport<F>,
    /// The delta from the trust-region algorithm
    delta: F,
    lambda: F,
    /// `$\|\mathbf{D}\vec{x}\|`
    xnorm: F,
    gnorm: F,
    residuals_norm: F,
    /// The diagonal of `$\mathbf{D}$`
    diag: OVector<F, N>,
    /// Flag to check if it is the first trust region iteration
    first_trust_region_iteration: bool,
    /// Flag to check if it is the first diagonal update
    first_update: bool,
    max_fev: usize,
    m: usize,
}

impl<'a, F, N, M, O> LM<'a, F, N, M, O>
where
    F: RealField + Float + Copy,
    N: Dim,
    M: DimMin<N> + DimMax<N>,
    O: LeastSquaresProblem<F, M, N>,
    DefaultAllocator: Allocator<F, N> + Allocator<F, DimMaximum<M, N>, N>,
{
    #[allow(clippy::type_complexity)]
    fn new(
        config: &'a LevenbergMarquardt<F>,
        target: O,
    ) -> Result<(Self, Vector<F, M, O::ResidualStorage>), (O, MinimizationReport<F>)> {
        let mut report = MinimizationReport {
            termination: TerminationReason::ResidualsZero,
            number_of_evaluations: 1,
            objective_function: <F as Float>::nan(),
        };

        // Evaluate at start point
        let x = target.params();
        let (residuals, residuals_norm) = if let Some(residuals) = target.residuals() {
            let norm = enorm(&residuals);
            report.objective_function = norm * norm * convert(0.5);
            (residuals, norm)
        } else {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::User("residuals"),
                    ..report
                },
            ));
        };

        // Initialize diagonal
        let n = x.shape_generic().0;
        let diag = OVector::<F, N>::from_element_generic(n, Dim::from_usize(1), F::one());
        // Check n > 0
        if diag.nrows() == 0 {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::NoParameters,
                    ..report
                },
            ));
        }

        let m = residuals.nrows();
        if m == 0 {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::NoResiduals,
                    ..report
                },
            ));
        }

        if !residuals_norm.is_finite() && !cfg!(feature = "minpack-compat") {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::Numerical("residuals norm"),
                    ..report
                },
            ));
        }

        if residuals_norm <= Float::min_positive_value() && !cfg!(feature = "minpack-compat") {
            // Already zero, nothing to do
            return Err((target, report));
        }

        Ok((
            Self {
                config,
                target,
                report,
                tmp: x.clone(),
                x,
                diag,
                delta: F::zero(),
                lambda: F::zero(),
                xnorm: F::zero(),
                gnorm: F::zero(),
                residuals_norm,
                first_trust_region_iteration: true,
                first_update: true,
                max_fev: config.patience * (n.value() + 1),
                m,
            },
            residuals,
        ))
    }

    fn into_report(self, termination: TerminationReason) -> (O, MinimizationReport<F>) {
        (
            self.target,
            MinimizationReport {
                termination,
                ..self.report
            },
        )
    }

    fn jacobian(&self) -> Result<Matrix<F, M, N, O::JacobianStorage>, TerminationReason> {
        match self.target.jacobian() {
            Some(jacobian) => Ok(jacobian),
            None => Err(TerminationReason::User("jacobian")),
        }
    }

    fn update_diag(
        &mut self,
        lls: &mut LinearLeastSquaresDiagonalProblem<F, M, N>,
    ) -> Result<(), TerminationReason>
    where
        DefaultAllocator: Allocator<usize, N>,
    {
        // Compute norm of scaled gradient and detect degeneracy
        self.gnorm = match lls.max_a_t_b_scaled(self.residuals_norm) {
            Some(max_at_b) => max_at_b,
            None if !cfg!(feature = "minpack-compat") => {
                return Err(TerminationReason::Numerical("jacobian"))
            }
            None => F::zero(),
        };
        if self.gnorm <= self.config.gtol {
            return Err(TerminationReason::Orthogonal);
        }

        if self.first_update {
            // Initialize diag and xnorm
            self.xnorm = if self.config.scale_diag {
                for (d, col_norm) in self.diag.iter_mut().zip(lls.column_norms.iter()) {
                    *d = if col_norm.is_zero() {
                        F::one()
                    } else {
                        *col_norm
                    };
                }
                self.tmp.cmpy(F::one(), &self.diag, &self.x, F::zero());
                enorm(&self.tmp)
            } else {
                enorm(&self.x)
            };
            if !self.xnorm.is_finite() && !cfg!(feature = "minpack-compat") {
                return Err(TerminationReason::Numerical("subproblem x"));
            }
            // Initialize delta
            self.delta = if self.xnorm.is_zero() {
                self.config.stepbound
            } else {
                self.config.stepbound * self.xnorm
            };
            self.first_update = false;
        } else if self.config.scale_diag {
            // Update diag
            for (d, norm) in self.diag.iter_mut().zip(lls.column_norms.iter()) {
                *d = Float::max(*norm, *d);
            }
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn trust_region_iteration(
        &mut self,
        lls: &mut LinearLeastSquaresDiagonalProblem<F, M, N>,
        param: LMParameter<F, N>,
    ) -> Result<Option<Vector<F, M, O::ResidualStorage>>, TerminationReason>
    where
        DefaultAllocator: Allocator<usize, N>,
    {
        const P1: f64 = 0.1;
        const P0001: f64 = 1.0e-4;

        self.lambda = param.lambda;
        let pnorm = param.dp_norm;
        if !pnorm.is_finite() && !cfg!(feature = "minpack-compat") {
            return Err(TerminationReason::Numerical("subproblem ||Dp||"));
        }

        let predicted_reduction;
        let dir_der;
        {
            let temp1 = Float::powi(lls.a_x_norm(&param.step) / self.residuals_norm, 2);
            if !temp1.is_finite() && !cfg!(feature = "minpack-compat") {
                return Err(TerminationReason::Numerical("trust-region reduction"));
            }
            let temp2 = Float::powi((Float::sqrt(self.lambda) * pnorm) / self.residuals_norm, 2);
            if !temp2.is_finite() && !cfg!(feature = "minpack-compat") {
                return Err(TerminationReason::Numerical("trust-region reduction"));
            }
            predicted_reduction = temp1 + temp2 / convert(0.5);
            dir_der = -(temp1 + temp2);
        }

        if self.first_trust_region_iteration && pnorm < self.delta {
            self.delta = pnorm;
        }
        self.first_trust_region_iteration = false;

        // Compute new parameters: x - p
        self.tmp.copy_from(&self.x);
        self.tmp.axpy(-F::one(), &param.step, F::one());

        // Evaluate
        self.target.set_params(&self.tmp);
        self.report.number_of_evaluations += 1;
        let new_objective_function;
        let (residuals, new_residuals_norm) = if let Some(residuals) = self.target.residuals() {
            if residuals.nrows() != self.m {
                return Err(TerminationReason::WrongDimensions("residuals"));
            }
            let norm = enorm(&residuals);
            new_objective_function = norm * norm * convert(0.5);
            (residuals, norm)
        } else {
            return Err(TerminationReason::User("residuals"));
        };

        // Compute predicted and actual reduction
        let actual_reduction = if new_residuals_norm * convert(P1) < self.residuals_norm {
            F::one() - Float::powi(new_residuals_norm / self.residuals_norm, 2)
        } else {
            -F::one()
        };

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
                half * dir_der / (dir_der + half * actual_reduction)
            };
            if new_residuals_norm * convert(P1) >= self.residuals_norm || temp < convert(P1) {
                temp = convert(P1);
            };
            self.delta = temp * Float::min(self.delta, pnorm * convert(10.));
            self.lambda /= temp;
        } else if self.lambda.is_zero() || ratio >= convert(0.75) {
            self.delta = pnorm / convert(0.5);
            self.lambda *= half;
        }

        let update_considered_good = ratio >= convert(P0001);
        if update_considered_good {
            // update x, residuals and their norms
            core::mem::swap(&mut self.x, &mut self.tmp);
            self.xnorm = if self.config.scale_diag {
                self.tmp.cmpy(F::one(), &self.diag, &self.x, F::zero());
                enorm(&self.tmp)
            } else {
                enorm(&self.x)
            };
            if !self.xnorm.is_finite() && !cfg!(feature = "minpack-compat") {
                return Err(TerminationReason::Numerical("new x"));
            }
            self.residuals_norm = new_residuals_norm;
            self.report.objective_function = new_objective_function;
        }

        // convergence tests
        if !cfg!(feature = "minpack-compat") && self.residuals_norm <= F::min_positive_value() {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::ResidualsZero);
        }
        let ftol_check = Float::abs(actual_reduction) <= self.config.ftol
            && predicted_reduction <= self.config.ftol
            && ratio * convert(0.5) <= F::one();
        let xtol_check = self.delta <= self.config.xtol * self.xnorm;
        if ftol_check || xtol_check {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::Converged {
                ftol: ftol_check,
                xtol: xtol_check,
            });
        }

        // termination tests
        if self.report.number_of_evaluations >= self.max_fev {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::LostPatience);
        }

        // We now check if one of the ftol, xtol or gtol criteria
        // is fulfilld with the machine epsilon.
        if Float::abs(actual_reduction) <= epsmch()
            && predicted_reduction <= epsmch()
            && ratio * convert(0.5) <= F::one()
        {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("ftol"));
        }
        if self.delta <= epsmch::<F>() * self.xnorm {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("xtol"));
        }
        if self.gnorm <= epsmch() {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("gtol"));
        }

        if update_considered_good {
            Ok(Some(residuals))
        } else {
            // Need another iteration, did not change the parameters
            Ok(None)
        }
    }

    #[inline]
    fn reset_params_if(&mut self, reset: bool) {
        if reset {
            self.target.set_params(&self.x);
        }
    }
}
