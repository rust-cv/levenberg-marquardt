use crate::qr::{LinearLeastSquaresDiagonalProblem, PivotedQR};
use crate::trust_region::{determine_lambda_and_parameter_update, LMParameter};
use crate::LeastSquaresProblem;
use nalgebra::{
    allocator::Allocator,
    constraint::{DimEq, ShapeConstraint},
    convert,
    storage::Storage,
    DefaultAllocator, DimMin, DimMinimum, Matrix, RealField, Vector, VectorN, U1,
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
        let (mut lm, mut residuals) = match LM::new(self, initial_x, target) {
            Err(report) => return report,
            Ok(res) => res,
        };
        loop {
            // Build LLS
            let mut lls = {
                let jacobian = match lm.jacobian() {
                    Err(reason) => return lm.into_report(reason),
                    Ok(jacobian) => jacobian,
                };
                let qr = PivotedQR::new(jacobian).ok().unwrap();
                qr.into_least_squares_diagonal_problem(residuals)
            };

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

/// Struct which holds the state of the LM algorithm and implements individual steps.
struct LM<'a, F, N, M, O>
where
    F: RealField,
    N: DimMin<M>,
    M: DimMin<N>,
    O: LeastSquaresProblem<F, N, M>,
    DefaultAllocator: Allocator<F, N>,
{
    config: &'a LevenbergMarquardt<F>,
    /// Current parameters
    x: Vector<F, N, O::ParameterStorage>,
    tmp: Vector<F, N, O::ParameterStorage>,
    target: O,
    report: MinimizationReport<F>,
    delta: F,
    lambda: F,
    xnorm: F,
    residuals_norm: F,
    diag: VectorN<F, N>,
    first_trust_region_iteration: bool,
    first_update: bool,
}

impl<'a, F, N, M, O> LM<'a, F, N, M, O>
where
    F: RealField + Float,
    N: DimMin<M>,
    M: DimMin<N>,
    O: LeastSquaresProblem<F, N, M>,
    DefaultAllocator: Allocator<F, N>,
{
    #[allow(clippy::type_complexity)]
    fn new(
        config: &'a LevenbergMarquardt<F>,
        initial_x: Vector<F, N, O::ParameterStorage>,
        mut target: O,
    ) -> Result<(Self, Vector<F, M, O::ResidualStorage>), (O, MinimizationReport<F>)> {
        let mut report = MinimizationReport {
            failure: None,
            number_of_evaluations: 1,
            objective_function: <F as Float>::nan(),
        };

        // Evaluate at start point
        let x = initial_x;
        target.set_params(&x);
        let (residuals, residuals_norm) = if let Some(residuals) = target.residuals() {
            let norm_squared = residuals.norm_squared();
            report.objective_function = norm_squared * convert(0.5);
            (residuals, Float::sqrt(norm_squared))
        } else {
            return Err((
                target,
                MinimizationReport {
                    failure: Some(Failure::User),
                    ..report
                },
            ));
        };

        // Initialize diagonal
        let n = x.data.shape().0;
        let diag = VectorN::<F, N>::from_element_generic(n, U1, F::one());
        // Check n > 0
        if diag.nrows() == 0 {
            return Err((
                target,
                MinimizationReport {
                    failure: Some(Failure::NoParameters),
                    ..report
                },
            ));
        }

        // Check m >= n
        if diag.nrows() > residuals.nrows() {
            return Err((
                target,
                MinimizationReport {
                    failure: Some(Failure::NotEnoughResiduals),
                    ..report
                },
            ));
        }

        if !residuals_norm.is_finite() {
            return Err((
                target,
                MinimizationReport {
                    failure: Some(Failure::Numerical),
                    ..report
                },
            ));
        }

        if residuals_norm <= Float::min_positive_value() {
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
                residuals_norm,
                first_trust_region_iteration: true,
                first_update: true,
            },
            residuals,
        ))
    }

    fn into_report(self, failure: Option<Failure>) -> (O, MinimizationReport<F>) {
        (
            self.target,
            MinimizationReport {
                failure,
                ..self.report
            },
        )
    }

    fn jacobian(&self) -> Result<Matrix<F, M, N, O::JacobianStorage>, Option<Failure>> {
        match self.target.jacobian() {
            Some(jacobian) => Ok(jacobian),
            None => Err(Some(Failure::User)),
        }
    }

    fn update_diag(
        &mut self,
        lls: &mut LinearLeastSquaresDiagonalProblem<F, M, N, O::JacobianStorage>,
    ) -> Result<(), Option<Failure>>
    where
        DefaultAllocator: Allocator<usize, N>,
    {
        // Compute norm of scaled gradient and detect degeneracy
        let gnorm = lls.max_a_t_b_scaled() / self.residuals_norm;
        if gnorm <= self.config.gtol {
            return Err(None);
        }

        if self.first_update {
            // Initialize diag and delta
            self.xnorm = if self.config.scale_diag {
                for (d, col_norm) in self.diag.iter_mut().zip(lls.column_norms.iter()) {
                    *d = if col_norm.is_zero() {
                        F::one()
                    } else {
                        *col_norm
                    };
                }
                self.tmp.cmpy(F::one(), &self.diag, &self.x, F::zero());
                self.tmp.norm()
            } else {
                self.x.norm()
            };
            if !self.xnorm.is_finite() {
                return Err(Some(Failure::Numerical));
            }
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
        lls: &mut LinearLeastSquaresDiagonalProblem<F, M, N, O::JacobianStorage>,
        param: LMParameter<F, N>,
    ) -> Result<Option<Vector<F, M, O::ResidualStorage>>, Option<Failure>>
    where
        DefaultAllocator: Allocator<usize, N>,
    {
        const P1: f64 = 0.1;
        const P0001: f64 = 1.0e-4;

        self.lambda = param.lambda;
        let pnorm = param.dp_norm;
        if !pnorm.is_finite() {
            return Err(Some(Failure::Numerical));
        }

        if self.first_trust_region_iteration && pnorm < self.delta {
            self.first_trust_region_iteration = false;
            self.delta = pnorm;
        }

        // These values are needed later. We check now to fail early.
        let temp2 = self.lambda * Float::powi(pnorm / self.residuals_norm, 2);
        if !temp2.is_finite() {
            return Err(Some(Failure::Numerical));
        }
        let temp1 = lls.a_x_norm_squared(&param.step) / Float::powi(self.residuals_norm, 2);
        if !temp1.is_finite() {
            return Err(Some(Failure::Numerical));
        }

        // Compute new parameters: x - p
        self.tmp.copy_from(&self.x);
        self.tmp.axpy(-F::one(), &param.step, F::one());
        // Evaluate
        self.target.set_params(&self.tmp);
        let new_objective_function;
        self.report.number_of_evaluations += 1;
        let (residuals, new_residuals_norm) = if let Some(residuals) = self.target.residuals() {
            let norm_squared = residuals.norm_squared();
            new_objective_function = norm_squared * convert(0.5);
            (residuals, Float::sqrt(norm_squared))
        } else {
            return Err(Some(Failure::User));
        };

        // Compute predicted and actual reduction
        let actual_reduction = if new_residuals_norm * convert(P1) < self.residuals_norm {
            F::one() - Float::powi(new_residuals_norm / self.residuals_norm, 2)
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
            if new_residuals_norm * convert(P1) >= self.residuals_norm || temp < convert(P1) {
                temp = convert(P1);
            };
            self.delta = temp * Float::min(self.delta, pnorm / convert(P1));
            self.lambda /= temp;
        } else if self.lambda.is_zero() || ratio >= convert(0.75) {
            self.delta = pnorm * convert(2.);
            self.lambda *= half;
        }

        let inner_success = ratio >= convert(P0001);
        // on sucess, update x, residuals and their norms
        if inner_success {
            core::mem::swap(&mut self.x, &mut self.tmp);
            self.xnorm = if self.config.scale_diag {
                self.tmp.cmpy(F::one(), &self.diag, &self.x, F::zero());
                self.tmp.norm()
            } else {
                self.x.norm()
            };
            if !self.xnorm.is_finite() {
                return Err(Some(Failure::Numerical));
            }
            self.residuals_norm = new_residuals_norm;
            self.report.objective_function = new_objective_function;
        }

        // convergence tests
        if self.residuals_norm <= F::min_positive_value()
            || (Float::abs(actual_reduction) <= self.config.ftol
                && predicted_reduction <= self.config.ftol
                && ratio <= convert(2.))
            || self.delta <= self.config.xtol * self.xnorm
        {
            return Err(None);
        }

        // termination tests
        if self.report.number_of_evaluations >= self.config.patience {
            return Err(Some(Failure::LostPatience));
        }
        if (Float::abs(actual_reduction) <= F::default_epsilon()
            && predicted_reduction <= F::default_epsilon()
            && ratio <= convert(2.))
            || self.delta <= F::default_epsilon() * self.xnorm
            || pnorm <= F::default_epsilon()
        {
            return Err(Some(Failure::NoImprovementPossible));
        }

        if inner_success {
            Ok(Some(residuals))
        } else {
            Ok(None)
        }
    }
}
