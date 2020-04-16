use approx::assert_relative_eq;
use nalgebra::storage::Owned;
use nalgebra::*;

use crate::utils::differentiate_numerically;
use crate::{LeastSquaresProblem, LevenbergMarquardt};

pub struct LinearFullRank {
    pub params: VectorN<f64, U5>,
    pub m: usize,
}

/// TOL value used by SciPy
const TOL: f64 = 1.49012e-08;

impl LeastSquaresProblem<f64, U5, Dynamic> for LinearFullRank {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, U5>;

    fn set_params(&mut self, params: &VectorN<f64, U5>) {
        self.params.copy_from(params);
    }

    fn residuals(&self) -> Option<VectorN<f64, Dynamic>> {
        let m = Dynamic::from_usize(self.m);
        let mut residuals = VectorN::<f64, Dynamic>::from_element_generic(
            m,
            U1,
            -2. * self.params.sum() / self.m as f64 - 1.,
        );
        for (el, p) in residuals
            .rows_range_mut(..5)
            .iter_mut()
            .zip(self.params.iter())
        {
            *el += p;
        }
        Some(residuals)
    }

    fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, U5>> {
        let m = Dynamic::from_usize(self.m);
        let mut jacobian = MatrixMN::from_element_generic(m, U5, -2. / self.m as f64);
        for i in 0..5 {
            jacobian[(i, i)] += 1.;
        }
        Some(jacobian)
    }
}

fn setup_linear_full_rank(m: usize, factor: f64) -> (VectorN<f64, U5>, LinearFullRank) {
    let guess = VectorN::<f64, U5>::from_element(factor);
    let mut problem = LinearFullRank {
        params: VectorN::<f64, U5>::zeros(),
        m,
    };
    let jac_num = differentiate_numerically(guess, &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-7);
    (guess, problem)
}

#[test]
fn test_linear_full_rank() {
    let (initial, problem) = setup_linear_full_rank(10, 1.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert!(report.termination.was_successful());
    assert_relative_eq!(report.objective_function, 2.5, epsilon = 1e-14);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U5>::from_element(-1.),
        epsilon = 1e-14
    );

    let (initial, problem) = setup_linear_full_rank(50, 1.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert!(report.termination.was_successful());
    assert_relative_eq!(report.objective_function, 22.5, epsilon = 1e-14);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U5>::from_element(-1.),
        epsilon = 1e-14
    );
}

struct LinearRank1 {
    params: VectorN<f64, U5>,
    m: usize,
}

impl LeastSquaresProblem<f64, U5, Dynamic> for LinearRank1 {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, U5>;

    fn set_params(&mut self, params: &VectorN<f64, U5>) {
        self.params.copy_from(params);
    }

    fn residuals(&self) -> Option<VectorN<f64, Dynamic>> {
        let m = Dynamic::from_usize(self.m);
        let weighted_sum: f64 = self
            .params
            .iter()
            .enumerate()
            .map(|(j, p)| (j + 1) as f64 * p)
            .sum();
        Some(VectorN::<f64, Dynamic>::from_iterator_generic(
            m,
            U1,
            (0..self.m).map(|i| (i + 1) as f64 * weighted_sum - 1.),
        ))
    }

    fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, U5>> {
        let m = Dynamic::from_usize(self.m);
        Some(MatrixMN::from_fn_generic(m, U5, |i, j| {
            ((i + 1) * (j + 1)) as f64
        }))
    }
}

fn setup_linear_rank1(m: usize, factor: f64) -> (VectorN<f64, U5>, LinearRank1) {
    let guess = VectorN::<f64, U5>::from_element(factor);
    let mut problem = LinearRank1 {
        params: VectorN::<f64, U5>::zeros(),
        m,
    };
    let jac_num = differentiate_numerically(guess, &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-6);
    (guess, problem)
}

#[test]
fn test_linear_rank1() {
    let (initial, mut problem) = setup_linear_rank1(10, 1.);
    problem.set_params(&initial);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert!(report.termination.was_successful());
    assert_relative_eq!(report.objective_function, 1.0714285714285714);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U5>::from_column_slice(&[
            -167.79681802396928,
            -83.39840901198468,
            221.11004307957813,
            -41.19920450599233,
            -32.759363604793855,
        ])
    );

    let (initial, problem) = setup_linear_rank1(50, 1.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert!(report.termination.was_successful());
    assert_relative_eq!(report.objective_function, 6.064356435643564);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U5>::from_column_slice(&[
            -20.29999900022674,
            -9.64999950011337,
            -165.2451975264496,
            -4.324999750056676,
            110.53305851006517
        ])
    );
}

struct LinearRank1ZeroColumns {
    params: VectorN<f64, U5>,
    m: usize,
}

impl LeastSquaresProblem<f64, U5, Dynamic> for LinearRank1ZeroColumns {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, U5>;

    fn set_params(&mut self, params: &VectorN<f64, U5>) {
        self.params.copy_from(params);
    }

    fn residuals(&self) -> Option<VectorN<f64, Dynamic>> {
        let m = Dynamic::from_usize(self.m);
        let weighted_sum: f64 = self
            .params
            .iter()
            .enumerate()
            .map(|(j, p)| {
                if j == 0 || j == 5 - 1 {
                    0.
                } else {
                    (j + 1) as f64 * p
                }
            })
            .sum();
        Some(VectorN::<f64, Dynamic>::from_iterator_generic(
            m,
            U1,
            (0..self.m).map(|i| {
                if i == self.m - 1 {
                    -1.
                } else {
                    i as f64 * weighted_sum - 1.
                }
            }),
        ))
    }

    fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, U5>> {
        let m = Dynamic::from_usize(self.m);
        Some(MatrixMN::from_fn_generic(m, U5, |i, j| {
            if i >= 1 && j >= 1 && j < 5 - 1 && i < self.m - 1 {
                ((j + 1) * i) as f64
            } else {
                0.
            }
        }))
    }
}

fn setup_linear_rank1_zero(m: usize, factor: f64) -> (VectorN<f64, U5>, LinearRank1ZeroColumns) {
    let guess = VectorN::<f64, U5>::from_element(factor);
    let mut problem = LinearRank1ZeroColumns {
        params: VectorN::<f64, U5>::zeros(),
        m,
    };
    let jac_num = differentiate_numerically(guess, &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-5);
    (guess, problem)
}

#[test]
fn test_linear_rank1_zero() {
    let (initial, problem) = setup_linear_rank1_zero(10, 1.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert!(report.termination.was_successful());
    assert_relative_eq!(
        report.objective_function,
        1.8235294117647058,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U5>::from_column_slice(&[
            1.,
            -210.3615324224772,
            32.120420811321296,
            81.13456824980642,
            1.
        ])
    );

    let (initial, problem) = setup_linear_rank1_zero(50, 1.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert_relative_eq!(report.objective_function, 6.814432989690721);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U5>::from_column_slice(&[
            1.,
            332.1494858957815,
            -439.6851914289522,
            163.69688258258626,
            1.,
        ])
    );
}
