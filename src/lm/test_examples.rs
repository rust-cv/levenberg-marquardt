//! Tests with example functions.
//!
//! There is also a Python implementation in `test_examples.py`.
//! This was used to get the output from MINPACK.
use approx::assert_relative_eq;
use nalgebra::storage::Owned;
use nalgebra::*;

use crate::utils::differentiate_numerically;
use crate::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};

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
    let jac_num =
        differentiate_numerically(VectorN::<f64, U5>::new_random(), &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-12);
    (guess, problem)
}

#[test]
fn test_linear_full_rank() {
    let (initial, problem) = setup_linear_full_rank(10, 1.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: true
        }
    );
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
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: true
        }
    );
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
    let jac_num =
        differentiate_numerically(VectorN::<f64, U5>::new_random(), &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-12);
    (guess, problem)
}

#[test]
fn test_linear_rank1() {
    let (initial, mut problem) = setup_linear_rank1(10, 1.);
    problem.set_params(&initial);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: false
        }
    );
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
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: false
        }
    );
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
    let jac_num =
        differentiate_numerically(VectorN::<f64, U5>::new_random(), &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-12);
    (guess, problem)
}

#[test]
fn test_linear_rank1_zero() {
    let (initial, problem) = setup_linear_rank1_zero(10, 1.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial, problem);
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: false
        }
    );
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
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: false
        }
    );
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

#[test]
fn test_rosenbruck() {
    #[derive(Clone)]
    struct Rosenbruck {
        params: VectorN<f64, U2>,
    }
    impl LeastSquaresProblem<f64, U2, U2> for Rosenbruck {
        type ParameterStorage = Owned<f64, U2>;
        type ResidualStorage = Owned<f64, U2>;
        type JacobianStorage = Owned<f64, U2, U2>;

        fn set_params(&mut self, params: &VectorN<f64, U2>) {
            self.params.copy_from(params);
        }

        fn residuals(&self) -> Option<VectorN<f64, U2>> {
            Some(Vector2::new(
                10. * (self.params[1] - self.params[0] * self.params[0]),
                1. - self.params[0],
            ))
        }

        fn jacobian(&self) -> Option<MatrixMN<f64, U2, U2>> {
            Some(Matrix2::new(-20. * self.params[0], 10., -1., 0.))
        }
    }
    let initial = Vector2::<f64>::new(-1.2, 1.);
    let mut problem = Rosenbruck {
        params: Vector2::<f64>::zeros(),
    };
    let jac_num = differentiate_numerically(Vector2::<f64>::new_random(), &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-12);

    let guess = initial.clone();
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(guess, problem.clone());
    if cfg!(feature = "minpack-compat") {
        // MINPACK gives "Orthogonal" but this is because
        // residual is identical zero which gives NaN in the `gnorm`
        // computation. Long story short, this will make the gnorm
        // termination test pass in MINPACK.
        assert_eq!(report.termination, TerminationReason::Orthogonal);
    } else {
        assert_eq!(report.termination, TerminationReason::ResidualsZero);
    }
    assert_relative_eq!(report.objective_function, 0.);
    assert_relative_eq!(problem.params, Vector2::<f64>::from_element(1.));

    let guess = initial.map(|x| x + 10.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(guess, problem.clone());
    if cfg!(feature = "minpack-compat") {
        assert_eq!(
            report.termination,
            TerminationReason::Converged {
                ftol: false,
                xtol: true
            }
        );
    } else {
        assert_eq!(report.termination, TerminationReason::ResidualsZero);
    }
    assert_relative_eq!(report.objective_function, 0.);
    assert_relative_eq!(problem.params, Vector2::<f64>::from_element(1.));

    let guess = initial.map(|x| x + 100.);
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(guess, problem.clone());
    if cfg!(feature = "minpack-compat") {
        assert_eq!(
            report.termination,
            TerminationReason::Converged {
                ftol: false,
                xtol: true
            }
        );
    } else {
        assert_eq!(report.termination, TerminationReason::ResidualsZero);
    }
    assert_relative_eq!(report.objective_function, 0.);
    assert_relative_eq!(problem.params, Vector2::<f64>::from_element(1.));
}

#[test]
fn test_helical_valley() {
    const TPI: f64 = ::core::f64::consts::PI * 2.;
    #[derive(Clone)]
    struct HelicalValley {
        params: VectorN<f64, U3>,
    }
    impl LeastSquaresProblem<f64, U3, U3> for HelicalValley {
        type ParameterStorage = Owned<f64, U3>;
        type ResidualStorage = Owned<f64, U3>;
        type JacobianStorage = Owned<f64, U3, U3>;

        fn set_params(&mut self, params: &VectorN<f64, U3>) {
            self.params.copy_from(params);
        }

        fn residuals(&self) -> Option<VectorN<f64, U3>> {
            let p = self.params;
            let tmp1 = if p[0] == 0. {
                (0.25f64).copysign(p[1])
            } else if p[0] > 0. {
                (p[1] / p[0]).atan() / TPI
            } else {
                (p[1] / p[0]).atan() / TPI + 0.5
            };
            let tmp2 = (p[0] * p[0] + p[1] * p[1]).sqrt();
            Some(Vector3::new(
                10. * (p[2] - 10. * tmp1),
                10. * (tmp2 - 1.),
                p[2],
            ))
        }

        #[rustfmt::skip]
        fn jacobian(&self) -> Option<MatrixMN<f64, U3, U3>> {
            let p = self.params;
            let temp = p[0] * p[0] + p[1] * p[1];
            let tmp1 = TPI * temp;
            let tmp2 = temp.sqrt();
            Some(Matrix3::new(
                100. * p[1] / tmp1, -100. * p[0] / tmp1, 10.,
                 10. * p[0] / tmp2,   10. * p[1] / tmp2,  0.,
                                0.,                  0.,  1.,
            ))
        }
    }
    let initial = Vector3::<f64>::new(-1., 0., 0.);
    let mut problem = HelicalValley {
        params: Vector3::<f64>::zeros(),
    };
    let jac_num = differentiate_numerically(Vector3::<f64>::new_random(), &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-8);

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: false,
            xtol: true
        }
    );
    assert_relative_eq!(report.objective_function, 4.936724569245567e-33);
    assert_relative_eq!(
        problem.params,
        Vector3::<f64>::new(1., -6.243301596789443e-18, 0.)
    );

    // MINPACK gives xtol, but have an exta residuals check in our code
    let termination_reason = if cfg!(feature = "minpack-compat") {
        TerminationReason::Converged {
            ftol: false,
            xtol: true,
        }
    } else {
        TerminationReason::ResidualsZero
    };

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x * 10.), problem.clone());
    assert_eq!(report.termination, termination_reason);
    assert_relative_eq!(report.objective_function, 5.456769505027268e-39);
    assert_relative_eq!(
        problem.params,
        Vector3::<f64>::new(1., 6.563910805155555e-21, 0.)
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x * 100.), problem.clone());
    assert_eq!(report.termination, termination_reason);
    assert_relative_eq!(report.objective_function, 4.9259630763847064e-58);
    assert_relative_eq!(
        problem.params,
        Vector3::<f64>::new(1., -1.9721522630525295e-30, 0.)
    );
}

#[test]
fn test_powell_singular() {
    #[derive(Clone)]
    struct PowellSingular {
        params: VectorN<f64, U4>,
    }
    impl LeastSquaresProblem<f64, U4, U4> for PowellSingular {
        type ParameterStorage = Owned<f64, U4>;
        type ResidualStorage = Owned<f64, U4>;
        type JacobianStorage = Owned<f64, U4, U4>;

        fn set_params(&mut self, params: &VectorN<f64, U4>) {
            self.params.copy_from(params);
        }

        fn residuals(&self) -> Option<VectorN<f64, U4>> {
            let p = self.params;
            Some(Vector4::new(
                p[0] + 10. * p[1],
                (5.).sqrt() * (p[2] - p[3]),
                (p[1] - 2. * p[2]).powi(2),
                (10.).sqrt() * (p[0] - p[3]).powi(2),
            ))
        }

        #[rustfmt::skip]
        fn jacobian(&self) -> Option<MatrixMN<f64, U4, U4>> {
            let p = self.params;
            let f = (5.).sqrt();
            let t = (10.).sqrt();
            let tmp1 = p[1] - 2. * p[2];
            let tmp2 = p[0] - p[3];
            Some(Matrix4::new(
                           1.,       10.,         0.,             0.,
                           0.,        0.,          f,             -f,
                           0., 2. * tmp1, -4. * tmp1,             0.,
                2. * t * tmp2,        0.,         0., -2. * t * tmp2,
            ))
        }
    }
    let mut problem = PowellSingular {
        params: Vector4::<f64>::zeros(),
    };
    let jac_num = differentiate_numerically(Vector4::<f64>::new_random(), &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-9);

    let initial = Vector4::<f64>::new(3., -1., 0., 1.);

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(
        report.termination,
        TerminationReason::NoImprovementPossible("gtol")
    );
    assert_relative_eq!(report.objective_function, 1.1986096828735036e-66);
    assert_relative_eq!(
        problem.params,
        Vector4::<f64>::new(
            1.6521175961683935e-17,
            -1.6521175961683934e-18,
            2.6433881538694683e-18,
            2.6433881538694683e-18
        )
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x * 10.), problem.clone());
    assert_eq!(
        report.termination,
        TerminationReason::NoImprovementPossible("gtol")
    );
    assert_relative_eq!(report.objective_function, 1.5185402226754922e-88);
    assert_relative_eq!(
        problem.params,
        Vector4::<f64>::new(
            1.0890568734571656e-22,
            -1.0890568734571657e-23,
            3.8699037166581656e-23,
            3.8699037166581656e-23
        )
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x * 10.), problem.clone());
    assert_eq!(
        report.termination,
        TerminationReason::NoImprovementPossible("gtol")
    );
    assert_relative_eq!(report.objective_function, 1.856300369212571e-68);
    assert_relative_eq!(
        problem.params,
        Vector4::<f64>::new(
            3.2267921800163781e-18,
            -3.2267921800163780e-19,
            5.1628674880262125e-19,
            5.1628674880262125e-19
        )
    );
}
