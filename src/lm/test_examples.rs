//! Tests with example functions.
//!
//! There is also a Python implementation in `test_examples.py`
//! which was used to get the output from MINPACK.
use approx::assert_relative_eq;
use nalgebra::*;
use nalgebra::{allocator::Allocator, storage::Owned};

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
    assert_relative_eq!(report.objective_function, 2.5000000000000004);
    assert_relative_eq!(
        problem.params,
        Vector5::<f64>::new(-1., -1.0000000000000004, -1., -1.0000000000000004, -1.)
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
    assert_relative_eq!(report.objective_function, 22.500000000000004);
    assert_relative_eq!(
        problem.params,
        Vector5::<f64>::new(
            -0.9999999999999953,
            -1.0000000000000049,
            -0.9999999999999976,
            -0.9999999999999956,
            -0.9999999999999991
        )
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
    assert_relative_eq!(report.objective_function, 6.064356435643563);
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
    assert_relative_eq!(report.objective_function, 1.8235294117647063,);
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
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-10);

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
    assert_relative_eq!(report.objective_function, 1.866194344564614e-67);
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
    assert_relative_eq!(report.objective_function, 1.518540222675492e-88);
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
    assert_relative_eq!(report.objective_function, 2.715670190176167e-70);
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

#[test]
fn test_freudenstein_roth() {
    #[derive(Clone)]
    struct FreudensteinRoth {
        params: VectorN<f64, U2>,
    }
    impl LeastSquaresProblem<f64, U2, U2> for FreudensteinRoth {
        type ParameterStorage = Owned<f64, U2>;
        type ResidualStorage = Owned<f64, U2>;
        type JacobianStorage = Owned<f64, U2, U2>;

        fn set_params(&mut self, params: &VectorN<f64, U2>) {
            self.params.copy_from(params);
        }

        #[rustfmt::skip]
        fn residuals(&self) -> Option<VectorN<f64, U2>> {
            let p = &self.params;
            Some(Vector2::new(
                -13. + p[0] + ((5. - p[1]) * p[1] - 2.) * p[1],
                -29. + p[0] + ((1. + p[1]) * p[1] - 14.) * p[1]
            ))
        }

        #[rustfmt::skip]
        fn jacobian(&self) -> Option<MatrixMN<f64, U2, U2>> {
            let p = &self.params;
            Some(Matrix2::new(
                1. , p[1] * (10. - 3. * p[1]) - 2.,
                1.,  p[1] * (2. + 3. * p[1]) - 14.,
            ))
        }
    }
    let initial = Vector2::<f64>::new(0.5, -2.);
    let mut problem = FreudensteinRoth {
        params: Vector2::<f64>::zeros(),
    };
    let jac_num = differentiate_numerically(Vector2::<f64>::new_random(), &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-9);

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: false
        }
    );
    assert_relative_eq!(report.objective_function, 24.492126863534953);
    assert_relative_eq!(
        problem.params,
        Vector2::<f64>::new(11.412484465499368, -0.8968279137315035)
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x * 10.), problem.clone());
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: false
        }
    );
    assert_relative_eq!(report.objective_function, 24.492126854042752);
    assert_relative_eq!(
        problem.params,
        Vector2::<f64>::new(11.413004661474561, -0.8967960386859591)
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x * 100.), problem.clone());
    assert_eq!(
        report.termination,
        TerminationReason::Converged {
            ftol: true,
            xtol: false
        }
    );
    assert_relative_eq!(report.objective_function, 24.49212683962172);
    assert_relative_eq!(
        problem.params,
        Vector2::<f64>::new(11.412781785788198, -0.8968051074920677)
    );
}

#[test]
fn test_bard() {
    #[derive(Clone)]
    struct Bard {
        params: VectorN<f64, U3>,
    }
    impl LeastSquaresProblem<f64, U3, U15> for Bard {
        type ParameterStorage = Owned<f64, U3>;
        type ResidualStorage = Owned<f64, U15>;
        type JacobianStorage = Owned<f64, U15, U3>;

        fn set_params(&mut self, params: &VectorN<f64, U3>) {
            self.params.copy_from(params);
        }

        #[rustfmt::skip]
        fn residuals(&self) -> Option<VectorN<f64, U15>> {
            const Y1: [f64; 15] = [
                0.14, 0.18, 0.22, 0.25, 0.29,
                0.32, 0.35, 0.39, 0.37, 0.58,
                0.73, 0.96, 1.34, 2.10, 4.39,
            ];
            let p = &self.params;
            Some(VectorN::<f64, U15>::from_fn(|i, _j| {
                let tmp2 = (15 - i) as f64;
                let tmp3 = if i > 7 { tmp2 } else { (i + 1) as f64 };
                Y1[i] - (p[0] + (i + 1) as f64 / (p[1] * tmp2 + p[2] * tmp3))
            }))
        }

        #[rustfmt::skip]
        fn jacobian(&self) -> Option<MatrixMN<f64, U15, U3>> {
            let p = &self.params;
            Some(MatrixMN::<f64, U15, U3>::from_fn(|i, j| {
                let tmp2 = (15 - i) as f64;
                let tmp3 = if i > 7 { tmp2 } else { (i + 1) as f64 };
                let tmp4 = (p[1] * tmp2 + p[2] * tmp3).powi(2);
                match j {
                    0 => -1.,
                    1 => (i + 1) as f64 * tmp2 / tmp4,
                    2 => (i + 1) as f64 * tmp3 / tmp4,
                    _ => unreachable!(),
                }
            }))
        }
    }

    let initial = Vector3::<f64>::new(1., 1., 1.);
    let reason = TerminationReason::Converged {
        ftol: true,
        xtol: false,
    };
    let mut problem = Bard {
        params: Vector3::<f64>::zeros(),
    };
    let jac_num =
        differentiate_numerically(Vector3::<f64>::new(0.81, 0.47, 0.43), &mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-9);

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(report.termination, reason);
    assert_relative_eq!(report.objective_function, 0.00410743865329062);
    assert_relative_eq!(
        problem.params,
        Vector3::<f64>::new(0.0824105765758334, 1.1330366534715044, 2.343694638941154)
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x * 10.), problem.clone());
    assert_eq!(report.termination, reason);
    assert_relative_eq!(report.objective_function, 8.71434685503351);
    assert_relative_eq!(
        problem.params,
        Vector3::<f64>::new(
            8.4066667381832927e-01,
            -1.5884803325956547e+08,
            -1.6437867165353525e+08
        )
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x * 100.), problem.clone());
    assert_eq!(report.termination, reason);
    assert_relative_eq!(report.objective_function, 8.714346854926243);
    assert_relative_eq!(
        problem.params,
        Vector3::<f64>::new(
            8.4066667386764549e-01,
            -1.5894616720551842e+08,
            -1.6446490685777116e+08
        )
    );
}

#[test]
fn test_kowalik_osborne() {
    #[derive(Clone)]
    struct KowalikOsborne {
        params: VectorN<f64, U4>,
    }
    const V: [f64; 11] = [
        4., 2., 1., 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625,
    ];
    const Y2: [f64; 11] = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ];
    impl LeastSquaresProblem<f64, U4, U11> for KowalikOsborne {
        type ParameterStorage = Owned<f64, U4>;
        type ResidualStorage = Owned<f64, U11>;
        type JacobianStorage = Owned<f64, U11, U4>;

        fn set_params(&mut self, params: &VectorN<f64, U4>) {
            self.params.copy_from(params);
        }

        #[rustfmt::skip]
        fn residuals(&self) -> Option<VectorN<f64, U11>> {
            let p = &self.params;
            Some(VectorN::<f64, U11>::from_fn(|i, _j| {
                let tmp1 = V[i] * (V[i] + p[1]);
                let tmp2 = V[i] * (V[i] + p[2]) + p[3];
                Y2[i] - p[0] * tmp1 / tmp2
            }))
        }

        #[rustfmt::skip]
        fn jacobian(&self) -> Option<MatrixMN<f64, U11, U4>> {
            let p = &self.params;
            Some(MatrixMN::<f64, U11, U4>::from_fn(|i, j| {
                let tmp1 = V[i] * (V[i] + p[1]);
                let tmp2 = V[i] * (V[i] + p[2]) + p[3];
                match j {
                    0 => -tmp1 / tmp2,
                    1 => -V[i] * p[0] / tmp2,
                    2 => (tmp1 / tmp2) * (V[i] * p[0] / tmp2),
                    3 => (tmp1 / tmp2) * (V[i] * p[0] / tmp2) / V[i],
                    _ => unreachable!(),
                }
            }))
        }
    }
    let initial = Vector4::<f64>::new(0.25, 0.39, 0.415, 0.39);
    let reason = TerminationReason::Converged {
        ftol: true,
        xtol: false,
    };
    let mut problem = KowalikOsborne {
        params: Vector4::<f64>::zeros(),
    };
    let jac_num =
        differentiate_numerically(Vector4::<f64>::new(0.25, 0.31, 0.19, 0.75), &mut problem)
            .unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-9);

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(report.termination, reason);
    assert_relative_eq!(report.objective_function, 0.00015375280229088455);
    assert_relative_eq!(
        problem.params,
        Vector4::<f64>::new(
            0.19280781047624931,
            0.1912626533540709,
            0.12305280104693087,
            0.13605322115051674
        )
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .with_patience(100)
        .minimize(initial.map(|x| x * 100.), problem.clone());
    assert_eq!(report.termination, TerminationReason::LostPatience);
    assert_relative_eq!(report.objective_function, 0.00015375283657222266);
    assert_relative_eq!(
        problem.params,
        Vector4::<f64>::new(
            0.19279840638465487,
            0.1914736844615448,
            0.1230924753714115,
            0.13615096290622444
        )
    );
}

#[test]
fn test_meyer() {
    #[rustfmt::skip]
    const Y3: [f64; 16] = [
        3.478e4, 2.861e4, 2.365e4, 1.963e4,
        1.637e4, 1.372e4, 1.154e4, 9.744e3,
        8.261e3, 7.03e3, 6.005e3, 5.147e3,
        4.427e3, 3.82e3, 3.307e3, 2.872e3
    ];
    #[derive(Clone)]
    struct Meyer {
        params: VectorN<f64, U3>,
    }
    impl LeastSquaresProblem<f64, U3, U16> for Meyer {
        type ParameterStorage = Owned<f64, U3>;
        type ResidualStorage = Owned<f64, U16>;
        type JacobianStorage = Owned<f64, U16, U3>;

        fn set_params(&mut self, params: &VectorN<f64, U3>) {
            self.params.copy_from(params);
        }

        #[rustfmt::skip]
        fn residuals(&self) -> Option<VectorN<f64, U16>> {
            let p = &self.params;
            Some(VectorN::<f64, U16>::from_fn(|i, _j| {
                let temp = 5. * (i + 1) as f64 + 45. + p[2];
                p[0] * (p[1] / temp).exp() - Y3[i]
            }))
        }

        #[rustfmt::skip]
        fn jacobian(&self) -> Option<MatrixMN<f64, U16, U3>> {
            let p = &self.params;
            Some(MatrixMN::<f64, U16, U3>::from_fn(|i, j| {
                let temp = 5. * (i + 1) as f64 + 45. + p[2];
                let tmp1 = p[1] / temp;
                let tmp2 = tmp1.exp();
                match j {
                    0 => tmp2,
                    1 => p[0] * tmp2 / temp,
                    2 => -(p[0] * tmp2) * tmp1 / temp,
                    _ => unreachable!(),
                }
            }))
        }
    }
    let initial = Vector3::<f64>::new(0.02, 4000., 250.);
    let reason = TerminationReason::Converged {
        ftol: false,
        xtol: true,
    };
    let mut problem = Meyer {
        params: Vector3::<f64>::zeros(),
    };
    let jac_num =
        differentiate_numerically(Vector3::<f64>::new(0.34, 0.62, 0.03), &mut problem).unwrap();
    problem.set_params(&Vector3::<f64>::new(0.34, 0.62, 0.03));
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-5);

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(report.termination, reason);
    assert_relative_eq!(report.objective_function, 43.972927585339875);
    assert_relative_eq!(
        problem.params,
        Vector3::<f64>::new(
            5.609636471027749e-03,
            6.181346346286417e+03,
            3.452236346241380e+02
        )
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .with_patience(100)
        .minimize(initial.map(|x| x * 10.), problem.clone());
    assert_eq!(report.termination, TerminationReason::LostPatience);
    assert_relative_eq!(report.objective_function, 324272.8973474361);
    assert_relative_eq!(
        problem.params,
        Vector3::<f64>::new(
            6.825630280624222e-12,
            3.514598925134810e+04,
            9.220430560142615e+02,
        )
    );
}

#[test]
fn test_watson() {
    #[derive(Clone)]
    struct Watson<P: DimName>
    where
        DefaultAllocator: Allocator<f64, P>,
    {
        params: VectorN<f64, P>,
    }
    impl<P: DimName> LeastSquaresProblem<f64, P, U31> for Watson<P>
    where
        DefaultAllocator: Allocator<f64, P> + Allocator<f64, U31, P>,
    {
        type ParameterStorage = Owned<f64, P>;
        type ResidualStorage = Owned<f64, U31>;
        type JacobianStorage = Owned<f64, U31, P>;

        fn set_params(&mut self, params: &VectorN<f64, P>) {
            self.params.copy_from(params);
        }

        #[rustfmt::skip]
        fn residuals(&self) -> Option<VectorN<f64, U31>> {
            let params = &self.params;
            let div = VectorN::<f64, U29>::from_fn(|i,_| (i + 1) as f64 / 29.);
            let mut s1 = VectorN::<f64, U29>::zeros();
            let mut dx = VectorN::<f64, U29>::from_element(1.);
            for (j, p) in params.iter().enumerate().skip(1) {
                s1 += (j as f64 * *p) * dx;
                dx.component_mul_assign(&div);
            }
            let mut s2 = VectorN::<f64, U29>::zeros();
            let mut dx = VectorN::<f64, U29>::from_element(1.);
            for p in params.iter() {
                s2 += dx * *p;
                dx.component_mul_assign(&div);
            }
            let mut residuals = VectorN::<f64, U31>::zeros();
            s1.cmpy(-1., &s2, &s2, 1.);
            s1.apply(|x| x - 1.);
            residuals.rows_range_mut(..29).copy_from(&s1);
            residuals[29] = params[0];
            residuals[30] = params[1] - params[0].powi(2) - 1.;
            Some(residuals)
        }

        #[rustfmt::skip]
        fn jacobian(&self) -> Option<MatrixMN<f64, U31, P>> {
            let params = &self.params;
            let div = VectorN::<f64, U29>::from_fn(|i,_| (i + 1) as f64 / 29.);
            let mut s2 = VectorN::<f64, U29>::zeros();
            let mut dx = VectorN::<f64, U29>::from_element(1.);
            for p in params.iter() {
                s2 += dx * *p;
                dx.component_mul_assign(&div);
            }
            let mut temp = VectorN::<f64, U29>::zeros();
            temp.cmpy(2., &div, &s2, 0.);
            dx.copy_from(&div);
            dx.apply(|x| x.recip());

            let mut jac = MatrixMN::<f64, U31, P>::zeros();
            for j in 0..params.len() {
                let mut col = jac.column_mut(j);
                let mut col_slice = col.rows_range_mut(..29);
                col_slice.copy_from(&dx);
                col_slice.cmpy(-1., &dx, &temp, j as f64);
                dx.component_mul_assign(&div);
            }
            jac[(29, 0)] = 1.;
            jac[(30, 0)] = -2. * params[0];
            jac[(30, 1)] = 1.;
            Some(jac)
        }
    }
    let mut problem = Watson {
        params: Vector6::<f64>::zeros(),
    };
    let p = Vector6::<f64>::new(0.421, 0.606, 0.705, 0.851, 0.669, 0.250);
    let jac_num = differentiate_numerically(p.clone(), &mut problem).unwrap();
    problem.set_params(&p);
    let jac_trait = problem.jacobian().unwrap();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-8);

    let initial = Vector6::<f64>::zeros();
    let reason_f = TerminationReason::Converged {
        ftol: true,
        xtol: false,
    };

    // ==================================================
    // watson 6
    // ==================================================

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(report.termination, reason_f);
    assert_eq!(report.number_of_evaluations, 8);
    assert_relative_eq!(report.objective_function, 0.001143835026786261);
    assert_relative_eq!(
        problem.params,
        Vector6::<f64>::new(
            -0.01572496150837828,
            1.0124348823296545,
            -0.23299172238767143,
            1.260431011028177,
            -1.5137303139441967,
            0.9929972729184159,
        )
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x + 10.), problem.clone());
    assert_eq!(report.termination, reason_f);
    assert_eq!(report.number_of_evaluations, 14);
    assert_relative_eq!(report.objective_function, 0.0011438350267831846);
    assert_relative_eq!(
        problem.params,
        Vector6::<f64>::new(
            -0.015725190138667525,
            1.0124348586010505,
            -0.23299154584382673,
            1.2604293208916204,
            -1.5137277670657403,
            0.9929957342632777,
        )
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x + 100.), problem.clone());
    assert_eq!(report.termination, reason_f);
    assert_eq!(report.number_of_evaluations, 15);
    assert_relative_eq!(report.objective_function, 0.0011438350268716062);
    assert_relative_eq!(
        problem.params,
        Vector6::<f64>::new(
            -0.01572470197125869,
            1.0124349092565827,
            -0.2329919227616415,
            1.2604329292955434,
            -1.513733204527061,
            0.9929990192232175,
        )
    );

    // ==================================================
    // watson 9
    // ==================================================

    let problem = Watson {
        params: VectorN::<f64, U9>::zeros(),
    };
    let reason_fg = TerminationReason::Converged {
        ftol: true,
        xtol: true,
    };
    let initial = VectorN::<f64, U9>::zeros();

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(report.termination, reason_fg);
    assert_eq!(report.number_of_evaluations, 8);
    assert_relative_eq!(report.objective_function, 6.998800690506343e-07);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U9>::from_column_slice(&[
            -1.5307064416628804e-05,
            9.9978970393459676e-01,
            1.4763963491099890e-02,
            1.4634233014597900e-01,
            1.0008210945482034e+00,
            -2.6177311207051202e+00,
            4.1044031394335869e+00,
            -3.1436122623624456e+00,
            1.0526264037876010e+00,
        ])
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x + 10.), problem.clone());
    assert_eq!(report.termination, reason_fg);
    assert_eq!(report.number_of_evaluations, 20);
    assert_relative_eq!(report.objective_function, 6.998800690471173e-07);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U9>::from_column_slice(&[
            -1.5307036495997912e-05,
            9.9978970393194666e-01,
            1.4763963693703627e-02,
            1.4634232829808710e-01,
            1.0008211030105516e+00,
            -2.6177311405327139e+00,
            4.1044031644962153e+00,
            -3.1436122785677023e+00,
            1.0526264080131180e+00,
        ])
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x + 100.), problem.clone());
    assert_eq!(report.termination, reason_f);
    assert_eq!(report.number_of_evaluations, 18);
    assert_relative_eq!(report.objective_function, 6.998800690486009e-07);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U9>::from_column_slice(&[
            -1.5306952335212645e-05,
            9.9978970395837152e-01,
            1.4763962518529752e-02,
            1.4634234109641628e-01,
            1.0008210472912598e+00,
            -2.6177310157356275e+00,
            4.1044030142719174e+00,
            -3.1436121860244794e+00,
            1.0526263851676092e+00,
        ])
    );

    // ==================================================
    // watson 12
    // ==================================================

    let problem = Watson {
        params: VectorN::<f64, U12>::zeros(),
    };
    let initial = VectorN::<f64, U12>::zeros();

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.clone(), problem.clone());
    assert_eq!(report.termination, reason_fg);
    assert_eq!(report.number_of_evaluations, 10);
    assert_relative_eq!(report.objective_function, 2.3611905506971735e-10);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U12>::from_column_slice(&[
            -6.6380604677589803e-09,
            1.0000016441178612e+00,
            -5.6393221015137217e-04,
            3.4782054049969546e-01,
            -1.5673150405406330e-01,
            1.0528151769858700e+00,
            -3.2472711527607245e+00,
            7.2884348965512684e+00,
            -1.0271848239579612e+01,
            9.0741136457303284e+00,
            -4.5413754661102059e+00,
            1.0120118884445952e+00,
        ])
    );

    let reason_x = TerminationReason::Converged {
        ftol: false,
        xtol: true,
    };
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x + 10.), problem.clone());
    assert_eq!(report.termination, reason_x);
    assert_eq!(report.number_of_evaluations, 13);
    assert_relative_eq!(report.objective_function, 2.361190552167311e-10);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U12>::from_column_slice(&[
            -6.6380604668544608e-09,
            1.0000016441178616e+00,
            -5.6393221029791976e-04,
            3.4782054050317829e-01,
            -1.5673150408911857e-01,
            1.0528151771767233e+00,
            -3.2472711533826666e+00,
            7.2884348978198767e+00,
            -1.0271848241212496e+01,
            9.0741136470182528e+00,
            -4.5413754666784278e+00,
            1.0120118885519702e+00
        ])
    );

    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(TOL)
        .minimize(initial.map(|x| x + 100.), problem.clone());
    assert_eq!(report.termination, reason_x);
    assert_eq!(report.number_of_evaluations, 34);
    assert_relative_eq!(report.objective_function, 2.361190551562772e-10);
    assert_relative_eq!(
        problem.params,
        VectorN::<f64, U12>::from_column_slice(&[
            -6.6380604636792693e-09,
            1.0000016441178616e+00,
            -5.6393221027197340e-04,
            3.4782054050235750e-01,
            -1.5673150407932457e-01,
            1.0528151771168239e+00,
            -3.2472711531707001e+00,
            7.2884348973610109e+00,
            -1.0271848240595697e+01,
            9.0741136465161336e+00,
            -4.5413754664517798e+00,
            1.0120118885084435e+00,
        ])
    );
}
