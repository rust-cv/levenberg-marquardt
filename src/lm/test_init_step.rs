use alloc::vec;
use approx::assert_relative_eq;
#[cfg(not(feature = "minpack-compat"))]
use core::f64::{INFINITY, MIN_POSITIVE, NAN};

use nalgebra::{Dim, Dynamic, OMatrix, OVector, Vector2, Vector3, U0, U2, U3};

use super::test_helpers::{MockCall, MockProblem};
use super::{LevenbergMarquardt, TerminationReason, LM};

#[test]
#[cfg(not(feature = "minpack-compat"))]
fn nan_or_inf_none_residual() {
    // residuals return None
    let problem = MockProblem::<U2, U3>::new(Vector2::zeros(), vec![]);
    let (mut problem, err) = LM::new(&LevenbergMarquardt::new(), problem).err().unwrap();
    assert_eq!(err.termination, TerminationReason::User("residuals"));
    assert_eq!(err.number_of_evaluations, 1);
    assert_eq!(problem.calls(), [MockCall::Residuals].as_ref());
    assert!(err.objective_function.is_nan());

    // residuals return inf
    let problem =
        MockProblem::<U2, U3>::new(Vector2::zeros(), vec![Some(Vector3::new(1., 1., INFINITY))]);
    let (mut problem, err) = LM::new(&LevenbergMarquardt::new(), problem).err().unwrap();
    assert_eq!(
        err.termination,
        TerminationReason::Numerical("residuals norm")
    );
    assert_eq!(err.number_of_evaluations, 1);
    assert_eq!(problem.calls(), [MockCall::Residuals].as_ref());
    assert!(err.objective_function.is_infinite());

    // residuals return nan
    let problem =
        MockProblem::<U2, U3>::new(Vector2::zeros(), vec![Some(Vector3::new(1., 1., NAN))]);
    let (mut problem, err) = LM::new(&LevenbergMarquardt::new(), problem).err().unwrap();
    assert_eq!(
        err.termination,
        TerminationReason::Numerical("residuals norm")
    );
    assert_eq!(err.number_of_evaluations, 1);
    assert_eq!(problem.calls(), [MockCall::Residuals].as_ref());
    assert!(err.objective_function.is_nan());
}

#[test]
#[cfg(not(feature = "minpack-compat"))]
fn already_zero() {
    use nalgebra::{Vector1, U1};
    use num_traits::Zero;
    let problem = MockProblem::<U2, U3>::new(Vector2::zeros(), vec![Some(Vector3::zeros())]);
    let (mut problem, err) = LM::new(&LevenbergMarquardt::new(), problem).err().unwrap();
    assert_eq!(err.termination, TerminationReason::ResidualsZero);
    assert_eq!(err.number_of_evaluations, 1);
    assert_eq!(problem.calls(), [MockCall::Residuals].as_ref());
    assert!(err.objective_function.is_zero());

    let problem =
        MockProblem::<U1, U1>::new(Vector1::new(10.), vec![Some(Vector1::new(MIN_POSITIVE))]);
    let (mut problem, err) = LM::new(&LevenbergMarquardt::new(), problem).err().unwrap();
    assert_eq!(err.termination, TerminationReason::ResidualsZero);
    assert_eq!(err.number_of_evaluations, 1);
    assert_eq!(problem.calls(), [MockCall::Residuals].as_ref());
    assert!(err.objective_function.is_zero());
}

#[test]
fn no_params() {
    // no parameters
    let problem = MockProblem::<U0, U3>::new(
        OVector::<f64, U0>::zeros(),
        vec![Some(Vector3::from_element(1.))],
    );
    let (mut problem, err) = LM::new(&LevenbergMarquardt::new(), problem).err().unwrap();
    assert_eq!(err.termination, TerminationReason::NoParameters);
    assert_eq!(err.number_of_evaluations, 1);
    assert_eq!(problem.calls(), [MockCall::Residuals].as_ref());
}

#[test]
fn wrong_dimensions() {
    // first return m=4 residuals, then m=5
    let m1 = Dynamic::from_usize(4);
    let m2 = Dynamic::from_usize(5);
    let u1 = Dim::from_usize(1);
    let u2 = Dim::from_usize(2);
    let mut problem = MockProblem::<U2, Dynamic>::new(
        OVector::zeros_generic(u2, u1),
        vec![
            Some(OVector::from_element_generic(m1, u1, 223.)),
            Some(OVector::from_element_generic(m2, u1, 223.)),
        ],
    );
    problem.jacobians = vec![
        Some(OMatrix::from_element_generic(m1, u2, 100.)),
        Some(OMatrix::from_element_generic(m2, u2, 100.)),
    ];
    let (_problem, report) = LevenbergMarquardt::new().minimize(problem);
    assert_eq!(
        report.termination,
        TerminationReason::WrongDimensions("residuals")
    );

    let mut problem = MockProblem::<U2, Dynamic>::new(
        OVector::zeros_generic(u2, u1),
        vec![Some(OVector::from_element_generic(m1, u1, 223.))],
    );
    problem.jacobians = vec![Some(OMatrix::from_element_generic(m2, u2, 100.))];
    let (_problem, report) = LevenbergMarquardt::new().minimize(problem);
    assert_eq!(
        report.termination,
        TerminationReason::WrongDimensions("jacobian")
    );
}

#[test]
fn initial_diagonal_and_residual() {
    let problem =
        MockProblem::<U2, U2>::new(Vector2::from_element(2.), vec![Some(Vector2::new(0.5, 1.))]);
    let config = LevenbergMarquardt::new();
    let (mut lm, residuals) = LM::new(&config, problem).ok().unwrap();
    assert_eq!(lm.target.calls(), [MockCall::Residuals].as_ref());
    assert_eq!(lm.diag, Vector2::new(1., 1.));
    assert_relative_eq!(
        lm.report.objective_function,
        Vector2::new(0.5, 1.).norm_squared() * 0.5
    );
    assert_eq!(residuals, Vector2::new(0.5, 1.));
}
