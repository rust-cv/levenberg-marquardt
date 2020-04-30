use ::core::f64::{INFINITY, NAN};
use alloc::vec;

use approx::assert_relative_eq;
use nalgebra::*;

use super::test_helpers::{MockCall, MockProblem};

use super::{LevenbergMarquardt, TerminationReason, LM};
use crate::qr::PivotedQR;

#[test]
fn gnorm_and_gtol() {
    let problem =
        MockProblem::<U2, U3>::new(Vector2::zeros(), vec![Some(Vector3::new(1., 2., 0.5))]);
    let config = LevenbergMarquardt::new().with_gtol(0.98);
    let jacobian = Matrix3x2::new(1., 2., 4., -2., 0.5, 0.1);

    let (mut lm, residuals) = LM::new(&config, problem.clone()).ok().unwrap();
    let mut lls = PivotedQR::new(jacobian.clone()).into_least_squares_diagonal_problem(residuals);
    assert_eq!(lm.update_diag(&mut lls), Err(TerminationReason::Orthogonal));
    assert_eq!(lm.target.calls(), &[MockCall::Residuals]);

    let config = LevenbergMarquardt::new().with_gtol(0.96);
    let (mut lm, residuals) = LM::new(&config, problem.clone()).ok().unwrap();
    let mut lls = PivotedQR::new(jacobian.clone()).into_least_squares_diagonal_problem(residuals);
    assert_ne!(lm.update_diag(&mut lls), Err(TerminationReason::Orthogonal));
    assert_eq!(lm.target.calls(), &[MockCall::Residuals]);
}

#[test]
fn diag_init_and_second_call() {
    let problem = MockProblem::<U2, U3>::new(
        Vector2::new(1.5, 10.),
        vec![Some(Vector3::new(1., 2., 0.5))],
    );
    let config = LevenbergMarquardt::new().with_stepbound(42.);
    let mut jacobian = Matrix3x2::new(1., 2., 4., -2., 0.5, 0.1);
    let (mut lm, residuals) = LM::new(&config, problem.clone()).ok().unwrap();

    let mut lls = PivotedQR::new(jacobian.clone()).into_least_squares_diagonal_problem(residuals);
    assert!(lm.update_diag(&mut lls).is_ok());
    assert_eq!(lm.target.calls(), &[MockCall::Residuals]);
    // set diagonal to the column norms of J
    assert_relative_eq!(lm.diag, Vector2::new(4.153311931459037, 2.8301943396169813));
    // xnorm = ||D * x||
    assert_relative_eq!(lm.xnorm, 28.979518629542486);
    assert_eq!(lm.delta, lm.xnorm * 42.);
    let delta = lm.delta;

    // change column norms of J
    jacobian[(0, 0)] = 100.;
    jacobian[(0, 1)] = 0.;
    let mut lls = PivotedQR::new(jacobian.clone()).into_least_squares_diagonal_problem(residuals);

    lm.xnorm = 123.;
    assert!(lm.update_diag(&mut lls).is_ok());
    assert_eq!(lm.target.calls(), &[MockCall::Residuals]);
    // on second call only pick max
    assert_relative_eq!(
        lm.diag,
        Vector2::new(100.08121701897915, 2.8301943396169813)
    );
    // on second call not touched
    assert_eq!(lm.xnorm, 123.);
    assert_eq!(lm.delta, delta);
}

#[test]
fn nan_inf_xnorm() {
    fn setup(x: Vector2<f64>, jacobian: Matrix3x2<f64>) -> TerminationReason {
        let problem = MockProblem::<U2, U3>::new(x, vec![Some(Vector3::new(1., 2., 0.5))]);
        let config = LevenbergMarquardt::new();
        let (mut lm, residuals) = LM::new(&config, problem).ok().unwrap();
        let mut lls = PivotedQR::new(jacobian).into_least_squares_diagonal_problem(residuals);
        let res = lm.update_diag(&mut lls).err().unwrap();
        assert_eq!(lm.target.calls(), &[MockCall::Residuals]);
        res
    }
    if cfg!(not(feature = "minpack-compat")) {
        let jacobian = Matrix3x2::new(1., 2., 4., -2., 0.5, 0.1);
        assert_eq!(
            setup(Vector2::new(INFINITY, 0.), jacobian.clone()),
            TerminationReason::Numerical("subproblem x")
        );
        assert_eq!(
            setup(Vector2::new(NAN, 0.), jacobian.clone()),
            TerminationReason::Numerical("subproblem x")
        );
    }

    let x = Vector2::new(1., 2.);
    let termination_reason = if cfg!(feature = "minpack-compat") {
        TerminationReason::Orthogonal
    } else {
        TerminationReason::Numerical("jacobian")
    };
    assert_eq!(
        setup(x.clone(), Matrix3x2::new(INFINITY, 2., 4., -2., 0.5, 0.1)),
        termination_reason
    );
    assert_eq!(
        setup(x.clone(), Matrix3x2::new(NAN, 2., 4., -2., 0.5, 0.1)),
        termination_reason
    );
}

#[test]
fn zero_x() {
    let problem =
        MockProblem::<U2, U3>::new(Vector2::zeros(), vec![Some(Vector3::new(1., 2., 0.5))]);
    let config = LevenbergMarquardt::new().with_stepbound(900.);
    let jacobian = Matrix3x2::new(1., 2., 4., -2., 0.5, 0.1);
    let (mut lm, residuals) = LM::new(&config, problem.clone()).ok().unwrap();
    let mut lls = PivotedQR::new(jacobian.clone()).into_least_squares_diagonal_problem(residuals);
    assert!(lm.update_diag(&mut lls).is_ok());
    assert_eq!(lm.xnorm, 0.);
    assert_eq!(lm.delta, 900.);
    assert_eq!(lm.target.calls(), &[MockCall::Residuals]);
}

#[test]
fn no_scale_diag() {
    let initial_x = Vector2::new(1.5, 10.);
    let problem =
        MockProblem::<U2, U3>::new(initial_x.clone(), vec![Some(Vector3::new(1., 2., 0.5))]);
    let config = LevenbergMarquardt::new()
        .with_scale_diag(false)
        .with_stepbound(0.5);
    let mut jacobian = Matrix3x2::new(1., 2., 4., -2., 0.5, 0.1);
    let (mut lm, residuals) = LM::new(&config, problem.clone()).ok().unwrap();
    let mut lls = PivotedQR::new(jacobian.clone()).into_least_squares_diagonal_problem(residuals);
    assert!(lm.update_diag(&mut lls).is_ok());
    assert_eq!(lm.target.calls(), &[MockCall::Residuals]);
    assert_relative_eq!(lm.diag, Vector2::new(1., 1.));
    // xnorm = ||D * x||
    assert_eq!(lm.xnorm, initial_x.norm());
    assert_eq!(lm.delta, lm.xnorm * 0.5);
    let delta = lm.delta;

    // change column norms of J
    jacobian[(0, 0)] = 100.;
    jacobian[(0, 1)] = 0.;
    let mut lls = PivotedQR::new(jacobian.clone()).into_least_squares_diagonal_problem(residuals);

    lm.xnorm = 123.;
    assert!(lm.update_diag(&mut lls).is_ok());
    assert_eq!(lm.target.calls(), &[MockCall::Residuals]);
    // on second call still no changed
    assert_relative_eq!(lm.diag, Vector2::new(1., 1.));
    // on second call not touched
    assert_eq!(lm.xnorm, 123.);
    assert_eq!(lm.delta, delta);
}
