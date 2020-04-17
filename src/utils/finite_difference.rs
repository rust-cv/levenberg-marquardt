//! Implementation of an adaptive finite difference approximation.
use alloc::{vec, vec::Vec};
use nalgebra::{convert, Matrix3, RealField};

use num_traits::float::Float;

#[cfg(test)]
use approx::assert_relative_eq;

const STEP_RATIO: f64 = 2.;

/// Compute the derivative using an adaptive central difference quotient.
///
/// The algorithm is based on
///
/// > R. S. Stepleman and N. D. Winarsky. Adaptive Numerical Differentiation.
/// > Mathematics of Computation, Vol. 33, No. 148 (Oct., 1979), pp. 1257-1264.
///
/// Additionally added:
///
/// - A Richardson extrapolation
/// - Wynn extrapolation with some outlier correction
///
pub fn derivative<F: Float + RealField>(x: F, f: impl Fn(F) -> Option<F>) -> Option<F> {
    let p_cbrt = Float::cbrt(F::default_epsilon());
    let step_ratio: F = convert(STEP_RATIO);

    // find initial h which is large enough
    let mut h = Float::max(
        Float::max(Float::ln(F::one() + Float::abs(x)), F::one()),
        step_ratio * p_cbrt * if x.is_zero() { convert(0.01) } else { x },
    );
    let mut f1 = f(x + h)?;
    let mut f2 = f(x - h)?;
    let mut changed = false;
    if f1 * f2 > F::zero() && f1 != f2 {
        let fx = match f(x)? {
            x if x.is_zero() => F::one(),
            x => x,
        };
        let delta_h0 = Float::abs((f1 - f2) / fx);
        if !delta_h0.is_zero() {
            let mut n_h0 = -Float::ln(delta_h0);
            while n_h0 > -Float::ln(p_cbrt * step_ratio) {
                h *= step_ratio;
                n_h0 -= Float::ln(step_ratio);
                changed = true;
            }
        }
    }
    if changed {
        f1 = f(x + h)?;
        f2 = f(x - h)?;
    }
    h /= step_ratio;
    let two: F = convert(2.);
    let mut quotients = vec![(f1 - f2) / (h * two), (f(x + h)? - f(x - h)?) / (h * two)];
    for i in 1.. {
        h /= step_ratio;
        if x + h == x || h < F::epsilon() {
            break;
        }
        let quot1 = quotients[quotients.len() - 2];
        let quot2 = quotients[quotients.len() - 1];
        let quot_new = (f(x + h)? - f(x - h)?) / (h * two);
        if i >= 4 && Float::abs(quot_new - quot2) > Float::abs(quot2 - quot1) * convert(10.) {
            break;
        }
        quotients.push(quot_new);
    }
    extrapolate(quotients)
}

fn extrapolate<F: RealField + Float>(evaluations: Vec<F>) -> Option<F> {
    let estimates = richardson_extrapolate(evaluations)?;
    let num = estimates.len();
    if num <= 2 {
        return estimates.last().copied();
    }
    let derivatives = wynn_extrapolate(estimates)?;
    outlier_aware_minimum(derivatives)
}

fn outlier_aware_minimum<F: std::fmt::Debug + Float>(mut values: Vec<(F, F)>) -> Option<F> {
    let num = values.len();
    let mut not_nan = num;
    // move NaN to the end
    for i in 0..num {
        let i2 = not_nan - 1;
        if values[i].0.is_nan() {
            values.swap(i, i2);
            not_nan -= 1;
        }
        if i == i2 {
            break;
        }
    }
    if not_nan == 0 {
        return None;
    }
    // sort and compute median
    let values = &mut values[..not_nan];
    let num = values.len();
    values.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let percentile = |p: f64| {
        let i = ((num - 1) as f64) * p;
        let a = F::from(i.fract()).unwrap();
        let b = F::one() - a;
        values[i.floor() as usize].0 * b + values[i.ceil() as usize].0 * a
    };
    let p25 = percentile(0.25);
    let median = percentile(0.5).abs();
    let abs_median = median;
    let p75 = percentile(0.75);
    let iqr = (p75 - p25).abs() * F::from(1.5).unwrap();

    // find lowest error considering deviation from median
    let trim_fact = F::from(10.).unwrap();
    let mut min_err = F::max_value();
    let mut best_der = F::one();
    for val in values.iter() {
        let (der, mut err) = val;
        let is_outlier = ((der.abs() < abs_median / trim_fact
            || der.abs() > abs_median * trim_fact)
            && abs_median > F::from(1.0e-8).unwrap())
            || *der < p25 - iqr
            || p75 + iqr < *der;
        if is_outlier {
            err = err + (*der - median).abs();
        }
        if err < min_err {
            best_der = *der;
            min_err = err;
        }
    }
    Some(best_der)
}

// Wynn's epsilon algorithm
fn wynn_extrapolate<F: std::fmt::Debug + Float>(estimates: Vec<F>) -> Option<Vec<(F, F)>> {
    let num = estimates.len();
    let e0 = &estimates[..num - 2];
    let e1 = &estimates[1..num - 1];
    let e2 = &estimates[2..];
    let tiny = F::min_positive_value();
    let eps = F::epsilon();
    let mut derivatives = Vec::with_capacity(num - 2);
    for i in 0..num - 2 {
        let mut d1 = e1[i] - e0[i];
        let mut d2 = e2[i] - e1[i];
        let err1 = d1.abs();
        let err2 = d2.abs();
        if err1 < tiny {
            d1 = tiny;
        }
        if err2 < tiny {
            d2 = tiny;
        }
        let tol1 = e1[i].abs().max(e0[i].abs()) * eps;
        let tol2 = e2[i].abs().max(e1[i].abs()) * eps;
        let ss = d2.recip() - d1.recip() + tiny;
        let is_small = (ss * e1[i]).abs() <= F::from(1.0e-3).unwrap();
        let converged = (err1 <= tol1 && err2 <= tol2) || is_small;
        let result = if converged { e2[i] } else { e1[i] + ss.recip() };
        let abserr = err1
            + err2
            + if converged {
                tol2 * F::from(10.).unwrap()
            } else {
                (result - e2[i]).abs()
            };
        derivatives.push((result, abserr));
    }
    Some(derivatives)
}

fn richardson_extrapolate<F: RealField>(evaluations: Vec<F>) -> Option<Vec<F>> {
    const STEP: i32 = 2;
    const ORDER: i32 = 2;
    let step_ratio: F = convert(STEP_RATIO);
    if evaluations.len() <= 3 {
        return evaluations.last().map(|x| vec![*x]);
    }
    let entry = |i: i32, j: i32| step_ratio.powi(-i * (STEP * j + ORDER));
    #[rustfmt::skip]
    let r_matrix = Matrix3::new(
        F::one(),    F::one(),    F::one(),
        F::one(), entry(1, 0), entry(2, 0),
        F::one(), entry(1, 1), entry(2, 1),
    );
    let pinv = r_matrix.pseudo_inverse(F::default_epsilon()).ok()?;
    let coeff = pinv.column(0);

    let num = evaluations.len();
    Some(
        (2..num)
            .map(|i| {
                coeff[0] * evaluations[i - 2]
                    + coeff[1] * evaluations[i - 1]
                    + coeff[2] * evaluations[i]
            })
            .collect(),
    )
}

#[test]
fn test_linear() {
    assert_relative_eq!(
        derivative(0.123f64, |x| Some(3. * x - 1.)).unwrap(),
        3.,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        derivative(-0.123, |x| Some(30. * x - 1.)).unwrap(),
        30.,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        derivative(0.0, |x| Some(-3. * x - 1.)).unwrap(),
        -3.,
        epsilon = 1e-14
    );
    assert_relative_eq!(derivative(0.2, |_: f64| Some(1.)).unwrap(), 0.);
}

#[test]
fn test_standard_functions() {
    assert_relative_eq!(
        derivative(0., |x| Some(x.exp())).unwrap(),
        1.,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        derivative(-1.2, |x| Some(x.exp())).unwrap(),
        (-1.2).exp(),
        epsilon = 5e-14
    );

    assert_relative_eq!(
        derivative(90., |x| Some(x.ln())).unwrap(),
        1. / 90.,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        derivative(1234., |x| Some(x.ln())).unwrap(),
        1. / 1234.,
        epsilon = 1e-14
    );

    assert_relative_eq!(
        derivative(238., |x| Some(x.sin())).unwrap(),
        (238.).cos(),
        epsilon = 2e-12
    );
    assert_relative_eq!(
        derivative(-34.233, |x| Some(x.sin())).unwrap(),
        (-34.233).cos(),
        epsilon = 1e-11
    );
}

#[test]
fn test_polynomial() {
    assert_relative_eq!(
        derivative(0., |x| Some(x * x)).unwrap(),
        0.,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        derivative(3., |x| Some(x * x)).unwrap(),
        6.,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        derivative(2., |x| Some(4. * x * x - 2. * x)).unwrap(),
        14.,
        epsilon = 1e-12
    );
}
