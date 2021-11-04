//! Tests with example functions.
//!
//! There is also a Python implementation in `test_examples.py`
//! which was used to get the output from MINPACK.
use approx::assert_relative_eq;
use nalgebra::*;
use nalgebra::{allocator::Allocator, storage::Owned};

use crate::utils::differentiate_numerically;
use crate::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};

cfg_if::cfg_if! {
    if #[cfg(feature = "minpack-compat")] {
        // in "minpack-compat" mode we want real equality
        macro_rules! assert_fp_eq {
            ($given:expr, $expected:expr) => {
                assert_eq!($given, $expected)
            };
        }
    } else {
        macro_rules! assert_fp_eq {
            ($given:expr, $expected:expr) => {
                assert_relative_eq!($given, $expected, epsilon = 1e-12)
            };
            ($given:expr, $expected:expr, $ep:expr) => {
                assert_relative_eq!($given, $expected, epsilon = $ep)
            };
        }
    }
}

/// TOL value used by SciPy
const TOL: f64 = 1.49012e-08;

#[derive(Clone)]
pub struct LinearFullRank {
    pub params: OVector<f64, U5>,
    pub m: usize,
}

impl LinearFullRank {
    fn new(params: OVector<f64, U5>, m: usize) -> Self {
        Self { params, m }
    }
}

impl LeastSquaresProblem<f64, Dynamic, U5> for LinearFullRank {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, U5>;

    fn set_params(&mut self, params: &OVector<f64, U5>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U5> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Dynamic>> {
        let m = Dynamic::from_usize(self.m);
        let u1 = Dim::from_usize(1);
        let mut residuals = OVector::<f64, Dynamic>::from_element_generic(
            m,
            u1,
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

    fn jacobian(&self) -> Option<OMatrix<f64, Dynamic, U5>> {
        let m = Dynamic::from_usize(self.m);
        let u5 = Dim::from_usize(5);
        let mut jacobian = OMatrix::from_element_generic(m, u5, -2. / self.m as f64);
        for i in 0..5 {
            jacobian[(i, i)] += 1.;
        }
        Some(jacobian)
    }
}

#[derive(Clone)]
struct LinearRank1 {
    params: OVector<f64, U5>,
    m: usize,
}

impl LinearRank1 {
    fn new(params: OVector<f64, U5>, m: usize) -> Self {
        Self { params, m }
    }
}

impl LeastSquaresProblem<f64, Dynamic, U5> for LinearRank1 {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, U5>;

    fn set_params(&mut self, params: &OVector<f64, U5>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U5> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Dynamic>> {
        let m = Dynamic::from_usize(self.m);
        let weighted_sum: f64 = self
            .params
            .iter()
            .enumerate()
            .map(|(j, p)| (j + 1) as f64 * p)
            .sum();
        let u1 = Dim::from_usize(1);
        Some(OVector::<f64, Dynamic>::from_iterator_generic(
            m,
            u1,
            (0..self.m).map(|i| (i + 1) as f64 * weighted_sum - 1.),
        ))
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dynamic, U5>> {
        let m = Dynamic::from_usize(self.m);
        let u5 = Dim::from_usize(5);
        Some(OMatrix::from_fn_generic(m, u5, |i, j| {
            ((i + 1) * (j + 1)) as f64
        }))
    }
}

#[derive(Clone)]
struct LinearRank1ZeroColumns {
    params: OVector<f64, U5>,
    m: usize,
}

impl LinearRank1ZeroColumns {
    fn new(params: OVector<f64, U5>, m: usize) -> Self {
        Self { params, m }
    }
}

impl LeastSquaresProblem<f64, Dynamic, U5> for LinearRank1ZeroColumns {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, U5>;

    fn set_params(&mut self, params: &OVector<f64, U5>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U5> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Dynamic>> {
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
        let u1 = Dim::from_usize(1);
        Some(OVector::<f64, Dynamic>::from_iterator_generic(
            m,
            u1,
            (0..self.m).map(|i| {
                if i == self.m - 1 {
                    -1.
                } else {
                    i as f64 * weighted_sum - 1.
                }
            }),
        ))
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dynamic, U5>> {
        let m = Dynamic::from_usize(self.m);
        let u5 = Dim::from_usize(5);
        Some(OMatrix::from_fn_generic(m, u5, |i, j| {
            if i >= 1 && j >= 1 && j < 5 - 1 && i < self.m - 1 {
                ((j + 1) * i) as f64
            } else {
                0.
            }
        }))
    }
}

#[derive(Clone)]
struct Rosenbruck {
    params: OVector<f64, U2>,
}
impl LeastSquaresProblem<f64, U2, U2> for Rosenbruck {
    type ParameterStorage = Owned<f64, U2>;
    type ResidualStorage = Owned<f64, U2>;
    type JacobianStorage = Owned<f64, U2, U2>;

    fn set_params(&mut self, params: &OVector<f64, U2>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U2> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, U2>> {
        Some(Vector2::new(
            10. * (self.params[1] - self.params[0] * self.params[0]),
            1. - self.params[0],
        ))
    }

    fn jacobian(&self) -> Option<OMatrix<f64, U2, U2>> {
        Some(Matrix2::new(-20. * self.params[0], 10., -1., 0.))
    }
}

const TPI: f64 = ::core::f64::consts::PI * 2.;

#[derive(Clone)]
struct HelicalValley {
    params: OVector<f64, U3>,
}
impl LeastSquaresProblem<f64, U3, U3> for HelicalValley {
    type ParameterStorage = Owned<f64, U3>;
    type ResidualStorage = Owned<f64, U3>;
    type JacobianStorage = Owned<f64, U3, U3>;

    fn set_params(&mut self, params: &OVector<f64, U3>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U3> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, U3>> {
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
    fn jacobian(&self) -> Option<OMatrix<f64, U3, U3>> {
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

#[derive(Clone)]
struct PowellSingular {
    params: OVector<f64, U4>,
}
impl LeastSquaresProblem<f64, U4, U4> for PowellSingular {
    type ParameterStorage = Owned<f64, U4>;
    type ResidualStorage = Owned<f64, U4>;
    type JacobianStorage = Owned<f64, U4, U4>;

    fn set_params(&mut self, params: &OVector<f64, U4>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U4> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, U4>> {
        let p = self.params;
        Some(Vector4::new(
            p[0] + 10. * p[1],
            (5.).sqrt() * (p[2] - p[3]),
            (p[1] - 2. * p[2]).powi(2),
            (10.).sqrt() * (p[0] - p[3]).powi(2),
        ))
    }

    #[rustfmt::skip]
    fn jacobian(&self) -> Option<OMatrix<f64, U4, U4>> {
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

#[derive(Clone)]
struct FreudensteinRoth {
    params: OVector<f64, U2>,
}
impl LeastSquaresProblem<f64, U2, U2> for FreudensteinRoth {
    type ParameterStorage = Owned<f64, U2>;
    type ResidualStorage = Owned<f64, U2>;
    type JacobianStorage = Owned<f64, U2, U2>;

    fn set_params(&mut self, params: &OVector<f64, U2>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U2> {
        self.params
    }

    #[rustfmt::skip]
    fn residuals(&self) -> Option<OVector<f64, U2>> {
        let p = &self.params;
        Some(Vector2::new(
            -13. + p[0] + ((5. - p[1]) * p[1] - 2.) * p[1],
            -29. + p[0] + ((1. + p[1]) * p[1] - 14.) * p[1]
        ))
    }

    #[rustfmt::skip]
    fn jacobian(&self) -> Option<OMatrix<f64, U2, U2>> {
        let p = &self.params;
        Some(Matrix2::new(
            1. , p[1] * (10. - 3. * p[1]) - 2.,
            1.,  p[1] * (2. + 3. * p[1]) - 14.,
        ))
    }
}

#[derive(Clone)]
struct Bard {
    params: OVector<f64, U3>,
}
impl LeastSquaresProblem<f64, U15, U3> for Bard {
    type ParameterStorage = Owned<f64, U3>;
    type ResidualStorage = Owned<f64, U15>;
    type JacobianStorage = Owned<f64, U15, U3>;

    fn set_params(&mut self, params: &OVector<f64, U3>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U3> {
        self.params
    }

    #[rustfmt::skip]
    fn residuals(&self) -> Option<OVector<f64, U15>> {
        const Y1: [f64; 15] = [
            0.14, 0.18, 0.22, 0.25, 0.29,
            0.32, 0.35, 0.39, 0.37, 0.58,
            0.73, 0.96, 1.34, 2.10, 4.39,
        ];
        let p = &self.params;
        Some(OVector::<f64, U15>::from_fn(|i, _j| {
            let tmp2 = (15 - i) as f64;
            let tmp3 = if i > 7 { tmp2 } else { (i + 1) as f64 };
            Y1[i] - (p[0] + (i + 1) as f64 / (p[1] * tmp2 + p[2] * tmp3))
        }))
    }

    #[rustfmt::skip]
    fn jacobian(&self) -> Option<OMatrix<f64, U15, U3>> {
        let p = &self.params;
        Some(OMatrix::<f64, U15, U3>::from_fn(|i, j| {
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

#[derive(Clone)]
struct KowalikOsborne {
    params: OVector<f64, U4>,
}
const V: [f64; 11] = [
    4., 2., 1., 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625,
];
const Y2: [f64; 11] = [
    0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
];
impl LeastSquaresProblem<f64, U11, U4> for KowalikOsborne {
    type ParameterStorage = Owned<f64, U4>;
    type ResidualStorage = Owned<f64, U11>;
    type JacobianStorage = Owned<f64, U11, U4>;

    fn set_params(&mut self, params: &OVector<f64, U4>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U4> {
        self.params
    }

    #[rustfmt::skip]
    fn residuals(&self) -> Option<OVector<f64, U11>> {
        let p = &self.params;
        Some(OVector::<f64, U11>::from_fn(|i, _j| {
            let tmp1 = V[i] * (V[i] + p[1]);
            let tmp2 = V[i] * (V[i] + p[2]) + p[3];
            Y2[i] - p[0] * tmp1 / tmp2
        }))
    }

    #[rustfmt::skip]
    fn jacobian(&self) -> Option<OMatrix<f64, U11, U4>> {
        let p = &self.params;
        Some(OMatrix::<f64, U11, U4>::from_fn(|i, j| {
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

#[rustfmt::skip]
const Y3: [f64; 16] = [
    3.478e4, 2.861e4, 2.365e4, 1.963e4,
    1.637e4, 1.372e4, 1.154e4, 9.744e3,
    8.261e3, 7.03e3, 6.005e3, 5.147e3,
    4.427e3, 3.82e3, 3.307e3, 2.872e3
];
#[derive(Clone)]
struct Meyer {
    params: OVector<f64, U3>,
}
impl LeastSquaresProblem<f64, U16, U3> for Meyer {
    type ParameterStorage = Owned<f64, U3>;
    type ResidualStorage = Owned<f64, U16>;
    type JacobianStorage = Owned<f64, U16, U3>;

    fn set_params(&mut self, params: &OVector<f64, U3>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U3> {
        self.params
    }

    #[rustfmt::skip]
    fn residuals(&self) -> Option<OVector<f64, U16>> {
        let p = &self.params;
        Some(OVector::<f64, U16>::from_fn(|i, _j| {
            let temp = 5. * (i + 1) as f64 + 45. + p[2];
            p[0] * (p[1] / temp).exp() - Y3[i]
        }))
    }

    #[rustfmt::skip]
    fn jacobian(&self) -> Option<OMatrix<f64, U16, U3>> {
        let p = &self.params;
        Some(OMatrix::<f64, U16, U3>::from_fn(|i, j| {
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

#[derive(Clone)]
struct Watson<P: DimName>
where
    DefaultAllocator: Allocator<f64, P>,
{
    params: OVector<f64, P>,
}

impl<P: DimName> Watson<P>
where
    DefaultAllocator: Allocator<f64, P>,
{
    fn new(params: OVector<f64, P>, _n: usize) -> Self {
        Self { params }
    }
}

impl<P: DimName> LeastSquaresProblem<f64, U31, P> for Watson<P>
where
    DefaultAllocator: Allocator<f64, P> + Allocator<f64, U31, P>,
{
    type ParameterStorage = Owned<f64, P>;
    type ResidualStorage = Owned<f64, U31>;
    type JacobianStorage = Owned<f64, U31, P>;

    fn set_params(&mut self, params: &OVector<f64, P>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, P> {
        self.params.clone()
    }

    #[rustfmt::skip]
    fn residuals(&self) -> Option<OVector<f64, U31>> {
        let params = &self.params;
        let div = OVector::<f64, U29>::from_fn(|i,_| (i + 1) as f64 / 29.);
        let mut s1 = OVector::<f64, U29>::zeros();
        let mut dx = OVector::<f64, U29>::from_element(1.);
        for (j, p) in params.iter().enumerate().skip(1) {
            s1 += (j as f64 * *p) * dx;
            dx.component_mul_assign(&div);
        }
        let mut s2 = OVector::<f64, U29>::zeros();
        let mut dx = OVector::<f64, U29>::from_element(1.);
        for p in params.iter() {
            s2 += dx * *p;
            dx.component_mul_assign(&div);
        }
        let mut residuals = OVector::<f64, U31>::zeros();
        s1.cmpy(-1., &s2, &s2, 1.);
        s1.apply(|x| *x -= 1.);
        residuals.rows_range_mut(..29).copy_from(&s1);
        residuals[29] = params[0];
        residuals[30] = params[1] - params[0].powi(2) - 1.;
        Some(residuals)
    }

    #[rustfmt::skip]
    fn jacobian(&self) -> Option<OMatrix<f64, U31, P>> {
        let params = &self.params;
        let div = OVector::<f64, U29>::from_fn(|i,_| (i + 1) as f64 / 29.);
        let mut s2 = OVector::<f64, U29>::zeros();
        let mut dx = OVector::<f64, U29>::from_element(1.);
        for p in params.iter() {
            s2 += dx * *p;
            dx.component_mul_assign(&div);
        }
        let mut temp = OVector::<f64, U29>::zeros();
        temp.cmpy(2., &div, &s2, 0.);
        dx.copy_from(&div);
        dx.apply(|x| *x = x.recip());

        let mut jac = OMatrix::<f64, U31, P>::zeros();
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

#[derive(Clone)]
struct Beale {
    params: OVector<f64, U2>,
}

impl LeastSquaresProblem<f64, U1, U2> for Beale {
    type ParameterStorage = Owned<f64, U2>;
    type ResidualStorage = Owned<f64, U1>;
    type JacobianStorage = Owned<f64, U1, U2>;

    fn set_params(&mut self, params: &OVector<f64, U2>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U2> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, U1>> {
        let p = self.params;
        Some(Vector1::new(
            (1.5 - p[0] + p[0] * p[1]).powi(2) + (2.25 - p[0] + p[0] * (p[1] * p[1])).powi(2),
        ))
    }

    #[rustfmt::skip]
    fn jacobian(&self) -> Option<OMatrix<f64, U1, U2>> {
        let x = self.params[0];
        let y = self.params[1];
        let y3 = y * y * y;
        let dx = 0.5 * (-1. + y) * (15. + 9. * y + 4. * x * (-2. + y * y + y3));
        let dy = x * (3. + 9. * y + x * (-2. - 2. * y + 4. * y3));
        Some(Matrix1x2::new(dx, dy))
    }
}

include!("test_examples_gen.rs");
