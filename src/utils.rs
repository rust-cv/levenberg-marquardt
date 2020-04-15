use crate::LeastSquaresProblem;
use nalgebra::{
    allocator::Allocator, convert, storage::Storage, Complex, ComplexField, DefaultAllocator, Dim,
    Matrix, MatrixMN, RealField, Vector, VectorN, U1,
};
use num_traits::float::Float;

/// Compute a numerical approximation of the Jacobian.
///
/// The function is intended to be used for debugging or testing.
/// You can try to check your derivative implementation of an
/// [`LeastSquaresProblem`](trait.LeastSquaresProblem.html) with this.
///
/// Computing the derivatives numerically is unstable: You can construct
/// functions where the computed result is catastrophically wrong. If you
/// observe large differences between the derivative computed by this function
/// and your implementation the reason _might_ be due to instabilty.
///
/// The achieved precision by this function
/// is lower than the floating point precision in general. So the error is bigger
/// than `$10^{-15}$` for `f64` and bigger than `$10^{-7}$` for `f32`. See the example
/// below for what that means in your tests. If possible use `f64` for the testing.
///
/// A much more precise alternative is provided by
/// [`differentiate_holomorphic_numerically`](fn.differentiate_holomorphic_numerically.html)
/// but it requires your residuals to be holomorphic and `LeastSquaresProblem` to be implemented
/// for complex numbers.
///
/// # Example
///
/// You can use this function to check your derivative implementation in a unit test.
/// For example:
///
/// ```rust
/// # use levenberg_marquardt::{LeastSquaresProblem, differentiate_numerically};
/// # use approx::assert_relative_eq;
/// # use nalgebra::{convert, ComplexField, storage::Owned, Matrix2, Vector2, VectorN, U2};
/// #
/// # struct ExampleProblem<F: ComplexField> {
/// #     p: Vector2<F>,
/// # }
/// #
/// # impl<F: ComplexField> LeastSquaresProblem<F, U2, U2> for ExampleProblem<F> {
/// #     type ParameterStorage = Owned<F, U2>;
/// #     type ResidualStorage = Owned<F, U2>;
/// #     type JacobianStorage = Owned<F, U2, U2>;
/// #
/// #     fn set_params(&mut self, p: &VectorN<F, U2>) {
/// #         self.p.copy_from(p);
/// #     }
/// #
/// #     fn residuals(&self) -> Option<Vector2<F>> {
/// #         Some(Vector2::new(
/// #             self.p.x * self.p.x + self.p.y - convert(11.0),
/// #             self.p.x + self.p.y * self.p.y - convert(7.0),
/// #         ))
/// #     }
/// #
/// #     fn jacobian(&self) -> Option<Matrix2<F>> {
/// #         let two: F = convert(2.);
/// #         Some(Matrix2::new(
/// #             two * self.p.x,
/// #             F::one(),
/// #             F::one(),
/// #             two * self.p.y,
/// #         ))
/// #     }
/// # }
/// // Parameters for which we want to check your derivative
/// let x = Vector2::new(6., -10.);
/// // Let `problem` be an instance of `LeastSquaresProblem`
/// # let mut problem = ExampleProblem::<f64> { p: Vector2::zeros(), };
/// problem.set_params(&x);
/// let jacobian_trait = problem.jacobian().unwrap();
/// let jacobian_numerical = differentiate_numerically(x, &mut problem).unwrap();
/// // notice the relative low epsilon of 1e-7 for f64 here
/// assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-7);
/// ```
///
/// The `assert_relative_eq!` macro is from the `approx` crate.
pub fn differentiate_numerically<F, N, M, O>(
    mut params: Vector<F, N, O::ParameterStorage>,
    problem: &mut O,
) -> Option<Matrix<F, M, N, O::JacobianStorage>>
where
    F: RealField + Float,
    N: Dim,
    M: Dim,
    O: LeastSquaresProblem<F, N, M>,
    O::JacobianStorage: Clone,
    DefaultAllocator: Allocator<F, M, N, Buffer = O::JacobianStorage>,
{
    problem.set_params(&params);
    let mut quotient = problem.residuals()?;
    let n = params.data.shape().0;
    let m = quotient.data.shape().0;
    const STENCIL: &[[f64; 2]] = &[[-1., 2.], [8., 1.], [-8., -1.], [1., -2.]];
    const SCALE: f64 = 12.;

    let mut jacobian = Matrix::<F, M, N, O::JacobianStorage>::zeros_generic(m, n);
    let eps = Float::sqrt(F::default_epsilon());
    for i in 0..n.value() {
        quotient.fill(F::zero());
        let x0 = params[i];
        let h = Float::max(eps * eps, eps * Float::abs(x0));
        for [a, b] in STENCIL.iter() {
            params[i] = x0 + h * convert(*b);
            problem.set_params(&params);
            let residuals_h_ei = problem.residuals()?;
            let a: F = convert(*a);
            quotient.axpy(a / (h * convert(SCALE)), &residuals_h_ei, F::one());
        }
        params[i] = x0;
        jacobian.column_mut(i).copy_from(&quotient);
    }
    Some(jacobian)
}

/// Compute a numerical approximation of the Jacobian for _holomorphic_ residuals.
///
/// This method is _much_ more precise than
/// [`differentiate_numerically`](fn.differentiate_numerically.html) but
/// it requires that your residuals are holomorphic on a neighborhood of the real line.
/// You also must provide an implementation of
/// [`LeastSquaresProblem`](trait.LeastSquaresProblem.html) for complex numbers.
///
/// This method is mainly intended for testing your derivative implementation.
///
/// # Example
///
/// ```rust
/// # use levenberg_marquardt::{LeastSquaresProblem, differentiate_holomorphic_numerically};
/// # use approx::assert_relative_eq;
/// # use nalgebra::{convert, storage::Owned, Complex, Matrix2, Vector2, VectorN, U2};
/// use nalgebra::ComplexField;
///
/// struct ExampleProblem<F: ComplexField> {
///     p: Vector2<F>,
/// }
///
/// // Implement LeastSquaresProblem to be usable with complex numbers
/// impl<F: ComplexField> LeastSquaresProblem<F, U2, U2> for ExampleProblem<F> {
///     // ...
/// #     type ParameterStorage = Owned<F, U2>;
/// #     type ResidualStorage = Owned<F, U2>;
/// #     type JacobianStorage = Owned<F, U2, U2>;
/// #
/// #     fn set_params(&mut self, p: &VectorN<F, U2>) {
/// #         self.p.copy_from(p);
/// #     }
/// #
/// #     fn residuals(&self) -> Option<Vector2<F>> {
/// #         Some(Vector2::new(
/// #             self.p.x * self.p.x + self.p.y - convert(11.0),
/// #             self.p.x + self.p.y * self.p.y - convert(7.0),
/// #         ))
/// #     }
/// #
/// #     fn jacobian(&self) -> Option<Matrix2<F>> {
/// #         let two: F = convert(2.);
/// #         Some(Matrix2::new(
/// #             two * self.p.x,
/// #             F::one(),
/// #             F::one(),
/// #             two * self.p.y,
/// #         ))
/// #     }
/// }
///
/// // parameters for which you want to test your derivative
/// let x = Vector2::new(0.03877264483558185, -0.7734472300384164);
///
/// // instantiate f64 variant to compute the derivative we want to check
/// let jacobian_from_trait = {
///     let mut problem = ExampleProblem::<f64> {
///         p: Vector2::zeros(),
///     };
///     problem.set_params(&x);
///     problem.jacobian().unwrap()
/// };
///
/// // then use Complex<f64> and compute the numerical derivative
/// let jacobian_numerically = {
///     let mut problem = ExampleProblem::<Complex<f64>> {
///         p: Vector2::zeros(),
///     };
///     differentiate_holomorphic_numerically(&x, &mut problem).unwrap()
/// };
///
/// assert_relative_eq!(jacobian_from_trait, jacobian_numerically, epsilon = 1e-15);
/// ```
pub fn differentiate_holomorphic_numerically<F, N, M, O>(
    params: &VectorN<F, N>,
    problem: &mut O,
) -> Option<MatrixMN<F, M, N>>
where
    F: RealField,
    N: Dim,
    M: Dim,
    O: LeastSquaresProblem<Complex<F>, N, M>,
    DefaultAllocator: Allocator<Complex<F>, N, Buffer = O::ParameterStorage>
        + Allocator<F, N>
        + Allocator<F, M, N>,
{
    let n = params.data.shape().0;
    let mut params = Vector::<Complex<F>, N, O::ParameterStorage>::from_iterator_generic(
        n,
        U1,
        params.iter().map(|x| Complex::<F>::from_real(*x)),
    );
    let m = problem.residuals()?.data.shape().0;
    let mut jacobian = MatrixMN::<F, M, N>::zeros_generic(m, n);
    for i in 0..n.value() {
        let xi = params[i];
        let h = Complex::<F>::from_real(F::default_epsilon()) * xi.abs();
        params[i] = xi + Complex::<F>::i() * h;
        problem.set_params(&params);
        let mut residuals = problem.residuals()?;
        residuals /= h;
        for (dst, src) in jacobian.column_mut(i).iter_mut().zip(residuals.iter()) {
            *dst = src.imaginary();
        }
        params[i] = xi;
    }
    Some(jacobian)
}

#[inline]
pub(crate) fn enorm<F, N, VS>(v: &Vector<F, N, VS>) -> F
where
    F: nalgebra::RealField + Float,
    N: Dim,
    VS: Storage<F, N, U1>,
{
    let mut s1 = F::zero();
    let mut s2 = F::zero();
    let mut s3 = F::zero();
    let mut x1max = F::zero();
    let mut x3max = F::zero();
    let agiant = Float::sqrt(<F as Float>::max_value()) / convert(v.nrows() as f64);
    let rdwarf = Float::sqrt(<F as Float>::min_positive_value());
    for xi in v.iter() {
        let xabs = xi.abs();
        if xabs.is_nan() {
            return xabs;
        }
        if xabs >= agiant || xabs <= rdwarf {
            if xabs > rdwarf {
                // sum for large components
                if xabs > x1max {
                    s1 = F::one() + s1 * Float::powi(x1max / xabs, 2);
                    x1max = xabs;
                } else {
                    s1 += Float::powi(xabs / x1max, 2);
                }
            } else {
                // sum for small components
                if xabs > x3max {
                    s3 = F::one() + s3 * Float::powi(x3max / xabs, 2);
                    x3max = xabs;
                } else if xabs != F::zero() {
                    s3 += Float::powi(xabs / x3max, 2);
                }
            }
        } else {
            s2 += xabs * xabs;
        }
    }

    if !s1.is_zero() {
        x1max * Float::sqrt(s1 + (s2 / x1max) / x1max)
    } else if !s2.is_zero() {
        Float::sqrt(if s2 >= x3max {
            s2 * (F::one() + (x3max / s2) * (x3max * s3))
        } else {
            x3max * ((s2 / x3max) + (x3max * s3))
        })
    } else {
        x3max * Float::sqrt(s3)
    }
}

#[inline]
pub(crate) fn dot<F, N, AS, BS>(a: &Vector<F, N, AS>, b: &Vector<F, N, BS>) -> F
where
    F: nalgebra::RealField,
    N: Dim,
    AS: Storage<F, N, U1>,
    BS: Storage<F, N, U1>,
{
    let mut dot = F::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x * *y;
    }
    dot
}

#[allow(dead_code)]
/// Debug helper to inspect the binary representation of  a `f64` or `f32`.
pub(crate) fn float_repr<F: Float>(f: F) -> String {
    assert!(F::one() / (F::one() + F::one()) != F::zero());
    let bytes = core::mem::size_of::<F>();
    let mut out;
    if bytes == 8 {
        out = String::with_capacity((8 * 2 + 8 - 1) + 27 + 3);
        let f = *unsafe { core::mem::transmute::<_, &f64>(&f) };
        let as_int: u64 = unsafe { core::mem::transmute(f) };
        for i in (0..bytes).rev() {
            out += &format!(
                "{:02x}{}",
                as_int >> 8 * i & 0xFF,
                if i == 0 { "" } else { ":" }
            );
        }
        out += &format!(" ({:+.20E})", f);
    } else if bytes == 4 {
        out = String::with_capacity((4 * 2 + 4 - 1) + 17 + 3);
        let f = *unsafe { core::mem::transmute::<_, &f32>(&f) };
        let as_int: u32 = unsafe { core::mem::transmute(f) };
        for i in (0..bytes).rev() {
            print!(
                "{:02x}{}",
                as_int >> 8 * i & 0xFF,
                if i == 0 { "" } else { ":" }
            );
        }
        println!(" ({:.10E})", f);
    } else {
        unimplemented!()
    }
    out
}

// #[test]
// fn test_linear_case() {
//     use crate::lm::test_examples::LinearFullRank;
//     use approx::assert_relative_eq;
//     use nalgebra::U5;
//     let mut x = VectorN::<f64, U5>::from_element(1.);
//     x[2] = -10.;
//     let mut problem = LinearFullRank {
//         params: x.clone(),
//         m: 6,
//     };
//     let jac_num = differentiate_numerically(x, &mut problem).unwrap();
//     let jac_trait = problem.jacobian().unwrap();
//     assert_relative_eq!(jac_num, jac_trait);
// }
