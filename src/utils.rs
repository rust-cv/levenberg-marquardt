use crate::LeastSquaresProblem;
use nalgebra::{
    allocator::Allocator, convert, storage::Storage, DefaultAllocator, Dim, Matrix, RealField,
    Vector,
};
use num_traits::float::Float;

/// Compute a [numerical approximation](https://en.wikipedia.org/wiki/Numerical_differentiation)
/// to the Jacobian for testing.
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
/// # Example
///
/// You can use this function to check your derivative implementation in a unit test.
/// For example:
///
/// ```ignore
/// // assume `problem` is an instance of `LeastSquaresProblem`
/// // and `x` a point at which you want to check your derivative.
/// problem.set_params(&x);
/// let jacobian_expected = problem.jacobian().unwrap();
/// let jacobian_out = differentiate_numerically(x.clone(), &mut problem).unwrap();
/// assert_abs_diff_eq!(jacobian_out, jacobian_expected, epsilon = 1e-7);
/// ```
///
/// The `assert_abs_diff_eq!` macro is from the `approx` crate.
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
    for i in 0..n.value() {
        quotient.fill(F::zero());
        let x0 = params[i];
        let h = Float::sqrt(F::default_epsilon()) * x0;
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

#[cfg(test)]
mod tests {
    use super::differentiate_numerically;
    use crate::LeastSquaresProblem;
    use approx::assert_abs_diff_eq;
    use nalgebra::{storage::Owned, Matrix2, Vector2, VectorN, U2};

    struct ExampleProblem {
        p: Vector2<f64>,
    }

    impl LeastSquaresProblem<f64, U2, U2> for ExampleProblem {
        type ParameterStorage = Owned<f64, U2>;
        type ResidualStorage = Owned<f64, U2>;
        type JacobianStorage = Owned<f64, U2, U2>;

        fn set_params(&mut self, p: &VectorN<f64, U2>) {
            self.p.copy_from(p);
        }

        fn residuals(&self) -> Option<Vector2<f64>> {
            Some(Vector2::new(
                self.p.x * self.p.x + self.p.y - 11.0,
                self.p.x + self.p.y * self.p.y - 7.0,
            ))
        }

        fn jacobian(&self) -> Option<Matrix2<f64>> {
            Some(Matrix2::new(2.0 * self.p.x, 1.0, 1.0, 2.0 * self.p.y))
        }
    }

    #[test]
    fn test_numerical_differentiation() {
        let x = Vector2::new(0.615604610984124, 0.733034404976318);
        let mut problem = ExampleProblem {
            p: Vector2::zeros(),
        };
        problem.set_params(&x);
        let jacobian_expected = problem.jacobian().unwrap();
        let jacobian_out = differentiate_numerically(x.clone(), &mut problem).unwrap();
        assert_abs_diff_eq!(jacobian_out, jacobian_expected, epsilon = 1e-7);
    }
}
