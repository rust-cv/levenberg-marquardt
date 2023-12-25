use crate::{LeastSquaresProblem, LevenbergMarquardt, MinimizationReport};
use nalgebra::{ArrayStorage, Const, DVector, Dim, Dyn, Matrix, VecStorage, Vector};
use num_traits::Float;

/// A convenience trait to easily run [`LevenbergMarquardt::minimize`] for a given equation.
///
/// For example:
///
/// ```
/// # use approx::assert_relative_eq;
/// use levenberg_marquardt::Equation;
///
/// struct Problem;
///
/// impl Equation<2, f64> for Problem {
///     fn equation(&self, ws: &[f64; 2], x: f64) -> f64 {
///         // This is our equation; we want to find the coefficients `ws`.
///         ws[0] * 2.0 * x + ws[1] * 0.5 * x.powi(2)
///     }
///
///     fn derivatives(&self, ws: &[f64; 2], x: f64) -> [f64; 2] {
///         // These are the partial derivatives of our equation, one for each coefficient.
///         [
///             2.0 * x,
///             0.5 * x.powi(2),
///         ]
///     }
/// }
///
/// // Let's pick some coefficients...
/// let ws = [1.33, 0.66];
///
/// // ...and generate some data...
/// let xs = [1.0, 10.0, 100.0];
/// let ys = xs.map(|x| Problem.equation(&ws, x));
///
/// // Now we can run the LM algorithm to calculate the coefficients from the data.
/// let ([w0, w1], _) = Problem.least_squares_fit(&xs, &ys, [1.5, 1.0]);
///
/// // They're the same as what we've picked!
/// assert_relative_eq!(w0, 1.33);
/// assert_relative_eq!(w1, 0.66);
/// ```
pub trait Equation<const N_PARAMS: usize, T> {
    /// The equation for which we want to find the coefficients `ws`.
    fn equation(&self, ws: &[T; N_PARAMS], x: T) -> T;

    /// The partial derivatives of the equation for which we want to find the coefficients `ws`.
    fn derivatives(&self, ws: &[T; N_PARAMS], x: T) -> [T; N_PARAMS];

    /// Transforms this equation into a [`LeastSquaresProblem`].
    ///
    /// **This will panic if `xs` and `ys` are not of the same length!**
    fn as_least_squares_problem<'a>(
        &'a self,
        xs: &'a [T],
        ys: &'a [T],
        initial_guess: [T; N_PARAMS],
    ) -> impl LeastSquaresProblem<T, Dyn, Const<N_PARAMS>> + Into<[T; N_PARAMS]> + 'a
    where
        T: nalgebra::RealField + nalgebra::ComplexField + Copy + Float,
        Self: 'a,
    {
        struct State<'a, const N_PARAMS: usize, T, E: ?Sized + Equation<N_PARAMS, T>> {
            itself: &'a E,
            xs: &'a [T],
            ys: &'a [T],
            ws: [T; N_PARAMS],
        }

        impl<'a, const N_PARAMS: usize, T, E> From<State<'a, N_PARAMS, T, E>> for [T; N_PARAMS]
        where
            T: nalgebra::RealField + nalgebra::ComplexField + Copy,
            E: ?Sized + Equation<N_PARAMS, T>,
        {
            fn from(problem: State<'a, N_PARAMS, T, E>) -> [T; N_PARAMS] {
                problem.ws
            }
        }

        impl<'a, const N_PARAMS: usize, T, E> LeastSquaresProblem<T, Dyn, Const<N_PARAMS>>
            for State<'a, N_PARAMS, T, E>
        where
            T: nalgebra::RealField + nalgebra::ComplexField + Copy,
            E: ?Sized + Equation<N_PARAMS, T>,
        {
            type ParameterStorage = ArrayStorage<T, N_PARAMS, 1>;
            type ResidualStorage = VecStorage<T, Dyn, Const<1>>;
            type JacobianStorage = VecStorage<T, Dyn, Const<N_PARAMS>>;

            fn set_params(&mut self, p: &Vector<T, Const<N_PARAMS>, Self::ParameterStorage>) {
                self.ws.copy_from_slice(p.as_slice());
            }

            fn params(&self) -> Vector<T, Const<N_PARAMS>, Self::ParameterStorage> {
                Vector::from_data(ArrayStorage([self.ws; 1]))
            }

            fn residuals(&self) -> Option<DVector<T>> {
                assert_eq!(self.xs.len(), self.ys.len());
                let ws = &self.ws;
                Some(DVector::from_data(VecStorage::new(
                    Dim::from_usize(self.xs.len()),
                    Const::<1>,
                    self.xs
                        .iter()
                        .zip(self.ys.iter())
                        .map(|(&x, &y)| y - self.itself.equation(ws, x))
                        .collect(),
                )))
            }

            fn jacobian(&self) -> Option<Matrix<T, Dyn, Const<N_PARAMS>, Self::JacobianStorage>> {
                let ws = &self.ws;
                let mut jacobian =
                    Matrix::zeros_generic(Dyn::from_usize(self.xs.len()), Const::<N_PARAMS>);
                for (i, &x) in self.xs.iter().enumerate() {
                    let derivatives = self.itself.derivatives(ws, x);
                    for n in 0..derivatives.len() {
                        jacobian[(i, n)] = -T::one() * derivatives[n];
                    }
                }

                Some(jacobian)
            }
        }

        assert_eq!(xs.len(), ys.len());
        State::<N_PARAMS, T, Self> {
            itself: self,
            xs,
            ys,
            ws: initial_guess,
        }
    }

    /// A convenience function to directly run the optimization procedure on the equation and return the calculated coefficients.
    ///
    /// Equivalent to the following code:
    ///
    /// ```
    /// # use levenberg_marquardt::{Equation, LevenbergMarquardt};
    /// # struct Problem;
    /// # impl Equation<2, f64> for Problem {
    /// #     fn equation(&self, ws: &[f64; 2], x: f64) -> f64 { unimplemented!() }
    /// #     fn derivatives(&self, ws: &[f64; 2], x: f64) -> [f64; 2] { unimplemented!() }
    /// # }
    /// # fn dummy() {
    /// # let xs = unimplemented!();
    /// # let ys = unimplemented!();
    /// # let initial_guess = unimplemented!();
    /// let problem = Problem.as_least_squares_problem(xs, ys, initial_guess);
    /// let (result, report) = LevenbergMarquardt::new().minimize(problem);
    /// let result = result.into(); // Convert the result into an array `[T; N_PARAMS]`.
    /// # }
    /// ```
    fn least_squares_fit(
        &self,
        xs: &[T],
        ys: &[T],
        initial_guess: [T; N_PARAMS],
    ) -> ([T; N_PARAMS], MinimizationReport<T>)
    where
        T: nalgebra::RealField + nalgebra::ComplexField + Copy + Float,
    {
        let (result, report) = LevenbergMarquardt::new().minimize(self.as_least_squares_problem(
            xs,
            ys,
            initial_guess,
        ));

        (result.into(), report)
    }
}
