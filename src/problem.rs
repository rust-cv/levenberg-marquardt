use num_traits::FromPrimitive;
use nalgebra::{
    storage::{ContiguousStorageMut, Storage},
    Dim, Matrix, RealField, Vector,
};

/// A least-squares minimization problem.
///
/// **Important.** The object implementing this is cloned during iteration.
/// The object should not own huge chunks of data. Instead, only store references
/// to it.
///
/// This is what [`LevenbergMarquardt`](struct.LevenbergMarquardt.html) needs
/// to compute the residuals and the Jacobian. See the [module documentation](index.html)
/// for a usage example.
///
/// A type implementing this owns the current set of parameters `$\vec{x}$`. During
/// iteration the parameters are updated ([`apply_parameter_step`](#tymethod.apply_parameter_step))
/// by providing an update vector. The residuals and Jacobians must be computed
/// with respect to the stored parameters.
pub trait LeastSquaresProblem<F, N, M>: Clone
where
    F: RealField + FromPrimitive,
    N: Dim,
    M: Dim,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage: Storage<F, M>;
    type JacobianStorage: Storage<F, M, N>;
    type ParameterStorage: ContiguousStorageMut<F, N> + Clone;

    /// Update the stored parameters `$\vec{x}$` by adding the provided
    /// `delta` vector to it.
    ///
    /// The parameters may be modified after the update by this function, for example to
    /// enforce a constraint.
    fn apply_parameter_step(&mut self, delta: &Vector<F, N, Self::ParameterStorage>);

    /// Compute the residual vector.
    fn residuals(&self) -> Vector<F, M, Self::ResidualStorage>;

    /// Compute the Jacobian for the residual vector.
    ///
    /// # Default implemntation
    ///
    /// The default implementation uses (forward) finite-differences.
    /// However, we strongly recommend to derive the Jacobian symbolically and implement it
    /// directly.
    fn jacobian(&self) -> Matrix<F, M, N, Self::JacobianStorage> {
        todo!()
    }

    /// Evaluate the objective function.
    ///
    /// This is not needed during optimization. It is a convenience
    /// method provided for your pleasure.
    fn objective_function(&self) -> F {
        F::from_f32(0.5).unwrap() * self.residuals().norm_squared()
    }
}