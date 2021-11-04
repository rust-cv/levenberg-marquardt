use nalgebra::{
    storage::{IsContiguous, RawStorageMut, Storage},
    ComplexField, Dim, Matrix, Vector,
};

/// A least squares minimization problem.
///
/// This is what [`LevenbergMarquardt`](struct.LevenbergMarquardt.html) needs
/// to compute the residuals and the Jacobian. See the [module documentation](index.html)
/// for a usage example.
pub trait LeastSquaresProblem<F, M, N>
where
    F: ComplexField + Copy,
    N: Dim,
    M: Dim,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage: RawStorageMut<F, M> + Storage<F, M> + IsContiguous;
    type JacobianStorage: RawStorageMut<F, M, N> + Storage<F, M, N> + IsContiguous;
    type ParameterStorage: RawStorageMut<F, N> + Storage<F, N> + IsContiguous + Clone;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector<F, N, Self::ParameterStorage>);

    /// Get the current parameter vector `$\vec{x}$`.
    fn params(&self) -> Vector<F, N, Self::ParameterStorage>;

    /// Compute the residual vector.
    fn residuals(&self) -> Option<Vector<F, M, Self::ResidualStorage>>;

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<Matrix<F, M, N, Self::JacobianStorage>>;
}
