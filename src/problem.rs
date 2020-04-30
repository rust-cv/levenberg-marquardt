use nalgebra::{storage::ContiguousStorageMut, ComplexField, Dim, Matrix, Vector};

/// A least squares minimization problem.
///
/// This is what [`LevenbergMarquardt`](struct.LevenbergMarquardt.html) needs
/// to compute the residuals and the Jacobian. See the [module documentation](index.html)
/// for a usage example.
pub trait LeastSquaresProblem<F, M, N>
where
    F: ComplexField,
    N: Dim,
    M: Dim,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage: ContiguousStorageMut<F, M>;
    type JacobianStorage: ContiguousStorageMut<F, M, N>;
    type ParameterStorage: ContiguousStorageMut<F, N> + Clone;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector<F, N, Self::ParameterStorage>);

    /// Compute the residual vector.
    fn residuals(&self) -> Option<Vector<F, M, Self::ResidualStorage>>;

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<Matrix<F, M, N, Self::JacobianStorage>>;
}
