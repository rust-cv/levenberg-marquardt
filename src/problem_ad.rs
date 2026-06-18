use nalgebra::{
    ComplexField, DefaultAllocator, Dim, Matrix, OMatrix, OVector, Owned, U1, Vector,
    allocator::Allocator,
};
use num_dual::{DualNum, DualNumFloat, jacobian};

use crate::LeastSquaresProblem;

/// A least squares minimization problem.
///
/// This is what [`LevenbergMarquardt`](struct.LevenbergMarquardt.html) needs
/// to compute the residuals and the Jacobian. See the [module documentation](index.html)
/// for a usage example.
pub trait LeastSquaresProblemAD<F, M, N>
where
    F: ComplexField + Copy,
    N: Dim,
    M: Dim,
    DefaultAllocator: Allocator<N> + Allocator<M>,
{
    /// Compute the residual vector.
    fn residuals<D: DualNum<F>>(&self, x: OVector<D, N>) -> Option<OVector<D, M>>
    where
        DefaultAllocator: Allocator<M> + Allocator<N>;
}

pub struct ADWrapper<T, F, M, N>
where
    F: ComplexField + Copy,
    M: Dim,
    N: Dim,
    DefaultAllocator: Allocator<M> + Allocator<N> + Allocator<M, N>,
{
    problem: T,
    parameters: OVector<F, N>,
    residuals: Option<(OVector<F, M>, OMatrix<F, M, N>)>,
}

impl<T, F, M, N> ADWrapper<T, F, M, N>
where
    T: LeastSquaresProblemAD<F, M, N>,
    F: ComplexField + Copy + DualNum<F> + DualNumFloat,
    M: Dim,
    N: Dim,
    DefaultAllocator: Allocator<M> + Allocator<U1, N> + Allocator<N> + Allocator<M, N>,
{
    pub fn new(problem: T, parameters: OVector<F, N>) -> Self {
        let residuals = jacobian(|x| problem.residuals(x), &parameters);
        Self {
            problem,
            parameters,
            residuals,
        }
    }
}

impl<
    T: LeastSquaresProblemAD<F, M, N>,
    F: ComplexField + Copy + DualNum<F> + DualNumFloat,
    M: Dim,
    N: Dim,
> LeastSquaresProblem<F, M, N> for ADWrapper<T, F, M, N>
where
    DefaultAllocator: Allocator<M> + Allocator<U1, N> + Allocator<N> + Allocator<M, N>,
{
    type ResidualStorage = Owned<F, M>;

    type JacobianStorage = Owned<F, M, N>;

    type ParameterStorage = Owned<F, N>;

    fn set_params(&mut self, x: &OVector<F, N>) {
        self.parameters.copy_from(x);
        self.residuals = jacobian(|x| self.problem.residuals(x), x)
    }

    fn params(&self) -> Vector<F, N, Self::ParameterStorage> {
        self.parameters.clone()
    }

    fn residuals(&self) -> Option<Vector<F, M, Self::ResidualStorage>> {
        self.residuals.as_ref().map(|(res, _)| res.clone())
    }

    fn jacobian(&self) -> Option<Matrix<F, M, N, Self::JacobianStorage>> {
        self.residuals.as_ref().map(|(_, jac)| jac.clone())
    }
}
