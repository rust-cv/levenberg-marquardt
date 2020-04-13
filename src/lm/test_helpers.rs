use alloc::{vec, vec::Vec};
use core::cell::RefCell;

use nalgebra::{
    allocator::Allocator, storage::Owned, DefaultAllocator, DimName, MatrixMN, VectorN,
};

use crate::LeastSquaresProblem;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockCall {
    SetParams,
    Residuals,
    Jacobian,
}

#[derive(Clone)]
pub struct MockProblem<N: DimName, M: DimName>
where
    DefaultAllocator: Allocator<f64, N> + Allocator<f64, M> + Allocator<f64, M, N>,
{
    call_history: RefCell<Vec<MockCall>>,
    params: Vec<VectorN<f64, N>>,
    residuals: Vec<Option<VectorN<f64, M>>>,
    residuals_index: RefCell<usize>,
    jacobians: Vec<Option<MatrixMN<f64, M, N>>>,
    jacobians_index: RefCell<usize>,
}

impl<N: DimName, M: DimName> MockProblem<N, M>
where
    DefaultAllocator: Allocator<f64, N> + Allocator<f64, M> + Allocator<f64, M, N>,
{
    pub fn new(residuals: Vec<Option<VectorN<f64, M>>>) -> Self {
        Self {
            residuals,
            jacobians: vec![],
            call_history: RefCell::new(vec![]),
            params: vec![],
            residuals_index: RefCell::new(0),
            jacobians_index: RefCell::new(0),
        }
    }

    pub fn calls(&mut self) -> &[MockCall] {
        self.call_history.get_mut().as_slice()
    }
}

impl<N: DimName, M: DimName> LeastSquaresProblem<f64, N, M> for MockProblem<N, M>
where
    DefaultAllocator: Allocator<f64, N> + Allocator<f64, M> + Allocator<f64, M, N>,
{
    type ResidualStorage = Owned<f64, M>;
    type ParameterStorage = Owned<f64, N>;
    type JacobianStorage = Owned<f64, M, N>;

    fn set_params(&mut self, params: &VectorN<f64, N>) {
        self.params.push(params.clone());
        self.call_history.borrow_mut().push(MockCall::SetParams);
    }

    fn residuals(&self) -> Option<VectorN<f64, M>> {
        self.call_history.borrow_mut().push(MockCall::Residuals);
        if *self.residuals_index.borrow() < self.residuals.len() {
            *self.residuals_index.borrow_mut() += 1;
            self.residuals[*self.residuals_index.borrow() - 1].clone()
        } else {
            None
        }
    }

    fn jacobian(&self) -> Option<MatrixMN<f64, M, N>> {
        self.call_history.borrow_mut().push(MockCall::Jacobian);
        if *self.jacobians_index.borrow() < self.jacobians.len() {
            *self.jacobians_index.borrow_mut() += 1;
            self.jacobians[*self.jacobians_index.borrow() - 1].clone()
        } else {
            None
        }
    }
}
