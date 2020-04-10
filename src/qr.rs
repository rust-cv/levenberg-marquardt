//! Pivoted QR factorization and a specialized LLS solver.
//!
//! The QR factorization is used to implement an efficient solver for the
//! linear least-squares problem which is repeatedly required to be
//! solved in the LM algorithm.
use core::cell::RefCell;
use nalgebra::{
    allocator::Allocator,
    convert,
    storage::{ContiguousStorageMut, Storage},
    DefaultAllocator, Dim, DimName, Matrix, MatrixSlice, RealField, Vector, VectorN,
};
use num_traits::FromPrimitive;

/// Erros which can occur using the pivoted QR factorization or the solver.
pub enum Error {
    ShapeConstraintFailed,
}

impl Error {
    pub fn msg(&self) -> &'static str {
        match self {
            Error::ShapeConstraintFailed => "m >= n is not fulfilled",
        }
    }
}

/// Pivoted QR decomposition.
pub struct PivotedQR<F, M, N, S>
where
    F: RealField + FromPrimitive,
    M: Dim,
    N: DimName,
    S: ContiguousStorageMut<F, M, N>,
    DefaultAllocator: Allocator<F, N> + Allocator<usize, N>,
{
    /// The column norms of the input matrix `$\mathbf{A}$`
    column_norms: VectorN<F, N>,
    /// Strictly upper part of `$\mathbf{R}$` and the Householder transformations,
    /// combined in one matrix.
    qr: Matrix<F, M, N, S>,
    /// Diagonal entries of R
    r_diag: VectorN<F, N>,
    /// Permution matrix. Entry `$i$` specifies which column of the identity
    /// matrix to use.
    permutation: VectorN<usize, N>,
    work: VectorN<F, N>,
}

impl<F, M, N, S> PivotedQR<F, M, N, S>
where
    F: RealField + FromPrimitive,
    M: Dim,
    N: DimName,
    S: ContiguousStorageMut<F, M, N>,
    DefaultAllocator: Allocator<F, N> + Allocator<F, N> + Allocator<usize, N>,
{
    /// Create a pivoted QR decomposition of a matrix `$\mathbf{A}\in\R^{m\times n}$`
    /// with `$m \geq n$`.
    ///
    /// # Errors
    ///
    /// Only returns `Err` when `$m < n$`.
    pub fn new(mut a: Matrix<F, M, N, S>) -> Result<Self, Error>
where {
        // The implementation is based more or less on LAPACK's "xGEQPF"
        let n = a.ncols();
        if a.nrows() < n {
            return Err(Error::ShapeConstraintFailed);
        }
        let column_norms = VectorN::<F, N>::from_iterator(a.column_iter().map(|c| c.norm()));
        let mut r_diag = column_norms.clone();
        let mut work = column_norms.clone();
        let mut permutation = VectorN::<usize, N>::from_iterator(0..);
        for j in 0..n {
            // pivot
            {
                let kmax = r_diag.slice_range(j.., ..).imax() + j;
                a.swap_columns(j, kmax);
                permutation.swap_rows(j, kmax);
                r_diag[kmax] = r_diag[j];
                work[kmax] = work[j];
            }
            // compute Householder reflection vector w_j to
            // reduce the j-th column
            let mut lower = a.rows_range_mut(j..);
            let (left, mut right) = lower.columns_range_pair_mut(j, j + 1..);
            let w_j = {
                let mut axis = left;
                let mut aj_norm = axis.norm();
                if aj_norm == F::zero() {
                    r_diag[j] = F::zero();
                    continue;
                }
                if axis[0].is_negative() {
                    aj_norm = -aj_norm;
                }
                r_diag[j] = -aj_norm;
                axis.unscale_mut(aj_norm);
                axis[0] += F::one();
                axis
            };
            // apply reflection to remaining rows
            for (mut k, mut col) in right.column_iter_mut().enumerate() {
                let temp = {
                    let sum = col.dot(&w_j);
                    sum / w_j[0]
                };
                col.axpy(-temp, &w_j, F::one());
                // update partial column norms
                // see "Lapack Working Note 176"
                k += j + 1;
                if r_diag[k] == F::zero() {
                    continue;
                }
                let temp = col[0] / r_diag[k];
                let temp = if temp.abs() < F::one() {
                    r_diag[k] *= (F::one() - temp * temp).sqrt();
                    r_diag[k] / work[k]
                } else {
                    F::zero()
                };
                let z005: F = convert(0.05f64);
                if temp.abs() == F::zero() || z005 * (temp * temp) <= F::default_epsilon() {
                    r_diag[k] = col.slice_range(1.., ..).norm();
                    work[k] = r_diag[k];
                }
            }
        }
        Ok(Self {
            column_norms,
            qr: a,
            permutation,
            r_diag,
            work,
        })
    }

    /// Consume the QR-decomposition and transform it into
    /// a parametrized least-squares problem.
    ///
    /// See [`LinearLeastSquaresDiagonalProblem`](struct.LinearLeastSquaresDiagonalProblem.html)
    /// for details.
    pub fn into_least_squares_diagonal_problem<QS>(
        self,
        mut b: Vector<F, M, QS>,
    ) -> LinearLeastSquaresDiagonalProblem<F, M, N, S>
    where
        QS: ContiguousStorageMut<F, M>,
    {
        // compute first n-entries of Q^T * b
        let mut qt_b = VectorN::<F, N>::from_column_slice(b.as_slice());
        let n = self.qr.ncols();
        for j in 0..n {
            let axis = self.qr.slice_range(j.., j);
            if axis[0] != F::zero() {
                let temp = b.rows_range(j..).dot(&axis) / axis[0];
                b.rows_range_mut(j..).axpy(-temp, &axis, F::one());
            }
            qt_b[j] = b[j];
        }
        let mut upper_r = self.qr;
        // reflect upper triangular part. This enables for
        // a nicer memory acces pattern in `copy_r_down`.
        for i in 0..n {
            for j in (i + 1..(n - i)).rev() {
                let tmp = upper_r[(j - 1, n - 1 - i)];
                upper_r[(j - 1, n - 1 - i)] = upper_r[(i, j)];
                upper_r[(i, j)] = tmp;
            }
        }
        LinearLeastSquaresDiagonalProblem {
            qt_b,
            upper_r,
            r_diag: self.r_diag,
            permutation: self.permutation,
            work: RefCell::new(self.work),
        }
    }
}

/// Parametrized linear least-squares problem for the LM algorithm.
///
/// The problem is of the form
/// ```math
///   \min_{\vec{x}\in\R^n}\frac{1}{2}\Bigl\|
///     \begin{bmatrix}
///        \mathbf{A} \\
///        \mathbf{D}
///     \end{bmatrix}\vec{x} -
///     \begin{bmatrix}
///         \vec{b} \\
///         \vec{0}
///     \end{bmatrix}
///   \Bigr\|^2,
/// ```
/// for a matrix `$\mathbf{A}\in\R^{m \times n}$`, diagonal matrix
/// `$\mathbf{D}\in\R^n$` and vector `$\vec{b}\in\R^m$`.
/// Everything except the diagonal matrix `$\mathbf{D}$` is considered
/// fixed.
///
/// The problem can be efficiently solved for a sequence of diagonal
/// matrices `$\mathbf{D}$`.
///
/// You must create an instance of this by first computing a pivotized
/// QR decomposition of `$\mathbf{A}$`, then use
/// [`into_least_squares_diagonal_problem`](struct.PivotedQR.html#into_least_squares_diagonal_problem).
pub struct LinearLeastSquaresDiagonalProblem<F, M, N, S>
where
    F: RealField,
    M: Dim,
    N: DimName,
    S: ContiguousStorageMut<F, M, N>,
    DefaultAllocator: Allocator<F, N> + Allocator<usize, N>,
{
    /// The first `$n$` entries of `$\mathbf{Q}^\top \vec{b}$`.
    qt_b: VectorN<F, N>,
    /// Strictly upper part of `$\mathbf{R}$`, also used to store `$\mathbf{L}$`.
    upper_r: Matrix<F, M, N, S>,
    /// Diagonal entries of `$\mathbf{R}$`.
    r_diag: VectorN<F, N>,
    /// Permution matrix. Entry `$i$` specifies which column of the identity
    /// matrix to use.
    permutation: VectorN<usize, N>,
    work: RefCell<VectorN<F, N>>,
}

impl<F, M, N, S> LinearLeastSquaresDiagonalProblem<F, M, N, S>
where
    F: RealField,
    M: Dim,
    N: DimName,
    S: ContiguousStorageMut<F, M, N>,
    DefaultAllocator: Allocator<F, N> + Allocator<usize, N>,
{
    /// Solve the linear least squares problem
    /// for a diagonal matrix `$\mathbf{D}$` (`diag`).
    ///
    /// This is equivalent to solving
    /// ```math
    /// (\mathbf{A}^\top\mathbf{A} + \mathbf{D}\mathbf{D})\vec{x} = \mathbf{A}^\top \vec{b}.
    /// ```
    ///
    /// # Return value
    ///
    /// Returns the solution `$\vec{x}$` and a reference to a lower triangular
    /// matrix `$\mathbf{L}\in\R^{n\times n}$` with
    /// ```math
    ///   \mathbf{P}^\top(\mathbf{A}^\top\mathbf{A} + \mathbf{D}\mathbf{D})\mathbf{P} = \mathbf{L}\mathbf{L}^\top.
    /// ```
    pub fn solve_with_diagonal<DS>(
        &mut self,
        diag: &Vector<F, N, DS>,
    ) -> (VectorN<F, N>, MatrixSlice<F, N, N, S::RStride, S::CStride>)
    where
        DS: Storage<F, N>,
    {
        let mut rhs = self.eliminate_diag(diag, self.qt_b.clone());
        self.solve_after_elimination(&mut rhs)
    }

    /// Solve the least-squares problem with a zero diagonal.
    pub fn solve_with_zero_diagonal(
        &mut self,
    ) -> (VectorN<F, N>, MatrixSlice<F, N, N, S::RStride, S::CStride>) {
        self.copy_r_down();
        let mut rhs = self.work.borrow_mut();
        rhs.copy_from(&self.qt_b);
        self.solve_after_elimination(&mut rhs)
    }

    pub fn rank(&self) -> usize {
        let n = self.r_diag.nrows();
        (0..n).find(|j| self.r_diag[*j] == F::zero()).unwrap_or(n)
    }

    pub fn has_full_rank(&self) -> bool {
        self.r_diag.iter().all(|x| *x != F::zero())
    }

    fn solve_after_elimination(
        &self,
        rhs: &mut VectorN<F, N>,
    ) -> (VectorN<F, N>, MatrixSlice<F, N, N, S::RStride, S::CStride>) {
        let n = self.upper_r.data.shape().1;
        let l = self.upper_r.generic_slice((0, 0), (n, n));

        // check for singular matrix
        let rank = self.rank();
        rhs.rows_range_mut(rank..).fill(F::zero());

        // solve
        l.slice_range(..rank, ..rank)
            .tr_solve_lower_triangular_mut(&mut rhs.rows_range_mut(..rank));

        let mut x = VectorN::<F, N>::zeros();
        for j in 0..n.value() {
            x[self.permutation[j]] = rhs[j];
        }
        (x, l)
    }

    fn eliminate_diag<DS>(
        &mut self,
        diag: &Vector<F, N, DS>,
        mut rhs: VectorN<F, N>,
    ) -> VectorN<F, N>
    where
        DS: Storage<F, N>,
    {
        self.copy_r_down();
        let l_diag = self.work.get_mut();
        // only lower triangular part is used which was filled with R^T by
        // `copy_r_down`. This part is then iteratively overwritten with L.
        let r_and_l = &mut self.upper_r;
        let n = diag.nrows();
        // eliminate the diagonal entries from D using Givens rotations
        for j in 0..n {
            if diag[self.permutation[j]] == F::zero() {
                continue;
            }
            l_diag[j] = diag[self.permutation[j]];
            l_diag.rows_range_mut(j + 1..).fill(F::zero());

            let mut qtbpj = F::zero();
            for k in j..n {
                if l_diag[k] == F::zero() {
                    continue;
                }
                // determine the Givens rotation
                let (sin, cos) = if r_and_l[(k, k)].abs() < l_diag[k].abs() {
                    let cot = r_and_l[(k, k)] / l_diag[k];
                    let sin = F::one() / (F::one() + cot * cot).sqrt();
                    let cos = sin * cot;
                    (sin, cos)
                } else {
                    let tan = l_diag[k] / r_and_l[(k, k)];
                    let cos = F::one() / (F::one() + tan * tan).sqrt();
                    let sin = cos * tan;
                    (sin, cos)
                };
                // compute the modified diagonal element of R and (Q^T*b,0)
                r_and_l[(k, k)] = cos * r_and_l[(k, k)] + sin * l_diag[k];
                let temp = cos * rhs[k] + sin * qtbpj;
                qtbpj = -sin * rhs[k] + cos * qtbpj;
                rhs[k] = temp;

                // accumulate the transformation in the row of U
                for i in k + 1..n {
                    let temp = cos * r_and_l[(i, k)] + sin * l_diag[i];
                    l_diag[i] = -sin * r_and_l[(i, k)] + cos * l_diag[i];
                    r_and_l[(i, k)] = temp;
                }
            }
        }
        rhs
    }

    fn copy_r_down(&mut self) {
        let r = &mut self.upper_r;
        let n = r.ncols();
        for j in 0..n {
            r[(j, j)] = self.r_diag[j];
            for i in j + 1..n {
                r[(i, j)] = r[(i - (j + 1), (n - 1) - j)];
            }
        }
    }
}

#[test]
fn test_pivoted_qr() {
    // Reference data was generated using the implementation from the library
    // "lmfit".
    // Also, the values were checked with SciPy's "qr" method.
    use nalgebra::{Matrix4x3, Vector3};
    let a = Matrix4x3::<f64>::from_iterator((0..).map(|i| i as f64));
    let qr = PivotedQR::new(a).ok().unwrap();

    assert_eq!(qr.permutation, nalgebra::Vector3::new(2, 0, 1));

    let column_norms = Vector3::new(3.7416574, 11.2249722, 19.1311265);
    assert!((qr.column_norms - column_norms).norm() < 1e-7);

    let r_diag = Vector3::new(-19.1311265, 1.8700983, 0.0);
    assert!((qr.r_diag - r_diag).norm() < 1e-7);

    let qr_ref = Matrix4x3::<f64>::from_iterator(
        [
            1.4181667,
            0.4704375,
            0.5227084,
            0.5749792,
            -3.2407919,
            1.0401278,
            -0.4307302,
            -0.9015882,
            -11.1859592,
            0.9350492,
            1.7310553,
            0.6823183,
        ]
        .iter()
        .map(|x| *x),
    );
    assert!((qr.qr - qr_ref).norm() < 1e-7);
}

#[test]
fn test_pivoted_qr_more_branches() {
    // This test case was crafted to hit all three
    // branches of the partial column norms
    use nalgebra::{Matrix4x3, Vector3};
    let a = Matrix4x3::<f64>::from_iterator(
        [
            30.0, 43.0, 34.0, 26.0, 30.0, 43.0, 34.0, 26.0, 24.0, 39.0, -10.0, -34.0,
        ]
        .iter()
        .map(|x| *x),
    );
    let qr = PivotedQR::new(a).ok().unwrap();
    let r_diag = Vector3::new(-67.683085036070864, -55.250741178610944, 0.00000000000001);
    assert!((qr.r_diag - r_diag).norm() < 1e-10);
}

#[cfg(test)]
fn default_lls(
    case: usize,
) -> LinearLeastSquaresDiagonalProblem<
    f64,
    nalgebra::U4,
    nalgebra::U3,
    nalgebra::storage::Owned<f64, nalgebra::U4, nalgebra::U3>,
> {
    use nalgebra::{Matrix4x3, Vector4};
    let a = match case {
        1 => Matrix4x3::<f64>::from_iterator((0..).map(|i| i as f64)),
        2 => Matrix4x3::<f64>::from_iterator(
            [30., 43., 34., 26., 30., 43., 34., 26., 24., 39., -10., -34.]
                .iter()
                .map(|x| *x),
        ),
        3 => Matrix4x3::new(1., 2., -1., 0., 1., 4., 0., 0., 0.5, 0., 0., 0.),
        _ => unimplemented!(),
    };
    let qr = PivotedQR::new(a).ok().unwrap();
    qr.into_least_squares_diagonal_problem(Vector4::new(1.0, 2.0, 5.0, 4.0))
}

#[test]
fn test_into_lls() {
    use nalgebra::Vector3;
    let lls = default_lls(1);
    let qt_b = Vector3::new(-6.272500481871799, 1.963603245291175, -0.288494026015405);
    assert!((lls.qt_b - qt_b).norm() < 1e-10);
}

#[test]
fn test_elimate_diag_and_l() {
    use nalgebra::{Matrix3, Vector3};
    let mut lls = default_lls(1);
    let rhs = lls.eliminate_diag(&Vector3::new(1.0, 0.5, 0.0), lls.qt_b.clone());
    let rhs_ref = Vector3::new(-6.272500481871799, 1.731584982206922, 0.612416936078506);
    assert!((rhs - rhs_ref).norm() < 1e-10);

    // contains L
    let r_ref = Matrix3::new(
        -19.131126469708992,
        0.935049164424371,
        -3.240791915633763,
        -3.240791915633763,
        2.120676250530203,
        -11.185959192671376,
        -11.185959192671376,
        0.824564277241393,
        0.666641352293790,
    );
    let r = lls.upper_r.slice_range(..3, ..3);
    assert!((r - r_ref).norm() < 1e-10);
}

#[test]
fn test_lls_x() {
    use nalgebra::Vector3;
    let mut lls = default_lls(1);
    let (x_out, _) = lls.solve_with_diagonal(&Vector3::new(1.0, 0.5, 0.0));
    let x_ref = Vector3::new(0.459330143540669, 0.918660287081341, -0.287081339712919);
    assert!((x_out - x_ref).norm() < 1e-10);
}

#[test]
fn test_standard_lls_case() {
    use nalgebra::Vector3;
    let mut lls = default_lls(3);
    assert!(lls.has_full_rank());
    let (x_out, _l) = lls.solve_with_zero_diagonal();
    let x_ref = Vector3::new(87., -38., 10.);
    assert!((x_out - x_ref).norm() < 1e-10);
}
