use arrsac::Arrsac;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{
    dimension::{U1, U2},
    storage::Owned,
    Dim, Dynamic, Matrix, Matrix2, OMatrix, VecStorage, Vector2,
};
use pcg_rand::Pcg64;
use rand::distributions::Uniform;
use rand::{distributions::Distribution, Rng};
use sample_consensus::{Consensus, Estimator, Model};

type F = f64;

const LINES_TO_ESTIMATE: usize = 1000;

#[derive(Debug, Clone)]
struct Line {
    normal_angle: F,
    c: F,
}

impl Line {
    fn xy_residuals(&self, point: Vector2<F>) -> Vector2<F> {
        let normal = self.normal();
        (self.c - normal.dot(&point)) * normal
    }

    /// This takes in a point and computes the Jacobian of the vector from
    /// the point projected onto the line to the point itself. The
    /// Jacobian is computed in respect to the model itself.
    #[rustfmt::skip]
    fn jacobian(&self, point: Vector2<F>) -> Matrix2<F> {
        let n = self.normal();
        let nd = Vector2::new(-self.normal_angle.sin(), self.normal_angle.cos());
        let c = self.c;
        let dist_d_angle = (c - n.dot(&point)) * nd - n * point.dot(&nd);
        Matrix2::new(
            dist_d_angle[0],  n[0],
            dist_d_angle[1],  n[1],
        )
    }

    fn into_vec(self) -> Vector2<F> {
        Vector2::new(self.normal_angle, self.c)
    }

    fn from_vec(v: Vector2<F>) -> Self {
        Self {
            normal_angle: v[0],
            c: v[1],
        }
    }

    fn norm_cosine_distance(&self, other: &Self) -> F {
        1.0 - self.normal().dot(&other.normal()).abs()
    }

    fn normal(&self) -> Vector2<F> {
        Vector2::new(self.normal_angle.cos(), self.normal_angle.sin())
    }
}

impl Model<Vector2<F>> for Line {
    fn residual(&self, point: &Vector2<F>) -> f64 {
        (self.normal().dot(point) - self.c).abs()
    }
}

struct LineEstimator;

impl Estimator<Vector2<F>> for LineEstimator {
    type Model = Line;
    type ModelIter = std::iter::Once<Line>;
    const MIN_SAMPLES: usize = 2;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector2<F>> + Clone,
    {
        let a = data.next().unwrap();
        let b = data.next().unwrap();
        let normal = Vector2::new(a.y - b.y, b.x - a.x).normalize();
        let c = -normal.dot(&b);
        let normal_angle = F::atan2(normal[1], normal[0]);
        std::iter::once(Line { normal_angle, c })
    }
}

struct LineFittingOptimizationProblem<'a> {
    points: &'a [Vector2<F>],
    model: Line,
}

impl<'a> LeastSquaresProblem<F, Dynamic, U2> for LineFittingOptimizationProblem<'a> {
    type ParameterStorage = Owned<F, U2, U1>;
    type JacobianStorage = Owned<F, Dynamic, U2>;
    type ResidualStorage = VecStorage<F, Dynamic, U1>;

    fn set_params(&mut self, p: &Vector2<F>) {
        self.model = Line::from_vec(*p);
    }

    fn params(&self) -> Vector2<F> {
        self.model.clone().into_vec()
    }

    fn residuals(&self) -> Option<Matrix<F, Dynamic, U1, Self::ResidualStorage>> {
        let residual_data = self
            .points
            .iter()
            .flat_map(|&point| {
                use std::iter::once;
                let vec = self.model.xy_residuals(point);
                once(vec.x).chain(once(vec.y))
            })
            .collect();
        Some(Matrix::<F, Dynamic, U1, Self::ResidualStorage>::from_vec(
            residual_data,
        ))
    }

    fn jacobian(&self) -> Option<OMatrix<F, Dynamic, U2>> {
        let u2 = Dim::from_usize(2);
        let mut jacobian = OMatrix::zeros_generic(Dynamic::from_usize(self.points.len() * 2), u2);
        for (i, point) in self.points.iter().enumerate() {
            jacobian
                .slice_range_mut(2 * i..2 * (i + 1), ..)
                .copy_from(&self.model.jacobian(*point));
        }
        Some(jacobian)
    }
}

#[test]
fn lines() {
    let mut rng = Pcg64::new_unseeded();
    // The max candidate hypotheses had to be increased dramatically to ensure all 1000 cases find a
    // good-fitting line.
    let mut arrsac = Arrsac::new(5.0, Pcg64::new_unseeded());
    let mut would_have_failed = false;
    for _ in 0..LINES_TO_ESTIMATE {
        // Generate <a, b> and normalize.
        let normal =
            Vector2::new(rng.gen_range(-10.0..10.0), rng.gen_range(-10.0..10.0)).normalize();
        let normal_angle = F::atan2(normal[1], normal[0]);
        // Get parallel ray.
        let ray = Vector2::new(normal.y, -normal.x);
        // Generate random c.
        let c = rng.gen_range(-10.0..10.0);

        // Generate random number of points.
        let num = rng.gen_range(100..1000);
        // The points should be no more than 5.0 away from the line and be evenly distributed away from the line.
        let residuals = Uniform::new(-5.0, 5.0);
        // The points must be generated along the line, but the distance should be bounded to make it more difficult.
        let distances = Uniform::new(-50.0, 50.0);
        // Generate the points.
        let points: Vec<Vector2<F>> = (0..num)
            .map(|_| {
                let residual: F = residuals.sample(&mut rng);
                let distance: F = distances.sample(&mut rng);
                let along = ray * distance;
                let against = (residual - c) * normal;
                along + against
            })
            .collect();

        let model = arrsac
            .model(&LineEstimator, points.iter().copied())
            .expect("unable to estimate a model");
        // Now perform Levenberg-Marquardt.
        let problem = LineFittingOptimizationProblem {
            model: model.clone(),
            points: &points,
        };
        let (problem, report) = LevenbergMarquardt::new().minimize(problem);
        assert!(report.termination.was_successful());
        let real_model = Line { normal_angle, c };
        would_have_failed = would_have_failed || model.norm_cosine_distance(&real_model) >= 0.01;
        let new_cosine_distance = problem.model.norm_cosine_distance(&real_model);

        // Check the slope using the cosine distance.
        assert!(new_cosine_distance < 0.001, "slope out of expected range");
    }
    // test that there were initial guesses that wouldn't have been enough
    assert!(would_have_failed);
}
