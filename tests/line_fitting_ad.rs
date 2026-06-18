#![cfg(feature = "num-dual")]
use arrsac::Arrsac;
use levenberg_marquardt::{
    ADWrapper, LeastSquaresProblem, LeastSquaresProblemAD, LevenbergMarquardt,
};
use nalgebra::{DVector, Dyn, SVector, U2, Vector2};
use num_dual::DualNum;
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, Uniform},
};
use rand_chacha::ChaCha20Rng;
use sample_consensus::{Consensus, Estimator, Model};

type F = f64;

const LINES_TO_ESTIMATE: usize = 1000;

#[derive(Debug, Clone)]
struct Line<D: DualNum<F> = F> {
    normal_angle: D,
    c: D,
}

impl<D: DualNum<F>> Line<D> {
    fn xy_residuals(&self, point: Vector2<F>) -> Vector2<D> {
        let normal = self.normal();
        &normal * (self.c.clone() - normal.dot(&point.map(D::from)))
    }

    fn into_vec(self) -> Vector2<D> {
        Vector2::new(self.normal_angle, self.c)
    }

    fn from_vec(v: Vector2<D>) -> Self {
        let [[normal_angle, c]] = v.data.0;
        Self { normal_angle, c }
    }

    fn norm_cosine_distance(&self, other: &Self) -> D {
        -self.normal().dot(&other.normal()).abs() + 1.0
    }

    fn normal(&self) -> Vector2<D> {
        Vector2::new(self.normal_angle.cos(), self.normal_angle.sin())
    }
}

impl Model<Vector2<F>> for Line {
    fn residual(&self, point: &Vector2<F>) -> F {
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
}

impl<'a> LeastSquaresProblemAD<F, Dyn, U2> for LineFittingOptimizationProblem<'a> {
    fn residuals<D: DualNum<F>>(&self, x: SVector<D, 2>) -> Option<DVector<D>> {
        let model = Line::from_vec(x);
        let residual_data = self
            .points
            .iter()
            .flat_map(|&point| {
                let [vec] = model.xy_residuals(point).data.0;
                vec
            })
            .collect();
        Some(DVector::from_vec(residual_data))
    }
}

#[test]
fn lines() {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    // The max candidate hypotheses had to be increased dramatically to ensure all 1000 cases find a
    // good-fitting line.
    let mut arrsac = Arrsac::new(5.0, ChaCha20Rng::seed_from_u64(1));
    let mut would_have_failed = false;
    for _ in 0..LINES_TO_ESTIMATE {
        // Generate <a, b> and normalize.
        let normal =
            Vector2::new(rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)).normalize();
        let normal_angle = F::atan2(normal[1], normal[0]);
        // Get parallel ray.
        let ray = Vector2::new(normal.y, -normal.x);
        // Generate random c.
        let c = rng.random_range(-10.0..10.0);

        // Generate random number of points.
        let num = rng.random_range(100..1000);
        // The points should be no more than 5.0 away from the line and be evenly distributed away from the line.
        let residuals = Uniform::new(-5.0, 5.0).unwrap();
        // The points must be generated along the line, but the distance should be bounded to make it more difficult.
        let distances = Uniform::new(-50.0, 50.0).unwrap();
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
        let problem = LineFittingOptimizationProblem { points: &points };
        let ad_problem = ADWrapper::new(problem, model.clone().into_vec());
        let (problem, report) = LevenbergMarquardt::new().minimize(ad_problem);
        assert!(report.termination.was_successful());
        let real_model = Line { normal_angle, c };
        would_have_failed = would_have_failed || model.norm_cosine_distance(&real_model) >= 0.01;
        let new_cosine_distance =
            Line::from_vec(problem.params()).norm_cosine_distance(&real_model);

        // Check the slope using the cosine distance.
        assert!(new_cosine_distance < 0.001, "slope out of expected range");
    }
    // test that there were initial guesses that wouldn't have been enough
    assert!(would_have_failed);
}
