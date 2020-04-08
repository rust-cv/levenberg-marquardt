use arrsac::{Arrsac, Config};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{
    dimension::{U1, U2, U3},
    storage::Owned,
    Dynamic, Matrix, Matrix3x2, VecStorage, Vector2, Vector3,
};
use pcg_rand::Pcg64;
use rand::distributions::Uniform;
use rand::{distributions::Distribution, Rng};
use sample_consensus::{Consensus, Estimator, Model};

const RESIDUAL_SCALE: f32 = 0.1;
const LINES_TO_ESTIMATE: usize = 1000;

#[derive(Debug, Clone)]
struct Line {
    normal: Vector2<f32>,
    c: f32,
}

impl Line {
    fn xy_residuals(&self, point: Vector2<f32>) -> Vector2<f32> {
        (self.c - self.normal.dot(&point)) * self.normal
    }

    /// This takes in a point and computes the Jacobian of the vector from
    /// the point projected onto the line to the point itself. The
    /// Jacobian is computed in respect to the model itself.
    ///
    /// J= dx,y/dnx,ny,c
    ///
    /// The Jacobian is the transpose of a normal Jacobian because nalgebra is
    /// column-major. This means that the columns are the vector x and y,
    /// while the rows are the model parameters.
    #[rustfmt::skip]
    fn jacobian(&self, point: Vector2<f32>) -> Matrix3x2<f32> {
        let (sx, sy) = (point.x, point.y);
        let (nx, ny) = (self.normal.x, self.normal.y);
        let c = self.c;
        Matrix3x2::new(
            c - 2.0 * sx * nx - sy * ny,   -sx * ny,
            - sy * nx,                      c - sx * nx - 2.0 * sy * ny,
            nx,                             ny,
        )
    }

    fn into_vec(self) -> Vector3<f32> {
        self.normal.push(self.c)
    }

    fn from_vec(v: Vector3<f32>) -> Self {
        Self {
            normal: v.xy(),
            c: v.z,
        }
    }

    fn norm_cosine_distance(&self, other: &Self) -> f32 {
        1.0 - self.normal.dot(&other.normal).abs()
    }
}

impl Model<Vector2<f32>> for Line {
    fn residual(&self, point: &Vector2<f32>) -> f32 {
        (self.normal.dot(point) - self.c).abs()
    }
}

struct LineEstimator;

impl Estimator<Vector2<f32>> for LineEstimator {
    type Model = Line;
    type ModelIter = std::iter::Once<Line>;
    const MIN_SAMPLES: usize = 2;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector2<f32>> + Clone,
    {
        let a = data.next().unwrap();
        let b = data.next().unwrap();
        let normal = Vector2::new(a.y - b.y, b.x - a.x).normalize();
        let c = -normal.dot(&b);
        std::iter::once(Line { normal, c })
    }
}

#[derive(Clone)]
struct LineFittingOptimizationProblem<'a> {
    points: &'a [Vector2<f32>],
    model: Line,
}

impl<'a> LeastSquaresProblem<f32, U3, Dynamic, U2> for LineFittingOptimizationProblem<'a> {
    type ParameterStorage = Owned<f32, U3, U1>;
    type JacobianStorage = Owned<f32, U3, U2>;
    type ResidualStorage = VecStorage<f32, U2, Dynamic>;

    fn apply_parameter_step(&mut self, delta: &Vector3<f32>) {
        let new_params = self.model.clone().into_vec() + delta;
        let new_params = new_params.xy().normalize().push(new_params.z);
        self.model = Line::from_vec(new_params);
    }

    fn residuals(&self) -> Matrix<f32, U2, Dynamic, Self::ResidualStorage> {
        let residual_data = self
            .points
            .iter()
            .flat_map(|&point| {
                use std::iter::once;
                let vec = self.model.xy_residuals(point);
                once(vec.x).chain(once(vec.y))
            })
            .collect();
        RESIDUAL_SCALE * Matrix::<f32, U2, Dynamic, Self::ResidualStorage>::from_vec(residual_data)
    }

    fn transposed_jacobian(&self, i: usize) -> Matrix3x2<f32> {
        self.model.jacobian(self.points[i])
    }
}

#[test]
fn lines() {
    let mut rng = Pcg64::new_unseeded();
    // The max candidate hypotheses had to be increased dramatically to ensure all 1000 cases find a
    // good-fitting line.
    let mut arrsac = Arrsac::new(Config::new(3.0), Pcg64::new_unseeded());

    for _ in 0..LINES_TO_ESTIMATE {
        // Generate <a, b> and normalize.
        let normal = Vector2::new(rng.gen_range(-10.0, 10.0), rng.gen_range(-10.0, 10.0)).normalize();
        // Get parallel ray.
        let ray = Vector2::new(normal.y, -normal.x);
        // Generate random c.
        let c = rng.gen_range(-10.0, 10.0);

        // Generate random number of points between 50 and 1000.
        let num = rng.gen_range(100, 1000);
        // The points should be no more than 5.0 away from the line and be evenly distributed away from the line.
        let residuals = Uniform::new(-5.0, 5.0);
        // The points must be generated along the line, but the distance should be bounded to make it more difficult.
        let distances = Uniform::new(-50.0, 50.0);
        // Generate the points.
        let points: Vec<Vector2<f32>> = (0..num)
            .map(|_| {
                let residual: f32 = residuals.sample(&mut rng);
                let distance: f32 = distances.sample(&mut rng);
                let along = ray * distance;
                let against = (residual - c) * normal;
                along + against
            })
            .collect();

        let model = arrsac
            .model(&LineEstimator, points.iter().copied())
            .expect("unable to estimate a model");
        // Now perform Levenberg-Marquardt.
        let model = LevenbergMarquardt::default()
            .minimize(LineFittingOptimizationProblem {
                // initial value
                model,
                points: &points,
            })
            .model;
        let real_model = Line { normal, c };
        let new_cosine_distance = model.norm_cosine_distance(&real_model);

        // Check the slope using the cosine distance.
        assert!(new_cosine_distance < 0.01, "slope out of expected range");
    }
}
