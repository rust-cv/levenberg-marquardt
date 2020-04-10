use arrsac::{Arrsac, Config};
use nalgebra::{dimension::U2, Dynamic, Matrix, Matrix3x2, VecStorage, Vector2, Vector3};
use pcg_rand::Pcg64;
use rand::distributions::Uniform;
use rand::{distributions::Distribution, Rng};
use sample_consensus::{Consensus, Estimator, Model};

const RESIDUAL_SCALE: f32 = 0.1;
const LINES_TO_ESTIMATE: usize = 1000;

#[derive(Copy, Clone, Debug)]
struct Line {
    norm: Vector2<f32>,
    c: f32,
}

impl Line {
    fn xy_residuals(&self, point: Vector2<f32>) -> Vector2<f32> {
        -self.norm * self.c - self.norm.dot(&point) * self.norm
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
        let (nx, ny) = (self.norm.x, self.norm.y);
        let c = self.c;
        Matrix3x2::new(
            c + 2.0 * sx * nx + sy * ny,    sx * ny,
            sy * nx,                        c + sx * nx + 2.0 * sy * ny,
            nx,                             ny,
        )
    }

    fn apply_delta(mut self, delta: Vector3<f32>) -> Self {
        self.norm += delta.xy();
        self.norm.normalize_mut();
        self.c += delta.z;
        self
    }

    fn norm_cosine_distance(&self, other: &Self) -> f32 {
        1.0 - self.norm.dot(&other.norm).abs()
    }
}

impl Model<Vector2<f32>> for Line {
    fn residual(&self, point: &Vector2<f32>) -> f32 {
        (self.norm.dot(point) + self.c).abs()
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
        let norm = Vector2::new(a.y - b.y, b.x - a.x).normalize();
        let c = -norm.dot(&b);
        std::iter::once(Line { norm, c })
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
        let norm = Vector2::new(rng.gen_range(-10.0, 10.0), rng.gen_range(-10.0, 10.0)).normalize();
        // Get parallel ray.
        let ray = Vector2::new(norm.y, -norm.x);
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
                let against = (residual - c) * norm;
                along + against
            })
            .collect();

        let model = arrsac
            .model(&LineEstimator, points.iter().copied())
            .expect("unable to estimate a model");

        let real_model = Line { norm, c };

        // Now perform Levenberg-Marquardt.
        let model = levenberg_marquardt::optimize(
            levenberg_marquardt::Config::default(),
            model,
            |model, delta| model.apply_delta(delta),
            |model| {
                let residual_data = points
                    .iter()
                    .flat_map(|&point| {
                        use std::iter::once;
                        let vec = model.xy_residuals(point);
                        once(vec.x).chain(once(vec.y))
                    })
                    .collect();
                RESIDUAL_SCALE
                    * Matrix::<f32, U2, Dynamic, VecStorage<f32, U2, Dynamic>>::from_vec(
                        residual_data,
                    )
            },
            |&model| points.iter().map(move |&point| model.jacobian(point)),
        );

        let new_cosine_distance = model.norm_cosine_distance(&real_model);

        // Check the slope using the cosine distance.
        assert!(new_cosine_distance < 0.01, "slope out of expected range");
    }
}
