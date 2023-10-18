use rand::Rng;

type Real = f32;
type Vec2 = nalgebra::Vector2<Real>;
type Mat22 = nalgebra::Matrix2<Real>;

#[derive(Clone)]
struct Particle {
    pos : Vec2,
    vel : Vec2,
    c : Mat22,
    mass : Real,
    dummy : Real, // for performance
}

impl Particle {
    fn new() -> Self {
        Self {
            pos : Vec2::new(0.0, 0.0),
            vel : Vec2::new(0.0, 0.0),
            c : Mat22::new(1.0, 0.0, 0.0, 1.0),
            mass : 1.0,
            dummy : 0.0,
        }
    }
}

#[derive(Clone)]
struct Cell {
    vel : Vec2,
    mass : Real,
    dummy : Real
}

impl Cell {
    fn new() -> Self {
        Self {
            vel : Vec2::new(0.0, 0.0),
            mass : 1.0,
            dummy : 0.0,
        }
    }
}

const N : usize = 32;
const PARTICLE_COUNT : usize = 32;
const DT : f32 = 1.0;
const GRAVITY : f32 = -0.05;

fn main() {
    // init
    let mut grid = vec![Cell::new(); N * N];

    let mut particles = vec![Particle::new(); PARTICLE_COUNT];
    // init particle velocities with random values
    {
        let mut rng = rand::thread_rng();
        for i in 0..PARTICLE_COUNT {
            particles[i].pos = Vec2::new(
                0.1 + i as Real / PARTICLE_COUNT as Real * 0.8,
                0.5);
            particles[i].vel = Vec2::new(
                rng.gen::<Real>() - 0.5,
                rng.gen::<Real>() - 0.5 + 2.75
            ) * 0.5;
        }
    }

}