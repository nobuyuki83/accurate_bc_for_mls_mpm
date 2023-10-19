use rand::Rng;
use num_traits::Pow;

type Real = f32;
type Vec2 = nalgebra::Vector2<Real>;
type Vec2i = nalgebra::Vector2<i32>;
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
            c : Mat22::new(0.0, 0.0, 0.0, 0.0),
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
            mass : 0.0,
            dummy : 0.0,
        }
    }
}

const N : usize = 32;
const NUM_CELLS : usize = N * N;
const PARTICLE_COUNT : usize = 32;
const DX : f32 = 1.0 / N as Real;
const DT : f32 = 1.0;
const GRAVITY : f32 = -0.05;

fn main() {
    // init
    let mut grid = vec![Cell::new(); NUM_CELLS];

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

    // simulation loop
    {
        // reset the grid
        {
            for cell in &mut grid {
                cell.mass = 0.0;
                cell.vel = Vec2::new(0.0, 0.0);
            }
        }

        // particle to grid
        {
            for p in particles {
                // quadratic interpolation weights
                let cell_diff = Vec2::new(p.pos.x % DX, p.pos.y  % DX);
                let mut weights = vec![Vec2::new(0.0, 0.0); 3];
                weights[0] = 0.5 * Vec2::new((0.5 - cell_diff.x).pow(2.0), (0.5 - cell_diff.y).pow(2.0));
                weights[1] = 0.75 * Vec2::new((cell_diff.x).pow(2.0), (cell_diff.y).pow(2.0));
                weights[2] = 0.5 * Vec2::new((0.5 + cell_diff.x).pow(2.0), (0.5 + cell_diff.y).pow(2.0));

                // for all surrounding 9 cells
                for gx in 0..3 {
                    for gy in 0..3 {
                        let weight = weights[gx].x * weights[gy].y;
                        let cell_id = Vec2i::new((p.pos.x * N as Real) as i32 + gx as i32 - 1, (p.pos.y * N as Real) as i32 + gy as i32 - 1);
                        let cell_dist = Vec2::new(
                            cell_id.x as Real / N as Real - p.pos.x + 0.5,
                            cell_id.y as Real / N as Real - p.pos.y + 0.5
                        );
                        let q = p.c * cell_dist;

                        let mass_contrib = weight * p.mass;
                        let cell_index = cell_id.x as usize * N + cell_id.y as usize;

                        grid[cell_index].mass += mass_contrib;
                        grid[cell_index].vel += mass_contrib * (p.vel + q);
                    }
                }
            }
        }
    }
}