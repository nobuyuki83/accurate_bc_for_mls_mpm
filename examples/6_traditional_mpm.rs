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
            pos : Vec2::zeros(),
            vel : Vec2::zeros(),
            c : Mat22::zeros(),
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
const DT : f32 = 1e-1;
const FRAME_DT : f32 = 0.1;
const GRAVITY : f32 = -0.1;

fn calc_weights(p : &Particle) -> [Vec2; 3] {
    let base_coord = Vec2::new(p.pos.x * N as Real - 0.5, p.pos.y * N as Real - 0.5);
    let base_coord = Vec2i::new(base_coord.x as i32, base_coord.y as i32);
    let fx = p.pos * N as Real - base_coord.cast::<Real>();
    let a = Vec2::repeat(1.5) - fx;
    let b = fx - Vec2::repeat(1.0);
    let c = fx - Vec2::repeat(0.5);
    [
        0.5 * a.component_mul(&a),
        Vec2::repeat(0.75) - b.component_mul(&b),
        0.5 * c.component_mul(&c)
    ]
}

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
                0.3 * (rng.gen::<Real>() - 0.5),
                0.3 * (rng.gen::<Real>() + 1.2)
            ) * 0.5;
        }
    }

    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new("6.gif"), (800, 800),
        &vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587));
    let mut istep = 0;

    // simulation loop
    loop {
        istep += 1;
        dbg!(istep);

        // reset the grid
        {
            for cell in &mut grid {
                cell.mass = 0.0;
                cell.vel = Vec2::new(0.0, 0.0);
            }
        }

        // particle to grid
        {
            for p in &particles {
                // quadratic interpolation weights
                let weights = calc_weights(p);

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

        // grid velocity update
        for i in 0..NUM_CELLS {
            let cell = &mut grid[i];

            if cell.mass > 0.0 {
                // convert momentum to vel
                cell.vel /= cell.mass;
                cell.vel += DT * Vec2::new(0.0, GRAVITY);

                // boundary conditions
                let x = i / N;
                let y = i % N;
                if x < 2 || x > N - 3 { cell.vel.x = 0.0; }
                if y < 2 || y > N - 3 { cell.vel.y = 0.0; }
            }
        }

        //  grid to particle
        for p in &mut particles {
            p.vel = Vec2::zeros();

            let weights = calc_weights(p);

            let mut b = Mat22::zeros();
            for gx in 0..3 {
                for gy in 0..3 {
                    let weight = weights[gx].x * weights[gy].y;

                    let cell_id = Vec2i::new((p.pos.x * N as Real) as i32 + gx as i32 - 1, (p.pos.y * N as Real) as i32 + gy as i32 - 1);
                    let cell_index = cell_id.x as usize * N + cell_id.y as usize;

                    let cell_dist = Vec2::new(
                        cell_id.x as Real / N as Real - p.pos.x + 0.5,
                        cell_id.y as Real / N as Real - p.pos.y + 0.5
                    );

                    let weighted_velocity = grid[cell_index].vel * weight;

                    let b_term = Mat22::new(
                        weighted_velocity.x * cell_dist.x,
                        weighted_velocity.x * cell_dist.y,
                        weighted_velocity.y * cell_dist.x,
                        weighted_velocity.y * cell_dist.y,
                    );

                    b += b_term;

                    p.vel += weighted_velocity;
                }
            }

            p.c = b * 4.0 * DX * DX;

            // advect particle
            p.pos += p.vel * DT;

            p.pos.x = p.pos.x.clamp( 0.1, 0.9);
            p.pos.y = p.pos.y.clamp( 0.1, 0.9);
        }

        if istep % ((FRAME_DT / DT) as i32) == 0 {
            canvas.clear(0);
            for p in particles.iter() {
                canvas.paint_circle(
                    p.pos.x * canvas.width as Real,
                    p.pos.y * canvas.height as Real,
                    2.,
                    1);
            }
            canvas.write();
        }
    }
}