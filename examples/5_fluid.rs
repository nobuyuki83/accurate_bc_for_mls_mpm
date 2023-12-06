//! refactored example from https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;

fn mpm2_p2g_first(
    grid: &mut Vec<nalgebra::Vector3::<Real>>,
    n: usize,
    particles: &Vec::<mpm2::particle_a::Particle<Real>>,
    particle_mass: Real)
{
    assert_eq!(grid.len(), (n + 1) * (n + 1));
    let dx: Real = 1.0 / n as Real;
    let inv_dx: Real = 1.0 / dx;

    // Reset grid
    grid.iter_mut().for_each(|p| p.set_zero());

    // P2G
    for p in particles {
        let gds = mpm2::grid_datas(&p.x, dx, inv_dx, n);
        for &gd in gds.iter() {
            let dv = p.velograd * gd.1; // increase of momentum
            let momentum = nalgebra::Vector3::<Real>::new(
                particle_mass * (p.v.x + dv.x),
                particle_mass * (p.v.y + dv.y),
                particle_mass);
            grid[gd.0] += gd.2 * momentum;
        }
    }
}

fn mpm2_p2g_second(
    grid: &mut Vec<nalgebra::Vector3::<Real>>,
    n: usize,
    particles: &Vec::<mpm2::particle_a::Particle<Real>>,
    particle_mass: Real,
    dt: Real) {
    assert_eq!(grid.len(), (n + 1) * (n + 1));
    let dx: Real = 1.0 / n as Real;
    let inv_dx: Real = 1.0 / dx;

    let eos_stiffness = 10.0e+1_f64;
    let eos_power = 4_i32;
    let dynamic_viscosity = 3e-1f64;
    // P2G
    for p in particles {
        let gds = mpm2::grid_datas(&p.x, dx, inv_dx, n);
        let mut density = 0.;
        for &gd in gds.iter() {
            density += grid[gd.0].z * gd.2;
        }
        let volume = particle_mass / density;
        let pressure = eos_stiffness * ((density/5.).powi(eos_power)-1.); // flag?
        let pressure = pressure.max(-0.1);
        let stress =  Matrix::new(
            -pressure, 0.,
            0., -pressure) + (p.velograd + p.velograd.transpose()).scale(dynamic_viscosity);
        let dinv = 4. * inv_dx * inv_dx;
        let stress = -(dt * volume) * (dinv * stress); // dt * volume * force = momentum grad
        for &gd in gds.iter() {
            let dm = stress * gd.1; // increase of momentum due to stress
            grid[gd.0].x += gd.2 * particle_mass * dm.x;
            grid[gd.0].y += gd.2 * particle_mass * dm.y;
        }
    }
}

fn mpm2_g2p(
    particles: &mut Vec::<mpm2::particle_a::Particle::<Real>>,
    grid: &[nalgebra::Vector3::<Real>],
    ngrid: usize,
    dt: Real) {
    let dx = 1.0 / ngrid as Real;
    let inv_dx = 1. / dx;
    for p in particles {
        p.velograd.set_zero();
        p.v.set_zero();
        let gds = mpm2::grid_datas(&p.x, dx, inv_dx, ngrid);
        for &gd in gds.iter() {
            let dpos = gd.1;
            let grid_v = Vector::new(grid[gd.0].x, grid[gd.0].y);
            let weight = gd.2;
            p.v += weight * Vector::new(grid_v.x, grid_v.y);
            let t: Matrix = Matrix::new(
                grid_v.x * dpos.x, grid_v.x * dpos.y,
                grid_v.y * dpos.x, grid_v.y * dpos.y);
            p.velograd += 4. * inv_dx * inv_dx * weight * t; // gradient of velocity, C*dx = dv
        }
        p.x += dt * p.v;  // step-time
        p.defgrad = (Matrix::identity() + dt * p.velograd) * p.defgrad; // step-time
    }
}

fn main() {
    let mut particles = Vec::<mpm2::particle_a::Particle::<Real>>::new();
    {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        mpm2::particle_a::add_object(&mut particles, Vector::new(0.55, 0.45), 1, &mut rng);
        mpm2::particle_a::add_object(&mut particles, Vector::new(0.45, 0.65), 2, &mut rng);
        mpm2::particle_a::add_object(&mut particles, Vector::new(0.55, 0.85), 3, &mut rng);
    }

    const DT: Real = 1e-4;
    const PARTICLE_MASS: Real = 1.0;

    const N: usize = 80;
    const M: usize = N + 1;
    let mut grid: Vec<nalgebra::Vector3::<Real>> = vec!(nalgebra::Vector3::<Real>::new(0., 0., 0.); M * M);

    const FRAME_DT: Real = 1e-3;
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new("target/5.gif"), (800, 800),
        &vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587));
    canvas.clear(0);
    for p in particles.iter() {
        canvas.paint_circle(p.x.x * canvas.width as Real,
                            p.x.y * canvas.height as Real,
                            2., p.c);
    }
    canvas.write();
    let mut istep = 0;

    loop {
        istep += 1;
        dbg!(istep);
        mpm2_p2g_first(
            &mut grid,
            N,
            &particles,
            PARTICLE_MASS);
        mpm2_p2g_second(
            &mut grid,
            N,
            &particles,
            PARTICLE_MASS, DT);
        {
            for igrid in 0..M {
                for jgrid in 0..M {  // For all grid nodes
                    let g0 = &mut grid[jgrid * M + igrid];
                    if g0.z <= 0. { continue; } // grid is empty
                    *g0 /= g0.z;
                    *g0 += DT * nalgebra::Vector3::new(0., -200., 0.);
                    let boundary = 0.05;
                    let x = igrid as Real / N as Real;
                    let y = jgrid as Real / N as Real;
                    if x < boundary || x > 1. - boundary || y > 1. - boundary {
                        *g0 = nalgebra::Vector3::repeat(0.); // Sticky
                    }
                    if y < boundary && g0.y < 0. {
                        g0.y = 0.;
                    }
                }
            }
        }
        mpm2_g2p(
            &mut particles,
            &grid,
            N,
            DT);
        if istep % ((FRAME_DT / DT) as i32) == 0 {
            canvas.clear(0);
            for p in particles.iter() {
                canvas.paint_circle(p.x.x * canvas.width as Real,
                                    p.x.y * canvas.height as Real,
                                    2., p.c);
            }
            canvas.write();
        }
        if istep > 3000 { break; }
    }
}


