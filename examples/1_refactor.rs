//! refactored example from https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;

fn mpm2_particle2grid(
    grid: &mut Vec<nalgebra::Vector3::<Real>>,
    n: usize,
    particles: &Vec::<mpm2::particle_solid::ParticleSolid<Real>>,
    vol: Real,
    particle_mass: Real,
    dt: Real,
    hardening: Real,
    young: Real,
    nu: Real) {
    assert_eq!(grid.len(), (n + 1) * (n + 1));
    let dx: Real = 1.0 / n as Real;
    let inv_dx: Real = 1.0 / dx;

    // Reset grid
    grid.iter_mut().for_each(|p| p.set_zero());

    // P2G
    for p in particles {
        let stress = { // (affine * dx) -> delta momentum
            let pf = mpm2::pf(&p.defgrad, hardening, young, nu, p.det_defgrad_plastic);
            let dinv = 4. * inv_dx * inv_dx;
            let stress = -(dt * vol) * (dinv * pf); // dt * volume * force = momentum grad
            stress
        };
        let gds = mpm2::grid_datas(&p.x, dx, inv_dx, n);
        for &gd in gds.iter() {
            let dv = p.velograd * gd.1; // gradient correction of velocity
            let dm = stress * gd.1; // increase of momentum due to stress
            let mass_x_velocity = nalgebra::Vector3::<Real>::new(
                particle_mass * (p.v.x + dv.x) + dm.x,
                particle_mass * (p.v.y + dv.y) + dm.y,
                particle_mass);
            grid[gd.0] += gd.2 * mass_x_velocity;
        }
    }
}

fn mpm2_grid2particle(
    particles: &mut Vec::<mpm2::particle_solid::ParticleSolid::<Real>>,
    grid: &[nalgebra::Vector3::<Real>],
    ngrid: usize,
    dt: Real,
    is_plastic: bool) {
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
        let mut defgrad_cand: Matrix = (Matrix::identity() + dt * p.velograd) * p.defgrad; // step-time
        let det_defgrad_cand = defgrad_cand.determinant();
        if is_plastic {  // updating deformation gradient tensor by clamping the eignvalues
            defgrad_cand = mpm2::clip_strain(&defgrad_cand);
        }
        p.det_defgrad_plastic = mpm2::myclamp(p.det_defgrad_plastic * det_defgrad_cand / defgrad_cand.determinant(), 0.6, 20.0);
        p.defgrad = defgrad_cand;
    }
}

fn main() {
    let mut particles = Vec::<mpm2::particle_solid::ParticleSolid::<Real>>::new();
    {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        mpm2::particle_solid::add_object(&mut particles, 1000, Vector::new(0.55, 0.45), 1, &mut rng);
        mpm2::particle_solid::add_object(&mut particles, 1000, Vector::new(0.45, 0.65), 2, &mut rng);
        mpm2::particle_solid::add_object(&mut particles, 1000, Vector::new(0.55, 0.85), 3, &mut rng);
    }

    const DT: Real = 1e-4;
    const HARDENING: Real = 10.0; // Snow HARDENING factor
    const YOUNG: Real = 1e4;          // Young's Modulus
    const POISSON: Real = 0.2;         // Poisson ratio
    // const YOUNG: Real = 1e0;          // Young's Modulus
    // const POISSON: Real = 0.49999;         // Poisson ratio
    const VOL: Real = 1.0;
    const PARTICLE_MASS: Real = 1.0;

    const N: usize = 80;
    const M: usize = N + 1;
    let mut grid: Vec<nalgebra::Vector3::<Real>> = vec!(nalgebra::Vector3::<Real>::new(0., 0., 0.); M * M);

    const FRAME_DT: Real = 1e-3;
    let mut canvas = del_canvas::canvas_gif::CanvasGif::new(
        std::path::Path::new("target/1.gif"), (800, 800),
        &vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587));
    let transform_to_scr = nalgebra::Matrix3::<Real>::new(
        canvas.width as Real, 0., 0.,
        0., -(canvas.height as Real), canvas.height as Real,
        0., 0., 1.);
    let mut istep = 0;

    loop {
        istep += 1;
        dbg!(istep);
        mpm2_particle2grid(&mut grid,
                           N,
                           &particles,
                           VOL,
                           PARTICLE_MASS,
                           DT, HARDENING, YOUNG, POISSON);
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
        mpm2_grid2particle(
            &mut particles,
            &grid,
            N,
            DT,
            true);
        if istep % ((FRAME_DT / DT) as i32) == 0 {
            canvas.clear(0);
            for p in particles.iter() {
                canvas.paint_point(p.x.x, p.x.y, &transform_to_scr, 2., p.c);
            }
            canvas.write();
        }
        if istep > 3000 { break; }
    }
}


