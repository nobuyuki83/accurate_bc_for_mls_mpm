//! refactored example from https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;

#[derive(Debug)]
struct Particle {
    /// Position
    x: Vector,
    /// Velocity
    v: Vector,
    /// Deformation gradient
    defgrad: Matrix,
    /// gradient of momentum from MLS
    velograd: Matrix,
    /// Determinant of the deformation gradient matrix
    det_defgrad_plastic: Real,
    /// Color
    c: u8,
}

impl Particle {
    fn new(x_: Vector, c_: u8) -> Self {
        Self {
            x: x_,
            v: Vector::zeros(),
            defgrad: Matrix::identity(),
            velograd: Matrix::zeros(),
            det_defgrad_plastic: 1.,
            c: c_,
        }
    }
}

/// Seed particles with position and color
fn add_object(
    particles: &mut Vec<Particle>,
    center: Vector,
    c: u8,
    rng: &mut rand::rngs::StdRng) {
    use rand::Rng;
    for _i in 0..1000 {
        let x: Real = (rng.gen::<Real>() * 2. - 1.) * 0.08 + center.x;
        let y: Real = (rng.gen::<Real>() * 2. - 1.) * 0.08 + center.y;
        let p = Particle::new(Vector::new(x, y), c);
        particles.push(p);
    }
}

fn grid_datas(pos_in: &Vector, dx: Real, inv_dx: Real, n: usize) -> Vec<(usize, Vector, Real)> {
    let base_coord = pos_in * inv_dx - Vector::repeat(0.5);
    let base_coord = nalgebra::Vector2::<i32>::new(
        base_coord.x as i32, // e.g., "3.6 -> 3", "3.1 -> 2"
        base_coord.y as i32);
    let fx = pos_in * inv_dx - base_coord.cast::<Real>();
    let wxy = {
        let a = Vector::repeat(1.5) - fx;
        let b = fx - Vector::repeat(1.0);
        let c = fx - Vector::repeat(0.5);
        [
            0.5 * a.component_mul(&a),
            Vector::repeat(0.75) - b.component_mul(&b),
            0.5 * c.component_mul(&c)
        ]
    };
    let mut res = Vec::<(usize, Vector, Real)>::new();
    res.reserve(9);
    for i in 0..3_usize {
        for j in 0..3_usize {
            let dpos = Vector::new(
                (i as Real - fx.x) * dx,
                (j as Real - fx.y) * dx);
            let iw = (base_coord.x + i as i32) as usize;
            let ih = (base_coord.y + j as i32) as usize;
            let w = wxy[i].x * wxy[j].y;
            let ig = ih * (n + 1) + iw;
            res.push((ig, dpos, w));
        }
    }
    res
}

fn mpm2_particle2grid(
    grid: &mut Vec<nalgebra::Vector3::<Real>>,
    n: usize,
    particles: &Vec::<Particle>,
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
        let affine = { // (affine * dx) -> delta momentum
            let pf = mpm2::pf(&p.defgrad, hardening, young, nu, p.det_defgrad_plastic);
            let dinv = 4. * inv_dx * inv_dx;
            let stress = -(dt * vol) * (dinv * pf);
            stress + particle_mass * p.velograd
        };

        let gds = grid_datas(&p.x, dx, inv_dx, n);
        for &gd in gds.iter() {
            let mass_x_velocity = nalgebra::Vector3::<Real>::new(
                particle_mass * p.v.x,
                particle_mass * p.v.y,
                particle_mass);
            let t = affine * gd.1; // increase of momentum
            let t = nalgebra::Vector3::<Real>::new(t.x, t.y, 0.);
            grid[gd.0] += gd.2 * (mass_x_velocity + t);
        }
    }
}

fn mpm2_grid2particle(
    particles: &mut Vec::<Particle>,
    grid: &[nalgebra::Vector3::<Real>],
    ngrid: usize,
    dt: Real,
    is_plastic: bool) {
    let dx = 1.0 / ngrid as Real;
    let inv_dx = 1. / dx;
    for p in particles {
        p.velograd.set_zero();
        p.v.set_zero();
        let gds = grid_datas(&p.x, dx, inv_dx, ngrid);
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
    let mut particles = Vec::<Particle>::new();
    {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        add_object(&mut particles, Vector::new(0.55, 0.45), 1, &mut rng);
        add_object(&mut particles, Vector::new(0.45, 0.65), 2, &mut rng);
        add_object(&mut particles, Vector::new(0.55, 0.85), 3, &mut rng);
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
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new("1.gif"), (800, 800),
        &vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587));
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
                canvas.paint_circle(p.x.x * canvas.width as Real,
                                    p.x.y * canvas.height as Real,
                                    2., p.c);
            }
            canvas.write();
        }
        if istep > 3000 { break; }
    }
}


