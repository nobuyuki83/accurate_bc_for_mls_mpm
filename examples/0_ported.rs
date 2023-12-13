/// example ported from  https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use mpm2::canvas;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;


#[derive(Debug)]
struct Particle {
    x: Vector,
    // Position
    v: Vector,
    // Velocity
    F: Matrix,
    // Deformation gradient
    C: Matrix,
    // Affine momentum from APIC
    Jp: Real,
    // Determinant of the deformation gradient (i.e. volume)
    c: u8,     // Color
}

impl Particle {
    fn new(x_: Vector, c_: u8) -> Self {
        Self {
            x: x_,
            v: Vector::zeros(),
            F: Matrix::identity(),
            C: Matrix::zeros(),
            Jp: 1. as Real,
            c: c_,
        }
    }
}

// Seed particles with position and color
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
    let dx: Real = 1. / n as Real;
    let inv_dx: Real = 1. / dx;

    // Reset grid
    grid.iter_mut().for_each(|p| p.set_zero());

    // P2G
    for p in particles {
        let base_coord = p.x * inv_dx - Vector::repeat(0.5_f32 as Real);
        let base_coord = nalgebra::Vector2::<i32>::new(base_coord.x as i32, base_coord.y as i32);
        let fx = p.x * inv_dx - base_coord.cast::<Real>();
        let w = {
            let a = Vector::repeat(1.5) - fx;
            let b = fx - Vector::repeat(1.0);
            let c = fx - Vector::repeat(0.5);
            [
                0.5 * a.component_mul(&a),
                Vector::repeat(0.75) - b.component_mul(&b),
                0.5 * c.component_mul(&c)
            ]
        };

        let affine = {
            let mu_0 = young / (2. * (1. + nu));
            let lambda_0 = young * nu / ((1. + nu) * (1. - 2. * nu));
            let e = (hardening * (1.0 - p.Jp)).exp();
            // let e = 1_f32;
            let mu = mu_0 * e;
            let lambda = lambda_0 * e;
            let J: Real = p.F.determinant();
            let (r, s) = del_geo::mat2::polar_decomposition(&p.F);
            let dinv = 4. * inv_dx * inv_dx;
            let pf = 2. * mu * (p.F - r) * (p.F).transpose() + lambda * (J - 1.) * J * Matrix::identity();
            let stress = -(dt * vol) * (dinv * pf);
            stress + particle_mass * p.C
        };

        for i in 0..3 as usize {
            for j in 0..3 as usize {
                let dpos = Vector::new(
                    (i as Real - fx.x) * dx, (j as Real - fx.y) * dx);
                let mass_x_velocity = nalgebra::Vector3::<Real>::new(
                    particle_mass * p.v.x,
                    particle_mass * p.v.y,
                    particle_mass);
                let iw = (base_coord.x + i as i32) as usize;
                let ih = (base_coord.y + j as i32) as usize;
                let t = affine * dpos;
                let t = nalgebra::Vector3::<Real>::new(t.x, t.y, 0.);
                grid[ih * (n + 1) + iw] +=
                    w[i].x * w[j].y * (mass_x_velocity + t);
            }
        }
    }
}

fn mpm2_grid2particle(
    particles: &mut Vec::<Particle>,
    grid: &Vec<nalgebra::Vector3::<Real>>,
    ngrid: usize,
    dt: Real,
    is_plastic: bool) {
    let dx = 1. / ngrid as Real;
    let mgrid = ngrid + 1;
    let inv_dx = 1. / dx;
    for p in particles {
        let base_coord = p.x * inv_dx - Vector::repeat(0.5);
        let base_coord = nalgebra::Vector2::<i32>::new(base_coord.x as i32, base_coord.y as i32);
        let fx = p.x * inv_dx - base_coord.cast::<Real>();
        let w = {
            let a = Vector::repeat(1.5) - fx;
            let b = fx - Vector::repeat(1.0);
            let c = fx - Vector::repeat(0.5);
            [
                0.5 * a.component_mul(&a),
                Vector::repeat(0.75) - b.component_mul(&b),
                0.5 * c.component_mul(&c)
            ]
        };
        p.C.set_zero();
        p.v.set_zero();
        for i in 0..3 {
            for j in 0..3 {
                let dpos = Vector::new(
                    i as Real - fx.x, j as Real - fx.y);
                let ig = (base_coord.x + i as i32) as usize;
                let jg = (base_coord.y + j as i32) as usize;
                assert!(ig < mgrid);
                assert!(jg < mgrid);
                let grid_v = Vector::new(
                    grid[jg * mgrid + ig].x,
                    grid[jg * mgrid + ig].y);
                let weight = w[i].x * w[j].y;
                p.v += weight * Vector::new(grid_v.x, grid_v.y);
                let t: Matrix = Matrix::new(
                    grid_v.x * dpos.x, grid_v.x * dpos.y,
                    grid_v.y * dpos.x, grid_v.y * dpos.y);
                p.C += 4. * inv_dx * weight * t;
            }
        }
        p.x += dt * p.v;
        let tmp0: Matrix = Matrix::identity() + dt * p.C;
        let mut F0: Matrix = tmp0 * p.F;
        let oldJ0 = F0.determinant();
        if is_plastic {  // updating deformation gradient tensor by clamping the eignvalues
            let svd = F0.svd(true, true);
            let svd_u0: Matrix = svd.u.unwrap();
            let mut sig0: Matrix = Matrix::from_diagonal(&svd.singular_values);
            let svd_v0 = svd.v_t.unwrap().transpose();
            for i in 0..2 {  // Snow Plasticity
                sig0[(i, i)] = mpm2::myclamp(sig0[(i, i)], 1.0 - 2.5e-2, 1.0 + 7.5e-3);
            }
            F0 = svd_u0 * sig0 * svd_v0.transpose();
        }
        let Jp_new0: Real = mpm2::myclamp(p.Jp * oldJ0 / F0.determinant(), 0.6, 20.0);
        p.Jp = Jp_new0;
        p.F = F0;
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
    const E: Real = 1e4;          // Young's Modulus
    const NU: Real = 0.2;         // Poisson ratio
    const VOL: Real = 1.0;
    const PARTICLE_MASS: Real = 1.0;

    const N: usize = 80;
    const M: usize = N + 1;
    let mut grid: Vec<nalgebra::Vector3::<Real>> = vec!(nalgebra::Vector3::<Real>::new(0., 0., 0.); M * M);

    const FRAME_DT: Real = 1e-3;
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new("target/0.gif"), (800, 800),
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
                           DT, HARDENING, E, NU);
        {
            for i_grid in 0..M {
                for j_grid in 0..M {  // For all grid nodes
                    let g0 = &mut grid[j_grid * M + i_grid];
                    if g0.z <= Real::zero() { continue; } // grid is empty
                    *g0 = *g0 / g0.z;
                    *g0 += DT * nalgebra::Vector3::new(0., -200., 0.);
                    let boundary = 0.05_f32;
                    let x = i_grid as f32 / N as f32;
                    let y = j_grid as f32 / N as f32;
                    if x < boundary || x > 1. - boundary || y > 1. - boundary {
                        *g0 = nalgebra::Vector3::repeat(0.); // Sticky
                    }
                    if y < boundary {
                        if g0.y < 0. {
                            g0.y = 0.;
                        }
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
    }
}


