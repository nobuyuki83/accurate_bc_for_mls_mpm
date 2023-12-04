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
    bg: &mut mpm2::background::Background<Real>,
    particles: &Vec::<Particle>,
    vol: Real,
    particle_mass: Real,
    dt: Real,
    hardening: Real,
    young: Real,
    nu: Real,
    istep: i32)
{
    // Reset grid
    bg.set_velocity_zero();

    // P2G
    for (ip, p) in particles.iter().enumerate() {
        let gds = bg.near_samples(&p.x);
        let mat_d = {
            let mut mat_d = nalgebra::Matrix3::<Real>::zero();
            for &gd in gds.iter() {
                let dpos = gd.1;
                let w = gd.2;
                mat_d[(0,0)] += w;
                mat_d[(0,1)] += w * dpos.x;
                mat_d[(0,2)] += w * dpos.y;
                mat_d[(1,0)] += w * dpos.x;
                mat_d[(1,1)] += w * dpos.x * dpos.x;
                mat_d[(1,2)] += w * dpos.x * dpos.y;
                mat_d[(2,0)] += w * dpos.y;
                mat_d[(2,1)] += w * dpos.y * dpos.x;
                mat_d[(2,2)] += w * dpos.y * dpos.y;
            }
            mat_d = mat_d.try_inverse().unwrap();
            mat_d
        };
        let stress = mpm2::pf(&p.defgrad, hardening, young, nu, p.det_defgrad_plastic);
        /*
        if istep >= 764 && ip == 2051 {
            let (r, _s) = mpm2::polar_decomposition(&p.defgrad);
            dbg!(stress, &p.defgrad.determinant());
        }
         */
        let mass_x_velocity = nalgebra::Vector3::<Real>::new(
            particle_mass * p.v.x,
            particle_mass * p.v.y,
            particle_mass);
        for &gd in gds.iter() {
            let q = nalgebra::Vector3::<Real>::new(1., gd.1.x, gd.1.y);
            let weight = gd.2 * (mat_d * q).x;
            let d = gd.2 * nalgebra::Vector2::<Real>::new((mat_d * q).y, (mat_d * q).z);
            // let force = -(vol) * (d.transpose() * stress).transpose();
            let force = (stress * d).scale(-vol);
            let moment = force.scale(dt) + particle_mass * weight * p.velograd * gd.1; // increase of momentum
            let moment = nalgebra::Vector3::<Real>::new(moment.x, moment.y, 0.);
            bg.vm[gd.0] += weight * mass_x_velocity + moment;
        }
    }
}

fn mpm2_grid2particle(
    particles: &mut Vec::<Particle>,
    bg: &mpm2::background::Background::<Real>,
    dt: Real,
    is_plastic: bool,
    istep: i32) {
    for (ip, p) in particles.iter_mut().enumerate() {
        p.velograd.set_zero();
        p.v.set_zero();
        let gds = bg.near_samples(&p.x);
        let (mat_d, mat_d_inv) = {
            let mut mat_d = nalgebra::Matrix3::<Real>::zero();
            for &gd in gds.iter() {
                let dpos = gd.1;
                let w = gd.2;
                mat_d[(0,0)] += w;
                mat_d[(0,1)] += w * dpos.x;
                mat_d[(0,2)] += w * dpos.y;
                mat_d[(1,0)] += w * dpos.x;
                mat_d[(1,1)] += w * dpos.x * dpos.x;
                mat_d[(1,2)] += w * dpos.x * dpos.y;
                mat_d[(2,0)] += w * dpos.y;
                mat_d[(2,1)] += w * dpos.y * dpos.x;
                mat_d[(2,2)] += w * dpos.y * dpos.y;
            }
            let d_inv = mat_d.try_inverse().unwrap();
            (mat_d, d_inv)
        };
        for &gd in gds.iter() {
            let q = nalgebra::Vector3::<Real>::new(1., gd.1.x, gd.1.y);
            let tmp = (mat_d_inv * q).scale(gd.2);
            let vxd = tmp * bg.vm[gd.0].x;
            p.v.x += vxd[0];
            p.velograd[(0,0)] += vxd[1];
            p.velograd[(0,1)] += vxd[2];
            let vyd = tmp * bg.vm[gd.0].y;
            p.v.y += vyd[0];
            p.velograd[(1,0)] += vyd[1];
            p.velograd[(1,1)] += vyd[2];
        }
        //
        p.x += dt * p.v;  // step-time

        // if istep >= 763 && ip == 2051 {
        // dbg!(&p.defgrad, dt*p.velograd);
        //}

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
    let mut bg = mpm2::background::Background::<Real>::new(80);
    let mut particles = Vec::<Particle>::new();
    {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        add_object(&mut particles, Vector::new(0.55, 0.45), 1, &mut rng);
        add_object(&mut particles, Vector::new(0.45, 0.65), 2, &mut rng);
        add_object(&mut particles, Vector::new(0.55, 0.85), 3, &mut rng);
    }

    const DT: Real = 5e-5;
    // const HARDENING: Real = 10.0; // Snow HARDENING factor
    const HARDENING: Real = 10.0; // Snow HARDENING factor
    const YOUNG: Real = 1e4;          // Young's Modulus
    const POISSON: Real = 0.2;         // Poisson ratio
    //const YOUNG: Real = 1e0;          // Young's Modulus
    //const POISSON: Real = 0.499;         // Poisson ratio
    const VOL: Real = 1.0;
    const PARTICLE_MASS: Real = 1.0;

    const FRAME_DT: Real = 1e-3;
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new("target/4.gif"), (800, 800),
        &vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587, 0xFFFFFF, 0xFF00FF));
    let mut istep = 0;

    loop {
        istep += 1;
        dbg!(istep);
        mpm2_particle2grid(
            &mut bg,
            &particles,
            VOL,
            PARTICLE_MASS,
            DT, HARDENING, YOUNG, POISSON,
            istep);
        bg.after_p2g(DT * nalgebra::Vector3::new(0., -200., 0.));
        mpm2_grid2particle(
            &mut particles,
            &bg,
            DT,
            true, istep);
        if istep % ((FRAME_DT / DT) as i32) == 0 {
            canvas.clear(0);
            for p in particles.iter() {
                canvas.paint_circle(
                    p.x.x * canvas.width as Real,
                    p.x.y * canvas.height as Real,
                    2., p.c);
            }
            for p in &bg.points {
                canvas.paint_circle(
                    p.x * canvas.width as Real,
                    p.y * canvas.height as Real,
                    3., 4);
            }
            for ip in 0..bg.m*bg.m {
                let p = bg.xy(ip);
                canvas.paint_circle(
                    p.x * canvas.width as Real,
                    p.y * canvas.height as Real,
                    1., 5);
            }
            canvas.write();
        }
        if istep > 6000 { break; }
    }
}


