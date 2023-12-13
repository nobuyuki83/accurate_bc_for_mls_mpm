//! refactored example from https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;

fn mpm2_particle2grid(
    bg: &mut mpm2::background::Background<Real>,
    particles: &Vec::<mpm2::particle_a::Particle<Real>>,
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
    particles: &mut Vec::<mpm2::particle_a::Particle<Real>>,
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

        let mut defgrad_cand: Matrix = (Matrix::identity() + p.velograd.scale(dt)) * p.defgrad; // step-time
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
    let mut particles = Vec::<mpm2::particle_a::Particle<Real>>::new();
    {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        mpm2::particle_a::add_object::<Real>(&mut particles, 1000, Vector::new(0.55, 0.45), 1, &mut rng);
        mpm2::particle_a::add_object::<Real>(&mut particles, 1000, Vector::new(0.45, 0.65), 2, &mut rng);
        mpm2::particle_a::add_object::<Real>(&mut particles, 1000, Vector::new(0.55, 0.85), 3, &mut rng);
    }

    const DT: Real = 2e-5;
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
    let transform_to_scr = nalgebra::Matrix3::<Real>::new(
        canvas.width as Real, 0., 0.,
        0., -(canvas.height as Real), canvas.height as Real,
        0., 0., 1.);
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
            false, istep);
        if istep % ((FRAME_DT / DT) as i32) == 0 {
            canvas.clear(0);
            for p in particles.iter() {
                canvas.paint_point(p.x.x, p.x.y, &transform_to_scr,
                    2., p.c);
            }
            for p in &bg.points {
                canvas.paint_point(p.x, p.y, &transform_to_scr,
                    3., 4);
            }
            for ip in 0..bg.m*bg.m {
                let p = bg.xy(ip);
                canvas.paint_point(p.x, p.y, &transform_to_scr,
                    1., 5);
            }
            canvas.write();
        }
        if istep > 6000 { break; }
    }
}


