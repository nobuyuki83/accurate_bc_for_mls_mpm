//! comparison in the paper for fluid

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Matrix = nalgebra::Matrix2<Real>;

fn mpm2_p2g_first(
    bg: &mut mpm2::background2::Grid<Real>,
    particles: &Vec::<mpm2::particle_fluid::Particle<Real>>,
    particle_mass: Real,
    is_ours: bool)
{
    // Reset grid
    bg.set_velocity_zero();
    bg.gp2mass.iter_mut().for_each(|v| *v = Real::zero() );

    // P2G
    for p in particles {
        let (gds, mat_d) = bg.velocity_interpolation(&p.x, is_ours);
        {
            for &gd in gds.iter() {
                let q = nalgebra::Vector3::<Real>::new(1., gd.1.x, gd.1.y);
                let weight = gd.2 * (mat_d * q).x;
                let dv = p.velograd * gd.1; // increase of momentum
                let momentum = nalgebra::Vector3::<Real>::new(
                    particle_mass * (p.v.x + dv.x),
                    particle_mass * (p.v.y + dv.y),
                    particle_mass);
                bg.vm[gd.0] += weight * momentum;
            }
        }
        {   // distribute momentum
            let gds = bg.near_grid_points(&p.x);
            for &gd in gds.iter() {
                bg.gp2mass[gd.0] += gd.2 * particle_mass;
            }
        }
    }
}

fn mpm2_p2g_second(
    bg: &mut mpm2::background2::Grid<Real>,
    particles: &Vec::<mpm2::particle_fluid::Particle<Real>>,
    particle_mass: Real,
    target_density: Real,
    eos_stiffness: Real,
    eos_power: i32,
    dynamic_viscosity: Real,
    dt: Real,
    is_ours: bool) {
    let dx: Real = 1.0 / bg.n as Real;
    // P2G
    for p in particles {
        let (volume, stress) = {
            let gds = bg.near_grid_points(&p.x);
            let mut mass_par_cell = 0.;
            for &gd in gds.iter() {
                mass_par_cell += bg.gp2mass[gd.0] * gd.2;
            }
            let target_mass_par_cell = target_density * dx * dx;
            let volume = particle_mass / mass_par_cell;
            let pressure = eos_stiffness * ((mass_par_cell / target_mass_par_cell).powi(eos_power) - 1.);
            let pressure = pressure.max(-0.1);
            let stress = Matrix::new(
                -pressure, 0.,
                0., -pressure) + (p.velograd + p.velograd.transpose()).scale(dynamic_viscosity);
            (volume, stress)
        };
        let (gds, mat_d_inv) = bg.velocity_interpolation(&p.x, is_ours);
        for &gd in gds.iter() {
            let q = nalgebra::Vector3::<Real>::new(1., gd.1.x, gd.1.y);
            let d = gd.2 * nalgebra::Vector2::<Real>::new((mat_d_inv * q).y, (mat_d_inv * q).z);
            let force = (stress * d).scale(-volume);
            let moment = force.scale(dt); // increase of momentum
            bg.vm[gd.0].x += moment.x;
            bg.vm[gd.0].y += moment.y;
        }
    }
}

fn mpm2_g2p(
    particles: &mut Vec::<mpm2::particle_fluid::Particle::<Real>>,
    bg: &mpm2::background2::Grid::<Real>,
    dt: Real,
    is_ours: bool) {
    for p in particles {
        p.velograd.set_zero();
        p.v.set_zero();
        let (gds, mat_d_inv) = bg.velocity_interpolation(&p.x, is_ours);
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
        p.x += dt * p.v;  // step-time
        // project

    }
}

fn sim(is_ours: bool, is_full_slip: bool, path: &str) {
    const DT: Real = 5e-5;
    const EOS_STIFFNESS: Real = 10.0e+1_f64;
    const EOS_POWER: i32 = 4_i32;
    const DYNAMIC_VISCOSITY: Real = 4e-1f64;
    const TARGET_DENSITY: Real = 60_000_f64; // mass par unit square
    //
    let mut particles = Vec::<mpm2::particle_fluid::Particle::<Real>>::new();
    let area = {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let cell_len = 0.007;
        // let vtx2xy0 = vec!(0.47, 0.37, 0.63, 0.37, 0.63, 0.53, 0.47, 0.53);
        let vtx2xy0 = del_msh::polyloop2::from_pentagram(&[0.55, 0.45], 0.13);
        let area0 = del_msh::polyloop2::area(&vtx2xy0);
        let xys0 = del_msh::polyloop2::to_uniform_density_random_points(&vtx2xy0, cell_len, &mut rng);
        xys0.chunks(2).for_each(
            |v|
                particles.push(
                    mpm2::particle_fluid::Particle::new(
                        nalgebra::Vector2::<Real>::new(v[0], v[1]), 1)));
        //
        // let vtx2xy1 = vec!(0.37, 0.57, 0.53, 0.57, 0.53, 0.73, 0.37, 0.73);
        let vtx2xy1 = del_msh::polyloop2::from_pentagram(&[0.45, 0.75], 0.17);
        let area1 = del_msh::polyloop2::area(&vtx2xy1);
        let xys1 = del_msh::polyloop2::to_uniform_density_random_points(&vtx2xy1, cell_len, &mut rng);
        xys1.chunks(2).for_each(
            |v|
                particles.push(
                    mpm2::particle_fluid::Particle::new(
                        nalgebra::Vector2::<Real>::new(v[0], v[1]), 2))
        );
        //
        area0 + area1
    };
    let particle_mass: Real = area * TARGET_DENSITY / (particles.len() as Real);

    const N: usize = 80;
    let mut bg = {
        let boundary = 0.047;
        let vtx2xy_boundary = [
            //boundary, boundary,
            boundary, 0.5,
            1. - boundary, boundary,
            1. - boundary, 1. - boundary,
            boundary, 1. - boundary];
        let delta = if is_ours {
            (1. / N as Real) * 0.25 }
        else{
            Real::zero() };
        // augmented grid
        mpm2::background2::Grid::new(
            N, &vtx2xy_boundary, is_ours,
            delta)
    };

    const SKIP_FRAME: usize = 20;
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new(path), (800, 800),
        &vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587, 0xffffff, 0xFF00FF, 0xFFFF00));
    let transform_to_scr = nalgebra::Matrix3::<Real>::new(
        canvas.width as Real, 0., 0.,
        0., -(canvas.height as Real), canvas.height as Real,
        0., 0., 1.);

    canvas.clear(0);
    canvas.paint_polyloop(
        &bg.vtx2xy_boundary, &transform_to_scr,
        2., 2);
    for p in particles.iter() {
        canvas.paint_point(p.x.x, p.x.y, &transform_to_scr,
                           2., p.c);
    }
    canvas.write();
    let mut i_step = 0;

    loop {
        i_step += 1;
        if i_step % 100 == 0 {
            dbg!(i_step);
        }
        mpm2_p2g_first(
            &mut bg,
            &particles,
            particle_mass, is_ours);
        mpm2_p2g_second(
            &mut bg,
            &particles,
            particle_mass, TARGET_DENSITY,
            EOS_STIFFNESS, EOS_POWER, DYNAMIC_VISCOSITY,
            DT, is_ours);
        bg.set_boundary(
            &(DT * nalgebra::Vector3::new(0., -200., 0.)),
            is_full_slip);
        mpm2_g2p(
            &mut particles,
            &bg,
            DT, is_ours);
        if i_step % SKIP_FRAME == 0 {
            canvas.clear(0);
            canvas.paint_polyloop(
                &bg.vtx2xy_boundary, &transform_to_scr,
                0.6, 2);
            for p in particles.iter() {
                canvas.paint_point(p.x.x, p.x.y, &transform_to_scr,
                                   2., p.c);
            }
            for p in &bg.gpbc2xy {
                canvas.paint_point(p.x, p.y, &transform_to_scr,
                                   1., 4);
            }
            for i in 0..bg.m {
                for j in 0..bg.m {
                    let dh = 1.0 / bg.n as Real;
                    if bg.is_inside[j*bg.m+i] {
                        canvas.paint_point(
                            dh * i as Real, dh * j as Real, &transform_to_scr,
                            1., 5);
                    } else {
                        canvas.paint_point(
                            dh * i as Real, dh * j as Real, &transform_to_scr,
                            1., 6);
                    }
                }
            }
            canvas.write();
        }
        if i_step > 6000 { break; }
    }
}


fn main() {
    sim(false, true,"target/7_naive_fullslip.gif");
    sim(false, false,"target/7_naive_nonslip.gif");
    sim(true, false,"target/7_ours_nonslip.gif");
}

