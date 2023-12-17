//! refactored example from https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;

fn mpm2_p2g_first(
    bg: &mut mpm2::background2::Grid<Real>,
    particles: &Vec::<mpm2::particle_fluid::Particle<Real>>,
    particle_mass: Real)
{
    // Reset grid
    bg.set_velocity_zero();

    // P2G
    for p in particles {
        let gds = bg.near_interior_grid_boundary_points(&p.x);
        for &gd in gds.iter() {
            let dv = p.velograd * gd.1; // increase of momentum
            let momentum = nalgebra::Vector3::<Real>::new(
                particle_mass * (p.v.x + dv.x),
                particle_mass * (p.v.y + dv.y),
                particle_mass);
            bg.vm[gd.0] += gd.2 * momentum;
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
    dt: Real) {
    let dx: Real = 1.0 / bg.n as Real;
    let inv_dx: Real = 1.0 / dx;
    // P2G
    for p in particles {
        let gds = bg.near_interior_grid_boundary_points(&p.x);
        let mut mass_par_cell = 0.;
        for &gd in gds.iter() {
            mass_par_cell += bg.vm[gd.0].z * gd.2;
        }
        let target_mass_par_cell = target_density * dx * dx;
        let volume = particle_mass / mass_par_cell;
        let pressure = eos_stiffness * ((mass_par_cell / target_mass_par_cell).powi(eos_power) - 1.);
        let pressure = pressure.max(-0.1);
        let stress = Matrix::new(
            -pressure, 0.,
            0., -pressure) + (p.velograd + p.velograd.transpose()).scale(dynamic_viscosity);
        let dinv = 4. * inv_dx * inv_dx;
        let stress = -(dt * volume) * (dinv * stress); // dt * volume * force = momentum grad
        for &gd in gds.iter() {
            let dm = stress * gd.1; // increase of momentum due to stress
            bg.vm[gd.0].x += gd.2 * particle_mass * dm.x;
            bg.vm[gd.0].y += gd.2 * particle_mass * dm.y;
        }
    }
}

fn mpm2_g2p(
    particles: &mut Vec::<mpm2::particle_fluid::Particle::<Real>>,
    bg: &mpm2::background2::Grid::<Real>,
    dt: Real) {
    let dx = 1.0 / bg.n as Real;
    let inv_dx = 1. / dx;
    for p in particles {
        p.velograd.set_zero();
        p.v.set_zero();
        let gds = bg.near_interior_grid_boundary_points(&p.x);
        for &gd in gds.iter() {
            let dpos = gd.1;
            let grid_v = Vector::new(bg.vm[gd.0].x, bg.vm[gd.0].y);
            let weight = gd.2;
            p.v += weight * Vector::new(grid_v.x, grid_v.y);
            let t: Matrix = Matrix::new(
                grid_v.x * dpos.x, grid_v.x * dpos.y,
                grid_v.y * dpos.x, grid_v.y * dpos.y);
            p.velograd += 4. * inv_dx * inv_dx * weight * t; // gradient of velocity, C*dx = dv
        }
        p.x += dt * p.v;  // step-time
    }
}

fn main() {
    const DT: Real = 4e-5;
    const EOS_STIFFNESS: Real = 10.0e+1_f64;
    const EOS_POWER: i32 = 4_i32;
    const DYNAMIC_VISCOSITY: Real = 3e-1f64;
    const TARGET_DENSITY: Real = 40_000_f64; // mass par unit square
    //////
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

    let mut bg = {
        let boundary = 0.047;
        let vtx2xy_boundary = [
            //boundary, boundary,
            boundary, 0.5,
            1. - boundary, boundary,
            1. - boundary, 1. - boundary,
            boundary, 1. - boundary];
        mpm2::background2::Grid::new(80, &vtx2xy_boundary, false)
    };

    const FRAME_DT: Real = 1e-3;
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new("target/6.gif"), (800, 800),
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
        dbg!(i_step);
        mpm2_p2g_first(
            &mut bg,
            &particles,
            particle_mass);
        mpm2_p2g_second(
            &mut bg,
            &particles,
            particle_mass, TARGET_DENSITY,
            EOS_STIFFNESS, EOS_POWER, DYNAMIC_VISCOSITY,
            DT);
        bg.set_boundary(&(DT * nalgebra::Vector3::new(0., -200., 0.)));
        mpm2_g2p(
            &mut particles,
            &bg,
            DT);
        if i_step % ((FRAME_DT / DT) as i32) == 0 {
            canvas.clear(0);
            canvas.paint_polyloop(
                &bg.vtx2xy_boundary, &transform_to_scr,
                0.6, 2);
            for p in particles.iter() {
                canvas.paint_point(p.x.x, p.x.y, &transform_to_scr,
                                   2., p.c);
            }
            for p in &bg.points {
                canvas.paint_point(p.x, p.y, &transform_to_scr,
                                   1., 4);
            }
            for i in 0..bg.m {
                for j in 0..bg.m {
                    let dh = 1.0 / bg.m as Real;
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
        if i_step > 10000 { break; }
    }
}


