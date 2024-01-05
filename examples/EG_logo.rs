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
    bg.gp2mass.iter_mut().for_each(|v| *v = Real::zero() );

    // P2G
    for p in particles {
        {
            let gds = bg.near_interior_grid_boundary_points(&p.x);
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
        {
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
    dt: Real) {
    let dx: Real = 1.0 / bg.n as Real;
    let inv_dx: Real = 1.0 / dx;
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
        let gds = bg.near_interior_grid_boundary_points(&p.x);
        let mat_d_inv = {
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
            mat_d.try_inverse().unwrap()
        };
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
    dt: Real) {
    let dx = 1.0 / bg.n as Real;
    let inv_dx = 1. / dx;
    for p in particles {
        p.velograd.set_zero();
        p.v.set_zero();
        let gds = bg.near_interior_grid_boundary_points(&p.x);
        let mat_d_inv = {
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
            mat_d.try_inverse().unwrap()
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
        p.x += dt * p.v;  // step-time
        // project

    }
}

fn scale_translate_vtx2xy<Real>(
    vtx2xy: &Vec<Real>,
    center: &[Real],
    scale: Real) -> Vec<Real>
    where Real: num_traits::Float + 'static + Copy
{
    let mut xys = Vec::<Real>::new();
    for i in 0..vtx2xy.len() / 2 {
        xys.push(vtx2xy[2 * i] * scale +  center[0]);
        xys.push(vtx2xy[2 * i + 1] * scale + center[1]);
    }
    xys
}

fn main() {
    const DT: Real = 1e-4;
    const EOS_STIFFNESS: Real = 10.0e+1_f64;
    const EOS_POWER: i32 = 4_i32;
    const DYNAMIC_VISCOSITY: Real = 3e-1f64;
    const TARGET_DENSITY: Real = 40_000_f64; // mass par unit square

    // for EG logo
    let e_coords: Vec<Real> = vec!(-0.5, -0.5, 0.5, -0.5, 0.5, -0.3, -0.25, -0.3, -0.25, -0.1, 0.5, -0.1, 0.5, 0.1, -0.25, 0.1, -0.25, 0.3, 0.5, 0.3, 0.5, 0.5, -0.5, 0.5);
    let g_coords: Vec<Real> = vec!(-0.5, -0.5, 0.5, -0.5, 0.5, 0.1, 0.0, 0.1, 0.0, -0.1, 0.25, -0.1, 0.25, -0.3, -0.25, -0.3, -0.25, 0.3, 0.5, 0.3, 0.5, 0.5, -0.5, 0.5);
    //////
    let mut particles = Vec::<mpm2::particle_fluid::Particle::<Real>>::new();
    let area = {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let cell_len = 0.007;

        let vtx2xy0 = scale_translate_vtx2xy(&e_coords, &[0.45, 0.75], 0.25);
        let area0 = del_msh::polyloop2::area(&vtx2xy0);
        let xys0 = del_msh::polyloop2::to_uniform_density_random_points(&vtx2xy0, cell_len, &mut rng);
        xys0.chunks(2).for_each(
            |v|
                particles.push(
                    mpm2::particle_fluid::Particle::new(
                        nalgebra::Vector2::<Real>::new(v[0], v[1]), 1)));
        
        let vtx2xy1 = scale_translate_vtx2xy(&g_coords, &[0.65, 0.45], 0.21);
        let area1 = del_msh::polyloop2::area(&vtx2xy1);
        let xys1 = del_msh::polyloop2::to_uniform_density_random_points(&vtx2xy1, cell_len, &mut rng);
        xys1.chunks(2).for_each(
            |v|
                particles.push(
                    mpm2::particle_fluid::Particle::new(
                        nalgebra::Vector2::<Real>::new(v[0], v[1]), 2))
        );
        area0 + area1
    };
    let particle_mass: Real = area * TARGET_DENSITY / (particles.len() as Real);

    let n: usize = 80;
    let mut bg = {
        let boundary = 0.047;
        let vtx2xy_boundary = [
            //boundary, boundary,
            boundary, 0.5,
            1. - boundary, boundary,
            1. - boundary, 1. - boundary,
            boundary, 1. - boundary];
        // augmented grid
        mpm2::background2::Grid::new(n, &vtx2xy_boundary, true, true, 1 as Real / n as Real / 4 as Real)
        // simple grid
        // mpm2::background2::Grid::new(n, &vtx2xy_boundary, false, false, 0 as Real)
    };

    const FRAME_DT: Real = 1e-4;
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new("target/EG_logo.gif"), (800, 800),
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
        if i_step > 3000 { break; }
    }
}