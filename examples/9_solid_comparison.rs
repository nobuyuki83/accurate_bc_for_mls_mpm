//! comparison in the paper for solid

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Matrix = nalgebra::Matrix2<Real>;

/*
fn mpm2_p2g_first(
    bg: &mut mpm2::background2::Grid<Real>,
    particles: &Vec::<mpm2::particle_solid::ParticleSolid<Real>>,
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
 */

fn mpm2_p2g(
    bg: &mut mpm2::background2::Grid<Real>,
    particles: &Vec::<mpm2::particle_solid::ParticleSolid<Real>>,
    vol: Real,
    particle_mass: Real,
    hardening: Real,
    young: Real,
    nu: Real,
    dt: Real,
    is_ours: bool)
{
    bg.set_velocity_zero();
    for p in particles {
        let stress = mpm2::pf(&p.defgrad, hardening, young, nu, p.det_defgrad_plastic);
        let mass_x_velocity = nalgebra::Vector3::<Real>::new(
            particle_mass * p.v.x,
            particle_mass * p.v.y,
            particle_mass);
        let (gds, mat_d_inv) = bg.velocity_interpolation(&p.x, is_ours);
        for &gd in gds.iter() {
            let q = nalgebra::Vector3::<Real>::new(1., gd.1.x, gd.1.y);
            let weight = gd.2 * (mat_d_inv * q).x;
            let d = gd.2 * nalgebra::Vector2::<Real>::new((mat_d_inv * q).y, (mat_d_inv * q).z);
            // let force = -(vol) * (d.transpose() * stress).transpose();
            let force = (stress * d).scale(-vol);
            let moment = force.scale(dt) + particle_mass * weight * p.velograd * gd.1; // increase of momentum
            let moment = nalgebra::Vector3::<Real>::new(moment.x, moment.y, 0.);
            bg.vm[gd.0] += weight * mass_x_velocity + moment;
        }
    }
}

fn mpm2_g2p(
    particles: &mut Vec::<mpm2::particle_solid::ParticleSolid::<Real>>,
    bg: &mpm2::background2::Grid::<Real>,
    dt: Real,
    is_plastic: bool,
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
        //
        p.x += dt * p.v;  // step-time
        let mut defgrad_cand: Matrix = (Matrix::identity() + p.velograd.scale(dt)) * p.defgrad; // step-time
        let det_defgrad_cand = defgrad_cand.determinant();
        if is_plastic {  // updating deformation gradient tensor by clamping the eignvalues
            defgrad_cand = mpm2::clip_strain(&defgrad_cand);
        }
        p.det_defgrad_plastic = mpm2::myclamp(p.det_defgrad_plastic * det_defgrad_cand / defgrad_cand.determinant(), 0.6, 20.0);
        p.defgrad = defgrad_cand;
    }
}

fn sim(is_ours: bool, is_snow: bool, is_full_slip: bool, path: &str) {
    const DT: Real = 5e-5;
    const TARGET_DENSITY: Real = 2000_0_f64; // mass par unit square
    const HARDENING: Real = 10.0; // Snow HARDENING factor
    const YOUNG: Real = 1e8;          // Young's Modulus
    const POISSON: Real = 0.2;         // Poisson ratio
    //
    let mut particles = Vec::<mpm2::particle_solid::ParticleSolid<Real>>::new();
    let area = {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let cell_len = 0.007;
        // let vtx2xy0 = vec!(0.47, 0.37, 0.63, 0.37, 0.63, 0.53, 0.47, 0.53);
        // let vtx2xy0 = del_msh::polyloop2::from_pentagram(&[0.55, 0.45], 0.13);
        let e_coords: Vec<Real> = vec!(-0.5, -0.5, 0.5, -0.5, 0.5, -0.3, -0.25, -0.3, -0.25, -0.1, 0.5, -0.1, 0.5, 0.1, -0.25, 0.1, -0.25, 0.3, 0.5, 0.3, 0.5, 0.5, -0.5, 0.5);
        let vtx2xy0 = mpm2::scale_translate_vtx2xy(&e_coords, &[0.45, 0.75], 0.25);
        let area0 = del_msh::polyloop2::area(&vtx2xy0);
        let xys0 = del_msh::polyloop2::to_uniform_density_random_points(&vtx2xy0, cell_len, &mut rng);
        xys0.chunks(2).for_each(
            |v|
                particles.push(
                    mpm2::particle_solid::ParticleSolid::new(
                        nalgebra::Vector2::<Real>::new(v[0], v[1]), 1)));
        //
        // let vtx2xy1 = vec!(0.37, 0.57, 0.53, 0.57, 0.53, 0.73, 0.37, 0.73);
        // let vtx2xy1 = del_msh::polyloop2::from_pentagram(&[0.45, 0.75], 0.17);
        let g_coords: Vec<Real> = vec!(-0.5, -0.5, 0.5, -0.5, 0.5, 0.1, 0.0, 0.1, 0.0, -0.1, 0.25, -0.1, 0.25, -0.3, -0.25, -0.3, -0.25, 0.3, 0.5, 0.3, 0.5, 0.5, -0.5, 0.5);
        let vtx2xy1 = mpm2::scale_translate_vtx2xy(&g_coords, &[0.65, 0.45], 0.21);
        let area1 = del_msh::polyloop2::area(&vtx2xy1);
        let xys1 = del_msh::polyloop2::to_uniform_density_random_points(&vtx2xy1, cell_len, &mut rng);
        xys1.chunks(2).for_each(
            |v|
                particles.push(
                    mpm2::particle_solid::ParticleSolid::new(
                        nalgebra::Vector2::<Real>::new(v[0], v[1]), 2))
        );
        //
        area0 + area1
    };
    let particle_mass: Real = area * TARGET_DENSITY / (particles.len() as Real);
    let vol = area / (particles.len() as Real);

    const N: usize = 80;
    let mut bg = {
        let boundary = 0.047;
        let vtx2xy_boundary = [
            //boundary, boundary,
            boundary, 0.5,
            1. - boundary, boundary,
            1. - boundary, 1. - boundary,
            boundary, 1. - boundary];
        let delta = (1. / N as Real)*0.25;
        // augmented grid
        mpm2::background2::Grid::new(
            N, &vtx2xy_boundary, is_ours,
            delta)
    };

    const SKIP_FRAME: usize = 40;
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new(path), (1600, 1600),
        &vec!(0xffffff, 0x00CCCC, 0x00CC00, 0x0000FF, 0xaaaaaa, 0xFF0000, 0xFFAA00));
        //&vec!(0xffffff, 0x00AAFF, 0x00FF00, 0x0000FF, 0x000000, 0xFF0000, 0xFFAA00));
    let transform_to_scr = nalgebra::Matrix3::<Real>::new(
        canvas.width as Real, 0., 0.,
        0., -(canvas.height as Real), canvas.height as Real,
        0., 0., 1.);

    canvas.clear(0);
    /*
    canvas.paint_polyloop(
        &bg.vtx2xy_boundary, &transform_to_scr,
        2., 2);
    for p in particles.iter() {
        canvas.paint_point(p.x.x, p.x.y, &transform_to_scr,
                           2., p.c);
    }
    canvas.write();
     */
    let mut i_step = 0;

    loop {
        i_step += 1;
        if i_step % 100 == 0 {
            dbg!(i_step);
        }
        mpm2_p2g(
            &mut bg,
            &particles,
            vol, particle_mass, HARDENING, YOUNG, POISSON,
            DT, is_ours);
        bg.set_boundary(
            &(DT * nalgebra::Vector3::new(0., -200., 0.)), is_full_slip);
        mpm2_g2p(
            &mut particles,
            &bg,
            DT, is_snow, is_ours);
        if i_step % SKIP_FRAME == 0 {
            canvas.clear(0);
            canvas.paint_polyloop(
                &bg.vtx2xy_boundary, &transform_to_scr,
                2.0, 4);
            for p in particles.iter() {
                canvas.paint_point(p.x.x, p.x.y, &transform_to_scr,
                                   6., p.c);
            }
            /*
            for p in &bg.gpbc2xy {
                canvas.paint_point(p.x, p.y, &transform_to_scr,
                                   1.5, 3);
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
             */
            canvas.write();
        }
        if i_step > 10000 { break; }
    }
}


fn main() {
    sim(true, true, true, "target/9_our_snow.gif");
    sim(true, false, true,"target/9_our_hyper.gif");
}

