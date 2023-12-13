//! hoge

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
    target_density: Real,
    eos_stiffness: Real,
    eos_power: i32,
    dynamic_viscosity: Real,
    dt: Real) {
    assert_eq!(grid.len(), (n + 1) * (n + 1));
    let dx: Real = 1.0 / n as Real;
    let inv_dx: Real = 1.0 / dx;
    // P2G
    for p in particles {
        let gds = mpm2::grid_datas(&p.x, dx, inv_dx, n);
        let mut mass_par_cell = 0.;
        for &gd in gds.iter() {
            mass_par_cell += grid[gd.0].z * gd.2;
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

    const DT: Real = 7e-5;

    const EOS_STIFFNESS: Real = 10.0e+1_f64;
    const EOS_POWER: i32 = 4_i32;
    const DYNAMIC_VISCOSITY: Real = 3e-1f64;
    const TARGET_DENSITY: Real = 40_000_f64; // mass par unit square

    //////

    let area = {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let cell_len = 0.007;
        let vtx2xy0 = del_msh::polyloop2::from_pentagram(&[0.25, 0.55], 0.13);
        let area0 = del_msh::polyloop2::area(&vtx2xy0);
        let xys0 = del_msh::polyloop2::to_uniform_density_random_points(&vtx2xy0, cell_len, &mut rng);
        xys0.chunks(2).for_each(
            |v|
                particles.push(
                    mpm2::particle_a::Particle::new(
                        nalgebra::Vector2::<Real>::new(v[0], v[1]), 1)));
        //
        let vtx2xy1 = del_msh::polyloop2::from_pentagram(&[0.40, 0.75], 0.17);
        let area1 = del_msh::polyloop2::area(&vtx2xy1);
        let xys1 = del_msh::polyloop2::to_uniform_density_random_points(&vtx2xy1, cell_len, &mut rng);
        xys1.chunks(2).for_each(
            |v|
                particles.push(
                    mpm2::particle_a::Particle::new(
                        nalgebra::Vector2::<Real>::new(v[0], v[1]), 2))
        );
        //
        area0 + area1
    };
    let particle_mass: Real = area * TARGET_DENSITY / (particles.len() as Real);

    const N: usize = 80;
    const M: usize = N + 1;
    let mut grid: Vec<nalgebra::Vector3::<Real>> = vec!(
        nalgebra::Vector3::<Real>::new(0., 0., 0.);
        M * M);

    {
        let area_pix = (2. * 0.08 * 2. * 0.08) * ((N * N) as Real);
        let density: Real = 1000.0 / area_pix;
        dbg!(density * (N * N) as Real);
    }

    let boundary = 0.067;
    let vtx2xy_boundary = [
        boundary, 0.5,
        // boundary, boundary,
        1. - boundary, boundary,
        1. - boundary, 1. - boundary,
        boundary, 1. - boundary];

    const FRAME_DT: Real = 50e-3;
    let mut canvas = mpm2::canvas_gif::CanvasGif::new(
        std::path::Path::new("target/7.gif"), (800, 800),
        &vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587, 0xffffff, 0xFF00FF));
    let transform_to_scr = nalgebra::Matrix3::<Real>::new(
        canvas.width as Real, 0., 0.,
        0., -(canvas.height as Real), canvas.height as Real,
        0., 0., 1.);

    canvas.clear(0);
    canvas.paint_polyloop(
        &vtx2xy_boundary, &transform_to_scr,
        2., 2);
    for p in particles.iter() {
        canvas.paint_point(p.x.x, p.x.y, &transform_to_scr,
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
            particle_mass);
        mpm2_p2g_second(
            &mut grid,
            N,
            &particles,
            particle_mass, TARGET_DENSITY,
            EOS_STIFFNESS, EOS_POWER, DYNAMIC_VISCOSITY,
            DT);
        {
            for i_grid in 0..M {
                for j_grid in 0..M {  // For all grid nodes
                    let g0 = &mut grid[j_grid * M + i_grid];
                    if g0.z <= 0. { continue; } // grid is empty
                    *g0 /= g0.z;
                    *g0 += DT * nalgebra::Vector3::new(0., -200., 0.);
                    let x = i_grid as Real / N as Real;
                    let y = j_grid as Real / N as Real;
                    if del_msh::polyloop2::is_inside(&vtx2xy_boundary, &[x,y]) {
                        continue;
                    }
                    *g0 = nalgebra::Vector3::repeat(0.); // Sticky
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
            canvas.paint_polyloop(
                &vtx2xy_boundary, &transform_to_scr,
                2., 2);
            for p in particles.iter() {
                canvas.paint_point(p.x.x, p.x.y, &transform_to_scr,
                                   2., p.c);
            }
            for i in 0..M {
                for j in 0..M {
                    let dh = 1.0 / M as Real;
                    canvas.paint_point(
                        dh * i as Real, dh * j as Real, &transform_to_scr,
                        1., 5);
                }
            }
            canvas.write();
        }
        if istep > 60000 { break; }
    }
}


