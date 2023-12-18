//! refactored example from https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;

fn interpolation(
    bg: &mpm2::background2::Grid::<Real>,
    x: &Vector,
    is_sample_exterior: bool) -> Real
{
    let gds = if is_sample_exterior {
        bg.near_grid_points(x)
    } else {
        bg.near_interior_grid_boundary_points(x)
    };
    let mat_d_inv = {
        let mut mat_d = nalgebra::Matrix3::<Real>::zero();
        for &gd in gds.iter() {
            let dpos = gd.1;
            let w = gd.2;
            mat_d[(0, 0)] += w;
            mat_d[(0, 1)] += w * dpos.x;
            mat_d[(0, 2)] += w * dpos.y;
            mat_d[(1, 0)] += w * dpos.x;
            mat_d[(1, 1)] += w * dpos.x * dpos.x;
            mat_d[(1, 2)] += w * dpos.x * dpos.y;
            mat_d[(2, 0)] += w * dpos.y;
            mat_d[(2, 1)] += w * dpos.y * dpos.x;
            mat_d[(2, 2)] += w * dpos.y * dpos.y;
        }
        let d_inv = mat_d.try_inverse();
        if d_inv.is_none() { return 0. as Real }
        d_inv.unwrap()
    };
    let mut val: Real = 0.;
    for &gd in gds.iter() {
        let q = nalgebra::Vector3::<Real>::new(1., gd.1.x, gd.1.y);
        let tmp = (mat_d_inv * q).scale(gd.2);
        if gd.0 < bg.m * bg.m && bg.is_inside[gd.0] {
            val += tmp.x;
        }
    }
    val
}

fn main() {
    let bg = {
        let boundary = 0.047;
        let vtx2xy_boundary = [
            //boundary, boundary,
            boundary, 0.5,
            1. - boundary, boundary,
            1. - boundary, 1. - boundary,
            boundary, 1. - boundary];
        mpm2::background2::Grid::new(30, &vtx2xy_boundary, false)
    };

    let mut canvas = mpm2::canvas::Canvas::new(
        (800, 800));
    //&vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587, 0xffffff, 0xFF00FF, 0xFFFF00));
    let transform_xy2pix = nalgebra::Matrix3::<Real>::new(
        canvas.width as Real, 0., 0.,
        0., -(canvas.height as Real), canvas.height as Real,
        0., 0., 1.);
    let transform_pix2xy = transform_xy2pix.try_inverse().unwrap();

    canvas.clear(0);
    for iw in 0..canvas.width {
        for ih in 0..canvas.height {
            let pix = nalgebra::Vector3::<Real>::new(iw as Real, ih as Real, 1.);
            let xy = transform_pix2xy * pix;
            let xy = Vector::new(xy.x, xy.y);
            let v = interpolation(&bg, &xy, true);
            let r = (v * 255.0) as u8;
            canvas.paint_pixel(iw, ih, r, r, r);
        }
    }
    canvas.paint_polyloop(
        &bg.vtx2xy_boundary, &transform_xy2pix,
        0.6, 0xffffff);
    for p in &bg.points {
        canvas.paint_point(p.x, p.y, &transform_xy2pix,
                           1., 0x0000ff);
    }
    for i in 0..bg.m {
        for j in 0..bg.m {
            let dh = 1.0 / bg.n as Real;
            if bg.is_inside[j * bg.m + i] {
                canvas.paint_point(
                    dh * i as Real, dh * j as Real, &transform_xy2pix,
                    1., 0xff0000);
            } else {
                canvas.paint_point(
                    dh * i as Real, dh * j as Real, &transform_xy2pix,
                    1., 0x00ff00);
            }
        }
    }
    canvas.write(std::path::Path::new("target/8.png"));

    {
        /*
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
         */
    }
}


