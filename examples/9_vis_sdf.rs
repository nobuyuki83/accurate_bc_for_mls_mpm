//! refactored example from https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;

// calc distance between a point p and a line segment a-b
fn point_line_dist(
    p: nalgebra::Vector2::<Real>,
    a: nalgebra::Vector2::<Real>, 
    b: nalgebra::Vector2::<Real>) -> Real
{
    let pa = p - a;
    let pb = p - b;
    if pa.dot(&(b - a)) < 0. as Real {
        return pa.norm();
    }
    else if pb.dot(&(a - b)) < 0. as Real {
        return pb.norm();
    }
    else {
        return num_traits::Float::abs((b.y - a.y)*p.x - (b.x - a.x)*p.y + b.x*a.y - b.y*a.x) / (b - a).norm();
    }
}

fn query_sdf(
    vtx2xy_boundary: &[Real], 
    p: &nalgebra::Vector2::<Real>) -> Real
{
    let mut min_dist: Real = 1000.;
    let np = vtx2xy_boundary.len() / 2;
    for ip in 0..np {
        let jp = (ip + 1) % np;
        let pi = nalgebra::Vector2::<Real>::from_row_slice(
            &vtx2xy_boundary[ip * 2 + 0..ip * 2 + 2]);
        let pj = nalgebra::Vector2::<Real>::from_row_slice(
            &vtx2xy_boundary[jp * 2 + 0..jp * 2 + 2]);
            
        let dist = point_line_dist(*p, pi, pj);
        if min_dist > dist {
            min_dist = dist;
        }
    }

    if del_msh::polyloop2::is_inside(vtx2xy_boundary, &[p.x, p.y]) {
        return min_dist;
    } else {
        return 0.;
    }
}

fn interpolation(
    bg: &mpm2::background2::Grid::<Real>,
    x: &Vector,
    is_ours: bool) -> Real
{
    let gds = if is_ours {
        bg.near_interior_grid_boundary_points(x)
    } else {
        bg.near_grid_points(x)
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

        if is_ours && gd.0 < bg.m * bg.m && !bg.is_inside[gd.0] { continue; }

        val += tmp.x * bg.vm[gd.0].x;
    }
    val
}

fn main() {
    let is_ours: bool = true;

    let mut bg = {
        let boundary = 0.047;
        let vtx2xy_boundary = [
            //boundary, boundary,
            boundary, 0.5,
            1. - boundary, boundary,
            1. - boundary, 1. - boundary,
            boundary, 1. - boundary];
        mpm2::background2::Grid::new(30, &vtx2xy_boundary, is_ours, is_ours, 1. / 120 as Real)
    };

    let mut canvas = mpm2::canvas::Canvas::new(
        (800, 800));
    //&vec!(0x112F41, 0xED553B, 0xF2B134, 0x068587, 0xffffff, 0xFF00FF, 0xFFFF00));
    let transform_xy2pix = nalgebra::Matrix3::<Real>::new(
        canvas.width as Real, 0., 0.,
        0., -(canvas.height as Real), canvas.height as Real,
        0., 0., 1.);
    let transform_pix2xy = transform_xy2pix.try_inverse().unwrap();

    // render ground truth
    {
        canvas.clear(0);
        for iw in 0..canvas.width {
            for ih in 0..canvas.height {
                let pix = nalgebra::Vector3::<Real>::new(iw as Real, ih as Real, 1.);
                let xy = transform_pix2xy * pix;
                let xy = Vector::new(xy.x, xy.y);
                let v = query_sdf(&bg.vtx2xy_boundary, &xy);
                let r = v * 55555.0;
                canvas.paint_pixel(iw, ih, 0, r as u8, 0);
            }
        }
        canvas.paint_polyloop(
            &bg.vtx2xy_boundary, &transform_xy2pix,
            0.6, 0xffffff);

        canvas.write(std::path::Path::new("target/9_ground_truth.png"));
    }

    // render interpolated field
    {
        // assign sdf to grid points (use bg.vm as memory)
        for i in 0..bg.m {
            for j in 0..bg.m {
                let dh = 1.0 / bg.n as Real;
                let xy = nalgebra::Vector2::new(dh * i as Real, dh * j as Real);
                bg.vm[j * bg.m + i].x = query_sdf(&bg.vtx2xy_boundary, &xy);
            }
        }

        for (i, p) in bg.points.iter().enumerate() {
            bg.vm[bg.m * bg.m + i].x = query_sdf(&bg.vtx2xy_boundary, &p)
        }

        canvas.clear(0);
        for iw in 0..canvas.width {
            for ih in 0..canvas.height {
                let pix = nalgebra::Vector3::<Real>::new(iw as Real, ih as Real, 1.);
                let xy = transform_pix2xy * pix;
                let xy = Vector::new(xy.x, xy.y);
                let v = interpolation(&bg, &xy, is_ours);
                let r = v * 55555.0;
                canvas.paint_pixel(iw, ih, 0, r as u8, 0);
            }
        }
        canvas.paint_polyloop(
            &bg.vtx2xy_boundary, &transform_xy2pix,
            0.6, 0xffffff);

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

        for p in &bg.points {
            canvas.paint_point(p.x, p.y, &transform_xy2pix, 1., 0x0000ff);
        }

        canvas.write(std::path::Path::new("target/9_interpolated.png")); 
    }
}


