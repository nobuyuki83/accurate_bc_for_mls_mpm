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
        return -min_dist;
    } else {
        return min_dist;
    }
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
        mpm2::background2::Grid::new(30, &vtx2xy_boundary, false, false, 0.)
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
                let r = v * 25555.0;
                if r > 0. {
                    canvas.paint_pixel(iw, ih, r as u8, 0, 0); 
                } else {
                    canvas.paint_pixel(iw, ih, 0, -r as u8, 0);
                }
            }
        }
        canvas.paint_polyloop(
            &bg.vtx2xy_boundary, &transform_xy2pix,
            0.6, 0xffffff);

        canvas.write(std::path::Path::new("target/9.png"));
    }
}


