//! refactored example from https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::identities::Zero;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;



fn interpolation(
    bg: &mpm2::background2::Grid::<Real>,
    x: &Vector,
    sdf: &[Real],
    is_sample_exterior: bool) -> Real
{
    let gds = if is_sample_exterior {
        bg.near_grid_points(x)
    } else {
        bg.near_interior_grid_boundary_points(x)
    };
    if gds.len() < 3 {
        return 0.;
    }
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
        if d_inv.is_none() { return 0. as Real; }
        d_inv.unwrap()
    };
    let mut val: Real = 0.;
    for &gd in gds.iter() {
        let q = nalgebra::Vector3::<Real>::new(1., gd.1.x, gd.1.y);
        let tmp = (mat_d_inv * q).scale(gd.2);
        if gd.0 < bg.m * bg.m {
            val += tmp.x * sdf[gd.0];
        }
    }
    val
}

fn main() {
    let vtx2xy_boundary = {
        let boundary = 0.059;
        [
            //boundary, boundary,
            boundary, 0.5,
            1. - boundary, boundary,
            1. - boundary, 1. - boundary,
            boundary, 1. - boundary]
    };

    let sdf = {
        let mut sdf = vec!(0.; M * M);
        for i_grid in 0..M {
            for j_grid in 0..M {  // For all grid nodes
                let x = i_grid as Real / N as Real;
                let y = j_grid as Real / N as Real;
                let is_inside = del_msh::polyloop2::is_inside(&vtx2xy_boundary, &[x, y]);
                if !is_inside { continue; }
                let dist = del_msh::polyloop2::distance_to_point(&vtx2xy_boundary, &[x, y]);
                sdf[j_grid * M + i_grid] = dist;
            }
        }
        sdf
    };

    // let colormap = mpm2::colormap::COLORMAP_JET;
    let colormap = del_canvas::colormap::COLORMAP_PLASMA;
    let colormap_max = 0.3;
    let circle_rad = 4.0;

    const N: usize = 30;
    const M: usize = N + 1;

    const CANVAS_SIZE: (usize, usize) = (1200, 1200);

    let transform_xy2pix = nalgebra::Matrix3::<Real>::new(
        CANVAS_SIZE.0 as Real, 0., 0.,
        0., -(CANVAS_SIZE.1 as Real), CANVAS_SIZE.1 as Real,
        0., 0., 1.);
    let transform_pix2xy = transform_xy2pix.try_inverse().unwrap();

    {   // our method (sample boundary)
        let bg = mpm2::background2::Grid::new(
            N, &vtx2xy_boundary,
            true, 1. / N as f64 * 0.25);
        {
            let mut canvas = del_canvas::canvas::Canvas::new(CANVAS_SIZE);
            canvas.clear(0);
            for iw in 0..canvas.width {
                for ih in 0..canvas.height {
                    let pix = nalgebra::Vector3::<Real>::new(iw as Real, ih as Real, 1.);
                    let xy = transform_pix2xy * pix;
                    let xy = Vector::new(xy.x, xy.y);
                    let v = interpolation(&bg, &xy, &sdf, false) * (N as Real);
                    let rgb = del_canvas::colormap::apply_colormap(
                        v, 0., colormap_max,
                        colormap
                    );
                    canvas.paint_pixel(iw, ih, (rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8);
                }
            }
            canvas.write(std::path::Path::new("target/8_ours.png"));
        }
        { // output inside gridpoint
            let path = "target/8_gridpoints_in.svg";
            let mut file = std::fs::File::create(path).expect("file not found.");
            use std::io::Write;
            writeln!(file, "<svg width=\"{}\" height=\"{}\">", CANVAS_SIZE.0, CANVAS_SIZE.1).expect("cannot write.");
            let _ = writeln!(file, "<polygon points=\"{}\" stroke=\"black\" stroke-width=\"2\" fill=\"none\"/>",
                             del_msh::polyloop2::to_svg(&[0., 0., 1., 0., 1., 1., 0., 1.], &transform_xy2pix));
            for i in 0..M {
                for j in 0..M {
                    let dh = 1.0 / N as Real;
                    let x = dh * i as Real;
                    let y = dh * j as Real;
                    let a = transform_xy2pix * nalgebra::Vector3::<Real>::new(x, y, 1.);
                    if bg.is_inside[j * M + i] {
                        let _ = writeln!(file, "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"#9999FF\" />", a.x, a.y, circle_rad);
                    }
                }
            }
            writeln!(file, "</svg>").expect("cannot write");
        }
        { // output inside gridpoint
            let path = "target/8_gridpoints_out.svg";
            let mut file = std::fs::File::create(path).expect("file not found.");
            use std::io::Write;
            writeln!(file, "<svg width=\"{}\" height=\"{}\">", CANVAS_SIZE.0, CANVAS_SIZE.1).expect("cannot write.");
            let _ = writeln!(file, "<polygon points=\"{}\" stroke=\"black\" stroke-width=\"2\" fill=\"none\"/>",
                             del_msh::polyloop2::to_svg(&[0., 0., 1., 0., 1., 1., 0., 1.], &transform_xy2pix));
            for i in 0..M {
                for j in 0..M {
                    let dh = 1.0 / N as Real;
                    let x = dh * i as Real;
                    let y = dh * j as Real;
                    let a = transform_xy2pix * nalgebra::Vector3::<Real>::new(x, y, 1.);
                    if !bg.is_inside[j * M + i] {
                        let _ = writeln!(file, "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"#99FF99\" />", a.x, a.y, circle_rad);
                    }
                }
            }
            writeln!(file, "</svg>").expect("cannot write");
        }
        { // output bc gridpoint
            let path = "target/8_gridpoints_bc.svg";
            let mut file = std::fs::File::create(path).expect("file not found.");
            use std::io::Write;
            writeln!(file, "<svg width=\"{}\" height=\"{}\">", CANVAS_SIZE.0, CANVAS_SIZE.1).expect("cannot write.");
            let _ = writeln!(file, "<polygon points=\"{}\" stroke=\"black\" stroke-width=\"2\" fill=\"none\"/>",
                             del_msh::polyloop2::to_svg(&[0., 0., 1., 0., 1., 1., 0., 1.], &transform_xy2pix));
            for p in &bg.gpbc2xy {
                let a = transform_xy2pix * nalgebra::Vector3::<Real>::new(p.x, p.y, 1.);
                let _ = writeln!(file, "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"#99FF99\" />", a.x, a.y,circle_rad);
            }
            writeln!(file, "</svg>").expect("cannot write");
        }
    }

    { // output boundary
        let path = "target/8_boundary.svg";
        let mut file = std::fs::File::create(path).expect("file not found.");
        use std::io::Write;
        writeln!(file, "<svg width=\"{}\" height=\"{}\">", CANVAS_SIZE.0, CANVAS_SIZE.1).expect("cannot write.");
        let _ = writeln!(file, "<polygon points=\"{}\" stroke=\"black\" stroke-width=\"2\" fill=\"none\"/>",
                 del_msh::polyloop2::to_svg(&vtx2xy_boundary, &transform_xy2pix));
        let _ = writeln!(file, "<polygon points=\"{}\" stroke=\"black\" stroke-width=\"2\" fill=\"none\"/>",
                 del_msh::polyloop2::to_svg(&[0., 0., 1., 0., 1., 1., 0., 1.], &transform_xy2pix));
        writeln!(file, "</svg>").expect("cannot write");
    }

    { // output all gridpoint
        let path = "target/8_gridpoints_all.svg";
        let mut file = std::fs::File::create(path).expect("file not found.");
        use std::io::Write;
        writeln!(file, "<svg width=\"{}\" height=\"{}\">", CANVAS_SIZE.0, CANVAS_SIZE.1).expect("cannot write.");
        let _ = writeln!(file, "<polygon points=\"{}\" stroke=\"black\" stroke-width=\"2\" fill=\"none\"/>",
                         del_msh::polyloop2::to_svg(&[0., 0., 1., 0., 1., 1., 0., 1.], &transform_xy2pix));
        for i in 0..M {
            for j in 0..M {
                let dh = 1.0 / N as Real;
                let x = dh * i as Real;
                let y = dh * j as Real;
                let a = transform_xy2pix * nalgebra::Vector3::<Real>::new(x, y, 1.);
                let _ = writeln!(file, "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"#FF5555\" />", a.x, a.y, circle_rad);
            }
        }
        writeln!(file, "</svg>").expect("cannot write");
    }

    {   // our method (sample boundary)
        let bg = mpm2::background2::Grid::new(
            N, &vtx2xy_boundary,
            false, 1. / N as f64 * 0.25);
        {
            let mut canvas = del_canvas::canvas::Canvas::new(CANVAS_SIZE);
            canvas.clear(0);
            for iw in 0..canvas.width {
                for ih in 0..canvas.height {
                    let pix = nalgebra::Vector3::<Real>::new(iw as Real, ih as Real, 1.);
                    let xy = transform_pix2xy * pix;
                    let xy = Vector::new(xy.x, xy.y);
                    let v = interpolation(&bg, &xy, &sdf, true) * (N as Real);
                    let rgb = del_canvas::colormap::apply_colormap(
                        v, 0., colormap_max,
                        colormap);
                    canvas.paint_pixel(iw, ih, (rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8);
                }
            }
            canvas.write(std::path::Path::new("target/8_naive.png"));
        }
    }

    {
        let mut canvas = del_canvas::canvas::Canvas::new(CANVAS_SIZE);
        canvas.clear(0);
        for iw in 0..canvas.width {
            for ih in 0..canvas.height {
                let pix = nalgebra::Vector3::<Real>::new(iw as Real, ih as Real, 1.);
                let xy = transform_pix2xy * pix;
                let xy = Vector::new(xy.x, xy.y);
                let mut dist = 0.;
                {
                    let is_inside = del_msh::polyloop2::is_inside(&vtx2xy_boundary, xy.as_slice());
                    if is_inside {
                        dist = del_msh::polyloop2::distance_to_point(&vtx2xy_boundary, xy.as_slice());
                    }
                }
                let dist = dist * 30.;
                let rgb = del_canvas::colormap::apply_colormap(
                    dist, 0., colormap_max,
                    colormap);
                canvas.paint_pixel(iw, ih, (rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8);
            }
        }
        canvas.write(std::path::Path::new("target/8_gt.png"));
    }

    {
        let mut canvas = del_canvas::canvas::Canvas::new((255,20));
        canvas.clear(0);
        for iw in 0..canvas.width {
            for ih in 0..canvas.height {
                let dist = iw as Real / canvas.width as Real;
                let rgb = del_canvas::colormap::apply_colormap(
                    dist, 0., 1.0,
                    colormap);
                canvas.paint_pixel(iw, ih, (rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8);
            }
        }
        canvas.write(std::path::Path::new("target/legend.png"));
    }

    /*
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
     */

}


