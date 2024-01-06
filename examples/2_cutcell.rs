type Vector = nalgebra::Vector2::<f32>;

fn intersections_polygon_gird(poly0: &Vec::<Vector>, n: usize)
                              -> Vec::<Vector>
{
    let inv_dx = n as f32;
    let dx = 1.0 / inv_dx;
    let mut poly1 = Vec::<Vector>::new();
    for i_edge in 0..poly0.len() {
        let p0 = poly0[i_edge] * inv_dx;
        let p1 = poly0[(i_edge + 1) % poly0.len()] * inv_dx;
        let len = (p1 - p0).norm();
        let d = (p1 - p0).normalize();
        poly1.push(p0 * dx);
        let mut p = p0;
        let mut tx = if d.x > 0. {
            (p.x.ceil() - p.x) / d.x
        } else if d.x < 0. {
            (p.x.floor() - p.x) / d.x
        } else { f32::MAX };
        let mut ty = if d.y > 0. {
            (p.y.ceil() - p.y) / d.y
        } else if d.y < 0. {
            (p.y.floor() - p.y) / d.y
        } else { f32::MAX };
        loop {
            let (mut tx_next, mut ty_next) = (tx, ty);
            if tx < ty { // point on vertical edge
                if tx > len { break; }
                p = p0 + tx * d;
                tx_next = if d.x > 0. { tx + 1. / d.x } else if d.x < 0. { tx - 1. / d.x } else { f32::MAX };
            } else { // point on horizontal edge
                if ty > len { break; }
                p = p0 + ty * d;
                ty_next = if d.y > 0. { ty + 1. / d.y } else if d.y < 0. { ty - 1. / d.y } else { f32::MAX };
            }
            poly1.push(p * dx);
            (tx, ty) = (tx_next, ty_next);
        }
    }
    poly1
}

fn main() {
    const N: usize = 10;

    let poly0 = vec!(
        Vector::new(0.141, 0.151),
        Vector::new(0.883, 0.151),
        Vector::new(0.511, 0.681),
    );

    let poly1 = intersections_polygon_gird(&poly0, N);

    let mut canvas = del_canvas::canvas::Canvas::new((827, 827));
    let transform_to_scr = nalgebra::Matrix3::<f32>::new(
        canvas.width as f32, 0., 0.,
        0., -(canvas.height as f32), canvas.height as f32,
        0., 0., 1.);

    for i in 0..N + 1 {
        let x = i as f32 / N as f32;
        canvas.paint_line(
            x, 0., x, 1.,
            &transform_to_scr, 0.5, 0x00888888);
    }
    for j in 0..N + 1 {
        let y = j as f32 / N as f32;
        canvas.paint_line(
            0., y, 1., y, &transform_to_scr,
            0.5, 0x00888888);
    }
    for iline in 0..poly1.len() {
        let ip0 = iline;
        let ip1 = (iline + 1) % poly1.len();
        canvas.paint_line(
            poly1[ip0].x, poly1[ip0].y, poly1[ip1].x, poly1[ip1].y, &transform_to_scr,
            2., 0x00ffffff);
    }
    for p in poly1.iter() {
        canvas.paint_point(
            p.x, p.y, &transform_to_scr,
            4., 0x00ff00ff);
    }

    canvas.write(&std::path::Path::new("target/2.png"));
}