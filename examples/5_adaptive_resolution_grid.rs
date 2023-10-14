use std::ops::Add;

type Real = f32;
type Vec2 = nalgebra::Vector2::<Real>;

enum GridState {
    EMPTY,
    FILLED,
    EDGE
}

fn cross(target : &Vec2, t0 : &Vec2, t1 : &Vec2) -> Real
{ (t1.x - t0.x) * target.y - (t1.y - t0.y) * target.x }

fn is_inside_triangle(target : &Vec2, t0 : &Vec2, t1 : &Vec2, t2 : &Vec2) -> bool
{
    (cross(target, t0, t1) > 0 as Real) &&
    (cross(target, t1, t2) > 0 as Real) &&
    (cross(target, t2, t0) > 0 as Real)
}

fn create_coarse_grid(poly : &Vec::<Vec2>, n : usize) -> Vec::<GridState>
{
    let mut grid = Vec::<GridState>::new();

    let dx = 1.0 / n;

    for j in 0..n {
        for i in 0..n {
            let points : [Vec2; 4];
            points = [
                Vec2::new(i * dx, j * dx),
                Vec2::new((i + 1) * dx, j * dx),
                Vec2::new(i * dx, (j + 1) * dx),
                Vec2::new((i + 1) * dx, (j + 1) * dx)
            ];

            // count the number of points inside the triangle
            let mut count = 0;
            for p in points {
                if is_inside_triangle(&p, &poly[0], &poly[1], &poly[2]) {
                    count += 1;
                }
            }

            // register the state
            if count == 0 {
                grid.push(GridState::EMPTY);
            }
            else if count == 4 {
                grid.push(GridState::FILLED);
            }
            else {
                grid.push(GridState::EDGE);
            }
        }
    }

    grid
}

fn main() {
    const N: usize = 10;
    const M: usize = N + 1;

    let poly0 = vec!(
        Vec2::new(0.141, 0.151),
        Vec2::new(0.883, 0.151),
        Vec2::new(0.511, 0.681),
    );

    let poly1 = intersections_polygon_gird(&poly0, N);

    use mpm2::canvas::Canvas;
    let mut canvas = Canvas::new((827, 827));

    for i in 0..N + 1 {
        let x = i as Real / N as Real * canvas.width as Real;
        canvas.paint_line(x, 0., x, canvas.height as Real,
                          0.5, 0x00888888);
    }
    for j in 0..N + 1 {
        let y = j as Real / N as Real * canvas.height as Real;
        canvas.paint_line(0., y, canvas.width as Real, y,
                          0.5, 0x00888888);
    }
    for iline in 0..poly1.len() {
        let ip0 = iline;
        let ip1 = (iline + 1) % poly1.len();
        canvas.paint_line(
            poly1[ip0].x * canvas.width as Real,
            poly1[ip0].y * canvas.height as Real,
            poly1[ip1].x * canvas.width as Real,
            poly1[ip1].y * canvas.height as Real,
            2., 0x00ffffff);
    }
    for p in poly1.iter() {
        canvas.paint_circle(
            p.x * canvas.width as Real,
            p.y * canvas.height as Real,
            4., 0x00ff00ff);
    }

    canvas.write(&std::path::Path::new("2.png"));
}