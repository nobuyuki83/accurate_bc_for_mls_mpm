/// example ported from  https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use std::collections::BTreeSet;
use del_geo::kdtree2::Node;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;

fn check(
    points_: &Vec<Vector>, idxs: &Vec<usize>,
    x: Real, y: Real, rad: Real)
{
    let idx_set: BTreeSet::<usize> = idxs.iter().cloned().collect();
    let b = std::collections::BTreeSet::<usize>::from_iter(0..points_.len());
    let comp_idx_set = b.difference(&idx_set);
    for &i in idx_set.iter() {
        let v = &points_[i];
        assert!((v.x - x).abs() < rad && (v.y - y).abs() < rad);
    }
    for &i in comp_idx_set {
        let v = &points_[i];
        assert!((v.x - x).abs() > rad || (v.y - y).abs() > rad);
    }
}

fn main() {
    let points_ = {
        use rand::Rng;
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let center = Vector::new(0.5, 0.5);
        let mut ps = Vec::<Vector>::new();
        for _i in 0..100 {
            let x: Real = (rng.gen::<Real>() * 2. - 1.) * 0.3 + center.x;
            let y: Real = (rng.gen::<Real>() * 2. - 1.) * 0.3 + center.y;
            ps.push(Vector::new(x, y));
        }
        ps
    };
    let nodes = {
        let mut nodes = Vec::<del_geo::kdtree2::Node<Real>>::new();
        let mut ps = points_.iter().enumerate().map(|v| (*v.1, v.0)).collect();
        nodes.resize(1, del_geo::kdtree2::Node::new());
        del_geo::kdtree2::construct_kdtree(&mut nodes, 0,
                                           &mut ps, 0, points_.len(),
                                           0);
        nodes
    };
    {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        let rad = 0.1;
        use rand::Rng;
        for _ in 0..1000 {
            let x: Real = rng.gen::<Real>();
            let y: Real = rng.gen::<Real>();
            let idxs: Vec<usize> = points_.iter().enumerate()
                .filter(|&v| (v.1.x - x).abs() < rad && (v.1.y - y).abs() < rad)
                .map(|v| v.0)
                .collect();
            check(&points_, &idxs, x, y, rad);
        }
    }
    {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([0_u8; 32]);
        use rand::Rng;
        for _ in 0..1000 {
            let p0 = Vector::new(rng.gen::<Real>(), rng.gen::<Real>());
            let mut pos_near = (Vector::new(Real::MAX, Real::MAX), usize::MAX);
            del_geo::kdtree2::nearest(&mut pos_near, p0,
                                      &nodes, 0,
                                      Vector::new(0., 0.), Vector::new(1., 1.),
                                      0);
            dbg!(((pos_near.0 - p0).norm(), pos_near.1));
            let dist_min = (points_[pos_near.1] - p0).norm();
            for i in 0..points_.len() {
                assert!((points_[i] - p0).norm() >= dist_min);
            }
        }
    }
    {   // draw canvas
        let mut canvas = mpm2::canvas::Canvas::new((800, 800));
        canvas.clear(0x003300);
        for p in points_.iter() {
            canvas.paint_circle(
                p.x * canvas.width as Real,
                p.y * canvas.height as Real,
                2.0, 0xffffff);
        }
        {
            let mut xys: Vec<Real> = vec!();
            del_geo::kdtree2::find_edges(
                &mut xys, &nodes,
                0,
                nalgebra::Vector2::<Real>::new(0., 0.),
                nalgebra::Vector2::<Real>::new(1., 1.),
                0);
            for xy in xys.chunks(4) {
                canvas.paint_line(
                    xy[0] * canvas.width as Real,
                    xy[1] * canvas.width as Real,
                    xy[2] * canvas.width as Real,
                    xy[3] * canvas.width as Real,
                    0.5, 0xff0000);
            }
        }
        canvas.write(std::path::Path::new("3.png"));
    }
}



