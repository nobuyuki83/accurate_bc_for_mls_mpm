//! example ported from  https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::AsPrimitive;

type Real = f32;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;


fn main() {
    let mut bg = mpm2::background::Background::new(80);
    for ig in 0..bg.vm.len() {
        let xy = bg.xy(ig);
        bg.vm[ig].x = xy.x;
        bg.vm[ig].y = xy.y;
        bg.vm[ig].z = 0.;
    }
    {
        use rand::Rng;
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
        for _i in 0..100 {
            let x: Real = (rng.gen::<Real>() * 2. - 1.) * 0.5 + 0.5;
            let y: Real = (rng.gen::<Real>() * 2. - 1.) * 0.5 + 0.5;
            let mut cmat = nalgebra::Matrix3::<Real>::zeros();
            let mut rvec = nalgebra::Matrix3x2::<Real>::repeat(0.);
            let samples = bg.near_samples(&Vector::new(x, y));
            for smpl in samples.iter() {
                assert!(smpl.1.x.abs() < 1.5 * bg.dx);
                assert!(smpl.1.y.abs() < 1.5 * bg.dx);
                let w = smpl.2;
                cmat[(0, 0)] += w;
                let mut a = cmat.view_mut((0, 1), (1, 2));
                a += w * smpl.1.transpose();
                let mut a = cmat.view_mut((1, 0), (2, 1));
                a += w * smpl.1;
                let mut a = cmat.view_mut((1, 1), (2, 2));
                a += w * smpl.1 * smpl.1.transpose();
                let ig = smpl.0;
                // dbg!(ig, bg.vm[ig].x);
                rvec[(0, 0)] += bg.vm[ig].x * w;
                rvec[(1, 0)] += bg.vm[ig].x * w * smpl.1.x;
                rvec[(2, 0)] += bg.vm[ig].x * w * smpl.1.y;
                rvec[(0, 1)] += bg.vm[ig].y * w;
                rvec[(1, 1)] += bg.vm[ig].y * w * smpl.1.x;
                rvec[(2, 1)] += bg.vm[ig].y * w * smpl.1.y;
            }
            let asol = cmat.try_inverse().unwrap() * rvec;
            if samples.len() > 4 {
                dbg!(asol);
                const EPS: Real = 1.0e-4;
                assert!((asol[(0, 0)] - x).abs() < EPS);
                assert!((asol[(1, 0)] - 1.0).abs() < EPS);
                assert!(asol[(2, 0)].abs() < EPS);
                assert!((asol[(0, 1)] - y).abs() < EPS);
                assert!(asol[(1, 1)].abs() < EPS);
                assert!((asol[(2, 1)] - 1.0).abs() < EPS);
            }
        }
    }
    {
        let mut canvas = del_canvas::canvas::Canvas::new((800, 800));
        let transform_to_scr = nalgebra::Matrix3::<f32>::new(
            canvas.width as f32, 0., 0.,
            0., -(canvas.height as f32), canvas.height as f32,
            0., 0., 1.);
        canvas.clear(0x003300);
        for p in bg.points.iter() {
            canvas.paint_point(
                p.x, p.y, &transform_to_scr,
                2.0, 0xffffff);
        }
        canvas.write(std::path::Path::new("target/3.png"));
    }
}


