/// example ported from  https://github.com/yuanming-hu/taichi_mpm

extern crate core;

use num_traits::AsPrimitive;

type Real = f64;
type Vector = nalgebra::Vector2<Real>;
type Matrix = nalgebra::Matrix2<Real>;

fn weight_spline_quad<Real>(x: Real) -> Real
    where Real: num_traits::Float + 'static,
          f64: AsPrimitive<Real>
{
    let half = 0.5_f64.as_();
    if x.abs() < half {
        0.75.as_() - x * x
    } else {
        (1.5.as_() - x.abs()) * (1.5.as_() - x.abs()) * half
    }
}

struct Background {
    pub n: usize,
    pub m: usize,
    pub dx: Real,
    pub inv_dx: Real,
    pub points: Vec<Vector>,
    pub kdtree2: del_geo::kdtree2::KdTree2<Real>,
    pub vm: Vec<nalgebra::Vector3::<Real>>, // velocity and mass
}

impl Background {
    fn new(n_: usize) -> Self {
        let m = n_ + 1;
        let points_ = {
            use rand::Rng;
            let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
            let center = Vector::new(0.5, 0.5);
            let mut ps = Vec::<Vector>::new();
            for _i in 0..1000 {
                /*
                let i = (rng.gen::<Real>() * (m as Real) * (m as Real)) as usize;
                let h = i / m;
                let w = i - h * m;
                let x = (w as Real) / n_ as Real;
                let y = (h as Real) / n_ as Real;
                ps.push(Vector::new(x, y));
                 */
                let x: Real = (rng.gen::<Real>() * 2. - 1.) * 0.3 + center.x;
                let y: Real = (rng.gen::<Real>() * 2. - 1.) * 0.3 + center.y;
                ps.push(Vector::new(x, y));
            }
            ps
        };
        let np = (n_ + 1) * (n_ + 1) + points_.len();
        let vm_ = vec!(nalgebra::Vector3::<Real>::repeat(0.); np);
        Self {
            n: n_,
            m: m,
            dx: 1. / n_ as Real,
            inv_dx: n_ as Real,
            kdtree2: del_geo::kdtree2::KdTree2::from_vec(&points_),
            points: points_,
            vm: vm_,
        }
    }

    fn xy(&self, ig: usize) -> Vector {
        assert!(ig < self.vm.len());
        if ig < self.m * self.m {
            let h = ig / self.m;
            let w = ig - h * self.m;
            return Vector::new(w as Real * self.dx, h as Real * self.dx);
        } else {
            let ip = ig - self.m * self.m;
            return self.points[ip];
        }
    }

    pub fn near_samples(&self, pos_in: &Vector)
                        -> Vec<(usize, Vector, Real)> {
        let base_coord = pos_in * self.inv_dx - Vector::repeat(0.5);
        let base_coord = nalgebra::Vector2::<i32>::new(
            base_coord.x.floor() as i32, // e.g., "3.6 -> 3", "3.1 -> 2"
            base_coord.y.floor() as i32);
        let fx = pos_in * self.inv_dx - base_coord.cast::<Real>();
        let wxy = {
            let a = Vector::repeat(1.5) - fx;
            let b = fx - Vector::repeat(1.0);
            let c = fx - Vector::repeat(0.5);
            [
                0.5 * a.component_mul(&a),
                Vector::repeat(0.75) - b.component_mul(&b),
                0.5 * c.component_mul(&c)
            ]
        };
        let mut res = Vec::<(usize, Vector, Real)>::new();
        res.reserve(9);
        for i in 0..3 as usize {
            for j in 0..3 as usize {
                let dpos = Vector::new(
                    (i as Real - fx.x) * self.dx,
                    (j as Real - fx.y) * self.dx);
                let (iw, ih) = (base_coord.x + i as i32, base_coord.y + j as i32);
                if iw < 0 || iw > self.n as i32 { continue; }
                if ih < 0 || ih > self.n as i32 { continue; }
                let (iw, ih) = (iw as usize, ih as usize);
                let weight = wxy[i].x * wxy[j].y;
                let ig = ih * (self.n + 1) + iw;
                res.push((ig, dpos, weight));
            }
        }
        for (ip, p) in self.points.iter().enumerate() {
            let diffx = p.x - pos_in.x;
            let diffy = p.y - pos_in.y;
            if diffx.abs() > 1.5 * self.dx { continue; }
            if diffy.abs() > 1.5 * self.dx { continue; }
            let wx = weight_spline_quad(diffx * self.inv_dx);
            let wy = weight_spline_quad(diffy * self.inv_dx);
            res.push((ip + self.m * self.m, p - pos_in, wx * wy));
        }
        res
    }
}


fn main() {
    let mut bg = Background::new(80);
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
            if samples.len() == 10 {
                const EPS: f64 = 1.0e-8;
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
        let mut canvas = mpm2::canvas::Canvas::new((800, 800));
        canvas.clear(0x003300);
        for p in bg.points.iter() {
            canvas.paint_circle(
                p.x * canvas.width as Real,
                p.y * canvas.height as Real,
                2.0, 0xffffff);
        }
        canvas.write(std::path::Path::new("3.png"));
    }
}


