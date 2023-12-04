use num_traits::AsPrimitive;

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

pub struct Background <Real>{
    pub n: usize,
    pub m: usize,
    pub dx: Real,
    pub inv_dx: Real,
    pub points: Vec<nalgebra::Vector2::<Real>>,
    pub kdtree: Vec<usize>,
    pub vm: Vec<nalgebra::Vector3::<Real>>, // velocity and mass
}

impl<Real> Background<Real>
where Real: num_traits::Float + Copy + 'static + nalgebra::RealField + num_traits::AsPrimitive<i32>,
      usize: num_traits::AsPrimitive<Real>,
      i32: num_traits::AsPrimitive<Real>,
      f64: num_traits::AsPrimitive<Real>,
      rand::distributions::Standard: rand::prelude::Distribution<Real>
{
    pub fn new(n_: usize) -> Self {
        let m = n_ + 1;
        let points_ = {
            use rand::Rng;
            let two: Real = 2_f64.as_();
            let rad: Real = 0.3f64.as_();
            // let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([13_u8; 32]);
            let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([14_u8; 32]);
            let center = nalgebra::Vector2::<Real>::new(0.5f64.as_(), 0.35f64.as_());
            let mut ps = Vec::<nalgebra::Vector2::<Real>>::new();
            for _i in 0..200 {
                let x: Real = (rng.gen::<Real>() * two - Real::one()) * rad + center[0];
                let y: Real = (rng.gen::<Real>() * two - Real::one()) * rad + center[1];
                let p = nalgebra::Vector2::<Real>::new(x, y);
                ps.push(p);
            }
            ps
        };
        let vm_ = {
            let np = m * m + points_.len();
            vec!(nalgebra::Vector3::<Real>::repeat(Real::zero()); np)
        };
        let kdtree = {
            let mut pairs = Vec::<(nalgebra::Vector2::<Real>,usize)>::new();
            for (iv,xy) in points_.iter().enumerate() {
                pairs.push((*xy,iv));
            }
            let mut kdtree = Vec::<usize>::new();
            del_msh::kdtree2::construct_kdtree(
                &mut kdtree,
                0, &mut pairs, 0, points_.len(),
                0);
            kdtree
        };
        Self {
            n: n_,
            m,
            dx: Real::one() / n_.as_(),
            inv_dx: n_.as_(),
            kdtree,
            points: points_,
            vm: vm_,
        }
    }

    pub fn xy(&self, ig: usize) -> nalgebra::Vector2::<Real> {
        assert!(ig < self.vm.len());
        return if ig < self.m * self.m {
            let h = ig / self.m;
            let w = ig - h * self.m;
            nalgebra::Vector2::<Real>::new(w.as_() * self.dx, h.as_() * self.dx)
        } else {
            let ip = ig - self.m * self.m;
            self.points[ip]
        }
    }

    pub fn near_samples(&self, pos_in: &nalgebra::Vector2::<Real>)
                        -> Vec<(usize, nalgebra::Vector2::<Real>, Real)> {
        let scr_base_pos = pos_in * self.inv_dx - nalgebra::Vector2::<Real>::repeat(0.5f64.as_());
        let scr_base_pos = nalgebra::Vector2::<i32>::new(
            num_traits::Float::floor(scr_base_pos.x).as_(), // e.g., "3.6 -> 3", "3.1 -> 2"
            num_traits::Float::floor(scr_base_pos.y).as_());
        let scr_diff_base = pos_in * self.inv_dx - nalgebra::Vector2::<Real>::new(
            scr_base_pos.x.as_(),
            scr_base_pos.y.as_());
        let wxy = {
            let a = nalgebra::Vector2::<Real>::repeat(1.5.as_()) - scr_diff_base;
            let b = scr_diff_base - nalgebra::Vector2::<Real>::repeat(1.0.as_());
            let c = scr_diff_base - nalgebra::Vector2::<Real>::repeat(0.5.as_());
            let half: Real = 0.5.as_();
            [
                a.component_mul(&a).scale(half),
                nalgebra::Vector2::<Real>::repeat(0.75.as_()) - b.component_mul(&b),
                c.component_mul(&c).scale(half)
            ]
        };
        let mut res = Vec::<(usize, nalgebra::Vector2::<Real>, Real)>::new();
        res.reserve(9);
        for i in 0..3usize {
            for j in 0..3usize {
                let dpos = nalgebra::Vector2::<Real>::new(
                    (i.as_() - scr_diff_base.x) * self.dx,
                    (j.as_() - scr_diff_base.y) * self.dx);
                let (iw, ih) = (scr_base_pos.x + i as i32, scr_base_pos.y + j as i32);
                if iw < 0 || iw > self.n as i32 { continue; }
                if ih < 0 || ih > self.n as i32 { continue; }
                let (iw, ih) = (iw as usize, ih as usize);
                // let wx = weight_spline_quad(dpos.x * self.inv_dx);
                // let wy = weight_spline_quad(dpos.y * self.inv_dx);
                // let weight = wx * wy;
                let weight = wxy[i].x * wxy[j].y;
                let ig = ih * (self.n + 1) + iw;
                res.push((ig, dpos, weight));
            }
        }
        for (ip, p) in self.points.iter().enumerate() {
            let diffx = p.x - pos_in.x;
            let diffy = p.y - pos_in.y;
            if num_traits::Float::abs(diffx) > 1.5.as_() * self.dx { continue; }
            if num_traits::Float::abs(diffy) > 1.5.as_() * self.dx { continue; }
            let wx = weight_spline_quad(diffx * self.inv_dx);
            let wy = weight_spline_quad(diffy * self.inv_dx);
            res.push((ip + self.m * self.m, p - pos_in, wx * wy));
        }
        res
    }

    pub fn after_p2g(&mut self, gravity_times_dt: nalgebra::Vector3::<Real>) {
        for ivm in 0..self.vm.len() {
            let g0 = &mut self.vm[ivm];
            if g0.z <= Real::zero() { continue; } // grid is emptyc
            *g0 /= g0.z;
            *g0 += gravity_times_dt;
        }
        for igrid in 0..self.m {
            for jgrid in 0..self.m {  // For all grid nodes
                let g0 = &mut self.vm[jgrid * self.m + igrid];
                let boundary: Real = 0.05.as_();
                let x: Real = igrid.as_() / self.n.as_();
                let y: Real = jgrid.as_() / self.n.as_();
                if x < boundary || x > Real::one() - boundary || y > Real::one() - boundary {
                    *g0 = nalgebra::Vector3::repeat(Real::zero()); // Sticky
                }
                if y < boundary && g0.y < Real::zero() {
                    g0.y = Real::zero();
                }
            }
        }
    }

    pub fn set_velocity_zero(&mut self) {
        assert_eq!(self.vm.len(), self.m * self.m + self.points.len());
        use num_traits::Zero;
        self.vm.iter_mut().for_each(|p| p.set_zero());
    }
}