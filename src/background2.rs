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

pub struct Grid<Real> {
    pub n: usize,
    pub m: usize,
    pub dx: Real,
    pub inv_dx: Real,
    pub points: Vec<nalgebra::Vector2::<Real>>,
    // additional points
    pub kdtree: Vec<usize>,
    pub vm: Vec<nalgebra::Vector3::<Real>>,
    // velocity and mass
    pub vtx2xy_boundary: Vec<Real>,
    pub is_inside: Vec<bool>,
    pub gp2mass: Vec<Real>,
}

impl<Real> Grid<Real>
    where Real: num_traits::Float + Copy + 'static + nalgebra::RealField + num_traits::AsPrimitive<i32>,
          usize: num_traits::AsPrimitive<Real>,
          i32: num_traits::AsPrimitive<Real>,
          f64: num_traits::AsPrimitive<Real>,
          rand::distributions::Standard: rand::prelude::Distribution<Real>
{
    // calc distance between a point p and a line segment a-b
    fn point_line_dist(
        p: nalgebra::Vector2::<Real>,
        a: nalgebra::Vector2::<Real>, 
        b: nalgebra::Vector2::<Real>) -> Real
    {
        let pa = p - a;
        let pb = p - b;
        if pa.dot(&(b - a)) < 0.as_() {
            return pa.norm();
        }
        else if pb.dot(&(a - b)) < 0.as_() {
            return pb.norm();
        }
        else {
            return num_traits::Float::abs((b.y - a.y)*p.x - (b.x - a.x)*p.y + b.x*a.y - b.y*a.x) / (b - a).norm();
        }
    }

    pub fn new(n: usize, vtx2xy_boundary: &[Real], is_sample_boundary: bool,
        ignore_near_grid_point: bool, ignore_threshold: Real) -> Self {
        let dx = Real::one() / n.as_();
        let m = n + 1;
        let points = {
            let mut ps = Vec::<nalgebra::Vector2::<Real>>::new();
            if is_sample_boundary {
                let np = vtx2xy_boundary.len() / 2;
                for ip in 0..np {
                    let jp = (ip + 1) % np;
                    let pi = nalgebra::Vector2::<Real>::from_row_slice(
                        &vtx2xy_boundary[ip * 2 + 0..ip * 2 + 2]);
                    ps.push(nalgebra::Vector2::<Real>::new(pi[0], pi[1]));
                    //
                    let pj = nalgebra::Vector2::<Real>::from_row_slice(
                        &vtx2xy_boundary[jp * 2 + 0..jp * 2 + 2]);
                    let len = (pi - pj).norm();
                    let ndiv = len / dx;
                    if ndiv < 2f64.as_() { continue; }
                    let ndiv: usize = ndiv.as_().try_into().unwrap();
                    let cell_len = len / ndiv.as_();
                    for idiv in 0..ndiv - 1 {
                        let p = pi + (pj - pi).normalize() * (cell_len * (idiv + 1).as_());
                        ps.push(p);
                    }
                }
            }
            ps
        };
        let is_inside: Vec<bool> = {
            let mut is_inside = vec!(true; m * m);
            for i_grid in 0..m {
                for j_grid in 0..m {  // For all grid nodes
                    let x = i_grid.as_() / n.as_();
                    let y = j_grid.as_() / n.as_();
                    if del_msh::polyloop2::is_inside(vtx2xy_boundary, &[x, y]) {
                        // ignore grid points which is near the boundary points
                        let mut is_too_near = false;
                        if ignore_near_grid_point {
                            let g = nalgebra::Vector2::<Real>::new(x, y);

                            // visit all the boudnary
                            let np = vtx2xy_boundary.len() / 2;
                            for ip in 0..np {
                                let jp = (ip + 1) % np;
                                let pi = nalgebra::Vector2::<Real>::from_row_slice(
                                    &vtx2xy_boundary[ip * 2 + 0..ip * 2 + 2]);
                                let pj = nalgebra::Vector2::<Real>::from_row_slice(
                                    &vtx2xy_boundary[jp * 2 + 0..jp * 2 + 2]);

                                let dist = Self::point_line_dist(g, pi, pj);
                                if dist < ignore_threshold {
                                    is_too_near = true;
                                    break;
                                }
                            }
                        }

                        if !is_too_near {
                            continue;
                        }
                    }
                    is_inside[j_grid * m + i_grid] = false;
                }
            }
            is_inside
        };
        let vm = {
            let np = m * m + points.len();
            vec!(nalgebra::Vector3::<Real>::repeat(Real::zero()); np)
        };
        let kdtree = {
            let mut pairs = Vec::<(nalgebra::Vector2::<Real>, usize)>::new();
            for (iv, xy) in points.iter().enumerate() {
                pairs.push((*xy, iv));
            }
            let mut kdtree = Vec::<usize>::new();
            del_msh::kdtree2::construct_kdtree(
                &mut kdtree,
                0, &mut pairs, 0, points.len(),
                0);
            kdtree
        };
        Self {
            n,
            m,
            dx,
            inv_dx: n.as_(),
            kdtree,
            points,
            vm,
            vtx2xy_boundary: Vec::<Real>::from(vtx2xy_boundary),
            is_inside,
            gp2mass: vec!(Real::zero(); m * m),
        }
    }

    pub fn xy(&self, ig: usize) -> nalgebra::Vector2::<Real> {
        assert!(ig < self.vm.len());
        if ig < self.m * self.m {
            let h = ig / self.m;
            let w = ig - h * self.m;
            nalgebra::Vector2::<Real>::new(w.as_() * self.dx, h.as_() * self.dx)
        } else {
            let ip = ig - self.m * self.m;
            self.points[ip]
        }
    }

    pub fn near_interior_grid_boundary_points(&self, pos_in: &nalgebra::Vector2::<Real>)
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
                let ig = ih * (self.n + 1) + iw;
                if !self.is_inside[ig] { continue; }
                // let wx = weight_spline_quad(dpos.x * self.inv_dx);
                // let wy = weight_spline_quad(dpos.y * self.inv_dx);
                // let weight = wx * wy;
                let weight = wxy[i].x * wxy[j].y;
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


    pub fn near_grid_points(&self, pos_in: &nalgebra::Vector2::<Real>)
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
                let ig = ih * (self.n + 1) + iw;
                let weight = wxy[i].x * wxy[j].y;
                res.push((ig, dpos, weight));
            }
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

    pub fn set_boundary(&mut self, gravity_dt: &nalgebra::Vector3::<Real>) {
        for i in 0..self.vm.len() {
            let g0 = &mut self.vm[i];
            if g0.z <= Real::zero() { continue; } // grid is empty
            *g0 /= g0.z;
            *g0 += gravity_dt;
            // left, right, top is sticky
            if i < self.m * self.m {
                if !self.is_inside[i] {
                    *g0 = nalgebra::Vector3::repeat(Real::zero()); // Sticky
                }
            } else {
                *g0 = nalgebra::Vector3::repeat(Real::zero()); // Sticky
            }
        }
    }
}