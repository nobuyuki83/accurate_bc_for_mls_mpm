extern crate core;

use num_traits::AsPrimitive;

pub mod canvas;
pub mod canvas_gif;
pub mod background;
pub mod background2;
pub mod particle_solid;
pub mod particle_fluid;
pub mod colormap;

pub fn myclamp<T>(
    v: T,
    vmin: T,
    vmax: T,
) -> T
    where T: std::cmp::PartialOrd
{
    let v0 = if v < vmin { vmin } else { v };
    if v0 > vmax { vmax } else { v0 }
}

pub fn pf<Real>(
    defgrad: &nalgebra::Matrix2<Real>,
    hardening: Real,
    young: Real,
    nu: Real,
    det_defgrad_plastic: Real)
    -> nalgebra::Matrix2<Real>
    where Real: nalgebra::RealField + num_traits::Float
{
    let one = Real::one();
    let two = one + one;
    let mu_0 = young / (two * (one + nu));
    let lambda_0 = young * nu / ((one + nu) * (one - two * nu));
    let e: Real = num_traits::Float::exp(hardening * (one - det_defgrad_plastic));
    let mu = mu_0 * e;
    let lambda = lambda_0 * e;
    let volratio: Real = defgrad.determinant(); // J
    let (r, _s) = del_geo::mat2::polar_decomposition(defgrad);
    (defgrad - r).scale(two * mu) * (defgrad).transpose() + nalgebra::Matrix2::<Real>::identity().scale(lambda * (volratio - one) * volratio)
}

pub fn clip_strain<Real>(
    defgrad0: &nalgebra::Matrix2::<Real>)
    -> nalgebra::Matrix2<Real>
    where Real: nalgebra::RealField + 'static + Copy,
          f64: AsPrimitive<Real>,
{
    let svd = defgrad0.svd(true, true);
    let svd_u0: nalgebra::Matrix2::<Real> = svd.u.unwrap();
    let mut sig0: nalgebra::Matrix2::<Real> = nalgebra::Matrix2::<Real>::from_diagonal(&svd.singular_values);
    let svd_v0 = svd.v_t.unwrap().transpose();
    for i in 0..2 {  // Snow Plasticity
        sig0[(i, i)] = myclamp(sig0[(i, i)], (1_f64 - 2.5e-2_f64).as_(), (1.0 + 7.5e-3).as_());
    }
    svd_u0 * sig0 * svd_v0.transpose()
}


pub fn grid_datas<Real>(
    pos_in: &nalgebra::Vector2::<Real>,
    dx: Real,
    inv_dx: Real,
    n: usize) -> Vec<(usize, nalgebra::Vector2::<Real>, Real)>
    where Real: nalgebra::RealField + 'static + Copy + AsPrimitive<i32>,
          f64: AsPrimitive<Real>,
          i32: AsPrimitive<Real>,
          usize: AsPrimitive<Real>
{
    let base_coord = pos_in * inv_dx - nalgebra::Vector2::<Real>::repeat(0.5.as_());
    let base_coord = nalgebra::Vector2::<i32>::new(
        base_coord.x.as_(), // e.g., "3.6 -> 3", "3.1 -> 2"
        base_coord.y.as_());
    let fx = pos_in * inv_dx - nalgebra::Vector2::<Real>::new(
        base_coord.x.as_(),
        base_coord.y.as_());
    let wxy = {
        let a = nalgebra::Vector2::<Real>::repeat(1.5.as_()) - fx;
        let b = fx - nalgebra::Vector2::<Real>::repeat(1.0.as_());
        let c = fx - nalgebra::Vector2::<Real>::repeat(0.5.as_());
        [
            a.component_mul(&a).scale(0.5.as_()),
            nalgebra::Vector2::<Real>::repeat(0.75.as_()) - b.component_mul(&b),
            c.component_mul(&c).scale(0.5.as_())
        ]
    };
    let mut res = Vec::<(usize, nalgebra::Vector2::<Real>, Real)>::new();
    res.reserve(9);
    for i in 0..3_usize {
        for j in 0..3_usize {
            let dpos = nalgebra::Vector2::<Real>::new(
                (i.as_() - fx.x) * dx,
                (j.as_() - fx.y) * dx);
            let iw = (base_coord.x + i as i32) as usize;
            let ih = (base_coord.y + j as i32) as usize;
            let w = wxy[i].x * wxy[j].y;
            let ig = ih * (n + 1) + iw;
            res.push((ig, dpos, w));
        }
    }
    res
}