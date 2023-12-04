extern crate core;

use num_traits::AsPrimitive;

pub mod canvas;
pub mod canvas_gif;
pub mod background;

pub fn polar_decomposition<T>(
    m: &nalgebra::Matrix2::<T>)
    -> (nalgebra::Matrix2::<T>, nalgebra::Matrix2::<T>)
// where T : num_traits::Float + nalgebra::Scalar + std::ops::AddAssign + std::ops::MulAssign
    where T : nalgebra::RealField + Copy
{
    let x = m[(0, 0)] + m[(1, 1)];
    let y = m[(1, 0)] - m[(0, 1)];
    let scale = T::one() / (x * x + y * y).sqrt();
    let c = x * scale;
    let s = y * scale;
    let u_mat = nalgebra::Matrix2::<T>::new(
        c, -s,
        s, c);
    let p_mat = u_mat * m;
    (u_mat, p_mat)
}

pub fn myclamp<T>(
    v: T,
    vmin: T,
    vmax: T
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
    let (r, _s) = polar_decomposition(defgrad);
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