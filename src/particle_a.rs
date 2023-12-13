use num_traits::AsPrimitive;

#[derive(Debug)]
pub struct Particle<Real> {
    /// Position
    pub x: nalgebra::Vector2::<Real>,
    /// Velocity
    pub v: nalgebra::Vector2::<Real>,
    /// Deformation gradient
    pub defgrad: nalgebra::Matrix2::<Real>,
    /// gradient of momentum from MLS
    pub velograd: nalgebra::Matrix2::<Real>,
    /// Determinant of the deformation gradient matrix
    pub det_defgrad_plastic: Real,
    /// Color
    pub c: u8,
}

impl<Real> Particle<Real>
where Real: nalgebra::RealField
{
    pub fn new(x_: nalgebra::Vector2::<Real>, c_: u8) -> Self {
        Self {
            x: x_,
            v: nalgebra::Vector2::<Real>::zeros(),
            defgrad: nalgebra::Matrix2::<Real>::identity(),
            velograd: nalgebra::Matrix2::<Real>::zeros(),
            det_defgrad_plastic: Real::one(),
            c: c_,
        }
    }
}

/// Seed particles with position and color
pub fn add_object<Real>(
    particles: &mut Vec<Particle<Real>>,
    num_particles: usize,
    center: nalgebra::Vector2::<Real>,
    c: u8,
    rng: &mut rand::rngs::StdRng)
    where rand::distributions::Standard: rand::prelude::Distribution<Real>,
        Real:nalgebra::RealField + 'static + Copy,
        f64: AsPrimitive<Real>
{
    use rand::Rng;
    for _i in 0..num_particles {
        let x: Real = (rng.gen::<Real>() * 2f64.as_() - Real::one()) * 0.08f64.as_() + center.x;
        let y: Real = (rng.gen::<Real>() * 2f64.as_() - Real::one()) * 0.08f64.as_() + center.y;
        let p = Particle::new(nalgebra::Vector2::<Real>::new(x, y), c);
        particles.push(p);
    }
}