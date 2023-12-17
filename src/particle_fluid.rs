

#[derive(Debug)]
pub struct Particle<Real> {
    /// Position
    pub x: nalgebra::Vector2::<Real>,
    /// Velocity
    pub v: nalgebra::Vector2::<Real>,
    /// gradient of momentum from MLS
    pub velograd: nalgebra::Matrix2::<Real>,
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
            velograd: nalgebra::Matrix2::<Real>::zeros(),
            c: c_,
        }
    }
}

