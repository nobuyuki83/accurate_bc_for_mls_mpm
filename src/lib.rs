pub mod canvas;

pub fn polar_decomposition(
    m: &nalgebra::Matrix2::<f32>) -> (nalgebra::Matrix2::<f32>, nalgebra::Matrix2::<f32>) {
    let x = m[(0, 0)] + m[(1, 1)];
    let y = m[(1, 0)] - m[(0, 1)];
    let scale = 1_f32 / (x * x + y * y).sqrt();
    let c = x * scale;
    let s = y * scale;
    let u_mat = nalgebra::Matrix2::<f32>::new(
        c, -s,
        s, c);
    let p_mat = u_mat.transpose() * m;
    (u_mat, p_mat)
}

pub fn myclamp(
    v: f32,
    vmin: f32,
    vmax: f32
) -> f32 {
    let v0 = if v < vmin { vmin } else { v };
    if v0 > vmax { vmax } else { v0 }
}