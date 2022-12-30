use cgmath::Vector3;

pub fn _to_srgb(val: f32) -> f32 {
    val.powf(2.2)
}

pub fn _from_srgb(val: f32) -> f32 {
    val.powf(1.0 / 2.2)
}

pub fn lerp(from: f32, to: f32, alpha: f32) -> f32 {
    (alpha * to) + ((1.0 - alpha) * from)
}

pub fn lerp_vec(a: Vector3<f32>, b: Vector3<f32>, alpha: f32) -> Vector3<f32> {
    b * alpha + a * (1.0 - alpha)
}
