use glam::f32::Vec3;

pub fn _to_srgb(val: f32) -> f32 {
    val.powf(2.2)
}

pub fn _from_srgb(val: f32) -> f32 {
    val.powf(1.0 / 2.2)
}

pub fn lerp(from: f32, to: f32, alpha: f32) -> f32 {
    (alpha * to) + ((1.0 - alpha) * from)
}

pub fn lerp_vec(a: Vec3, b: Vec3, alpha: f32) -> Vec3 {
    b * alpha + a * (1.0 - alpha)
}
