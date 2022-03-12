use cgmath::Matrix;
use cgmath::Matrix4;
use cgmath::Vector3;

// TODO: use cgmath::Rad instead of f32 for rotations

pub fn _to_srgb(val: f32) -> f32 {
    val.powf(2.2)
}

pub fn lerp_f32(from: f32, to: f32, alpha: f32) -> f32 {
    (alpha * to) + ((1.0 - alpha) * from)
}

pub fn _lerp_f64(from: f64, to: f64, alpha: f64) -> f64 {
    (alpha * to) + ((1.0 - alpha) * from)
}

// from https://en.wikipedia.org/wiki/Rotation_matrix
pub fn make_rotation_matrix(pitch: f32, yaw: f32, roll: f32) -> Matrix4<f32> {
    #[rustfmt::skip]
    let result = Matrix4::new(
        yaw.cos() * pitch.cos(),
        yaw.cos() * pitch.sin() * roll.sin() - yaw.sin() * roll.cos(),
        yaw.cos() * pitch.sin() * roll.cos() + yaw.sin() * roll.sin(),
        0.0,

        yaw.sin() * pitch.cos(),
        yaw.sin() * pitch.sin() * roll.sin() + yaw.cos() * roll.cos(),
        yaw.sin() * pitch.sin() * roll.cos() - yaw.cos() * roll.sin(),
        0.0,

        -pitch.sin(),
        pitch.cos() * roll.sin(),
        pitch.cos() * roll.cos(),
        0.0,
        
        0.0,
        0.0,
        0.0,
        1.0,
    ).transpose();
    result
}

pub fn make_translation_matrix(translation: Vector3<f32>) -> Matrix4<f32> {
    #[rustfmt::skip]
    let result = Matrix4::new(
        1.0, 0.0, 0.0, translation.x,
        0.0, 1.0, 0.0, translation.y,
        0.0, 0.0, 1.0, translation.z,
        0.0, 0.0, 0.0,           1.0,
    ).transpose();
    result
}

pub fn make_scale_matrix(scale: Vector3<f32>) -> Matrix4<f32> {
    #[rustfmt::skip]
    let result = Matrix4::new(
        scale.x, 0.0,     0.0,     0.0,
        0.0,     scale.y, 0.0,     0.0,
        0.0,     0.0,     scale.z, 0.0,
        0.0,     0.0,     0.0,     1.0,
    ).transpose();
    result
}

// from http://www.songho.ca/opengl/gl_projectionmatrix.html
pub fn make_perspective_matrix(
    near_plane_distance: f32,
    far_plane_distance: f32,
    vertical_fov: cgmath::Rad<f32>,
    aspect_ratio: f32,
) -> Matrix4<f32> {
    let n = near_plane_distance;
    let f = far_plane_distance;
    let cot = 1.0 / (vertical_fov.0 / 2.0).tan();
    let ar = aspect_ratio;
    #[rustfmt::skip]
    let result = Matrix4::new(
        cot/ar, 0.0, 0.0,                           0.0,
        0.0,    cot, 0.0,                           0.0,
        0.0,    0.0, (f+n)/(n-f), (2.0*f*n)/(n-f),
        0.0,    0.0, -1.0,                          0.0,
    ).transpose();
    result
}
