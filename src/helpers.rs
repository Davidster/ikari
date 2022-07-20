use cgmath::{InnerSpace, Matrix, Matrix4, Quaternion, Rad, Vector3};

pub fn to_srgb(val: f32) -> f32 {
    val.powf(2.2)
}

pub fn from_srgb(val: f32) -> f32 {
    val.powf(1.0 / 2.2)
}

pub fn lerp(from: f32, to: f32, alpha: f32) -> f32 {
    (alpha * to) + ((1.0 - alpha) * from)
}

pub fn _lerp_f64(from: f64, to: f64, alpha: f64) -> f64 {
    (alpha * to) + ((1.0 - alpha) * from)
}

pub fn lerp_vec(a: Vector3<f32>, b: Vector3<f32>, alpha: f32) -> Vector3<f32> {
    b * alpha + a * (1.0 - alpha)
}

// from https://stackoverflow.com/questions/4436764/rotating-a-quaternion-on-1-axis
pub fn make_quat_from_axis_angle(axis: Vector3<f32>, angle: Rad<f32>) -> Quaternion<f32> {
    let factor = (angle.0 / 2.0).sin();

    let x = axis.x * factor;
    let y = axis.y * factor;
    let z = axis.z * factor;

    let w = (angle.0 / 2.0).cos();

    Quaternion::new(w, x, y, z)
}

// from https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
pub fn make_rotation_matrix(r: Quaternion<f32>) -> Matrix4<f32> {
    let qr = r.s;
    let qi = r.v.x;
    let qj = r.v.y;
    let qk = r.v.z;
    let qr_2 = qr * qr;
    let qi_2 = qi * qi;
    let qj_2 = qj * qj;
    let qk_2 = qk * qk;
    let s = (qr_2 + qi_2 + qj_2 + qk_2).sqrt();
    #[rustfmt::skip]
    let result = Matrix4::new(
        1.0 - (2.0 * s * (qj_2 + qk_2)),
        2.0 * s * (qi*qj - qk*qr),
        2.0 * s * (qi*qk + qj*qr),
        0.0,
  
        2.0 * s * (qi*qj + qk*qr),
        1.0 - (2.0 * s * (qi_2 + qk_2)),
        2.0 * s * (qj*qk - qi*qr),
        0.0,
  
        
        2.0 * s * (qi*qk - qj*qr),
        2.0 * s * (qj*qk + qi*qr),
        1.0 - (2.0 * s * (qi_2 + qj_2)),
        0.0,
        
        0.0,
        0.0,
        0.0,
        1.0,
    ).transpose();
    result
}

// https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
pub fn get_quat_from_rotation_matrix(mat: Matrix4<f32>) -> Quaternion<f32> {
    let (t, q): (f32, Quaternion<f32>) = if mat.z.z < 0.0 {
        if mat.x.x > mat.y.y {
            let t = 1.0 + mat.x.x - mat.y.y - mat.z.z;
            (
                t,
                Quaternion::new(mat.y.z - mat.z.y, t, mat.x.y + mat.y.x, mat.x.z + mat.z.x),
            )
        } else {
            let t = 1.0 - mat.x.x + mat.y.y - mat.z.z;
            (
                t,
                Quaternion::new(mat.z.x - mat.x.z, mat.x.y + mat.y.x, t, mat.y.z + mat.z.y),
            )
        }
    } else if mat.x.x < -(mat.y.y) {
        let t = 1.0 - mat.x.x - mat.y.y + mat.z.z;
        (
            t,
            Quaternion::new(mat.x.y - mat.y.x, mat.z.x + mat.x.z, mat.y.z + mat.z.y, t),
        )
    } else {
        let t = 1.0 + mat.x.x + mat.y.y + mat.z.z;
        (
            t,
            Quaternion::new(t, mat.y.z - mat.z.y, mat.z.x - mat.x.z, mat.x.y - mat.y.x),
        )
    };
    q * (0.5 / t.sqrt())
}

// from https://en.wikipedia.org/wiki/Rotation_matrix
pub fn _make_rotation_matrix_from_eulers(
    pitch: Rad<f32>,
    yaw: Rad<f32>,
    roll: Rad<f32>,
) -> Matrix4<f32> {
    let pitch = pitch.0;
    let yaw = yaw.0;
    let roll = roll.0;
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

// from https://vincent-p.github.io/posts/vulkan_perspective_matrix/ and https://thxforthefish.com/posts/reverse_z/
pub fn make_perspective_proj_matrix(
    near_plane_distance: f32,
    far_plane_distance: f32,
    vertical_fov: cgmath::Rad<f32>,
    aspect_ratio: f32,
    reverse_z: bool,
) -> Matrix4<f32> {
    let n = near_plane_distance;
    let f = far_plane_distance;
    let cot = 1.0 / (vertical_fov.0 / 2.0).tan();
    let ar = aspect_ratio;
    #[rustfmt::skip]
    let persp_matrix = Matrix4::new(
        cot/ar, 0.0, 0.0,     0.0,
        0.0,    cot, 0.0,     0.0,
        0.0,    0.0, f/(n-f), n*f/(n-f),
        0.0,    0.0, -1.0,     0.0,
    ).transpose();
    if !reverse_z {
        persp_matrix
    } else {
        #[rustfmt::skip]
        let reverse_z = Matrix4::new(
            1.0, 0.0, 0.0,  0.0,
            0.0, 1.0, 0.0,  0.0,
            0.0, 0.0, -1.0, 1.0,
            0.0, 0.0, 0.0,  1.0,
        ).transpose();
        reverse_z * persp_matrix
    }
}

pub fn make_orthographic_proj_matrix(
    width: f32,
    height: f32,
    near_plane: f32,
    far_plane: f32,
    reverse_z: bool,
) -> Matrix4<f32> {
    let l = -width / 2.0;
    let r = width / 2.0;
    let t = height / 2.0;
    let b = -height / 2.0;
    let n = near_plane;
    let f = far_plane;
    #[rustfmt::skip]
    let orth_matrix = Matrix4::new(
        2.0/(r-l), 0.0,       0.0,       -(r+l)/(r-l),
        0.0,       2.0/(t-b), 0.0,       -(t+b)/(t-b),
        0.0,       0.0,       1.0/(n-f), n/(n-f),
        0.0,       0.0,       0.0,       1.0,
    ).transpose();
    if !reverse_z {
        orth_matrix
    } else {
        #[rustfmt::skip]
        let reverse_z = Matrix4::new(
            1.0, 0.0, 0.0,  0.0,
            0.0, 1.0, 0.0,  0.0,
            0.0, 0.0, -1.0, 1.0,
            0.0, 0.0, 0.0,  1.0,
        ).transpose();
        reverse_z * orth_matrix
    }
}

pub fn direction_vector_to_coordinate_frame_matrix(dir: Vector3<f32>) -> Matrix4<f32> {
    look_at_dir(Vector3::new(0.0, 0.0, 0.0), dir)
}

pub fn _look_at(eye_pos: Vector3<f32>, dst_pos: Vector3<f32>) -> Matrix4<f32> {
    look_at_dir(eye_pos, (dst_pos - eye_pos).normalize())
}

// this gives a coordinate frame for an object that points in directio dir.
// you can use it to point a model in direction dir from position eye_pos.
// to make a camera view matrix (one that transforms world space coordinates
// into the camera's view space, take the inverse of this matrix)
// warning: fails if pointing directly upward or downward
// a.k.a. if dir.normalize() is approximately (0, 1, 0) or (0, -1, 0)
pub fn look_at_dir(eye_pos: Vector3<f32>, dir: Vector3<f32>) -> Matrix4<f32> {
    let world_up = Vector3::new(0.0, 1.0, 0.0);
    let forward = dir.normalize();
    let left = world_up.cross(forward).normalize();
    let camera_up = forward.cross(left).normalize();
    #[rustfmt::skip]
    let look_at_matrix = Matrix4::new(
        left.x, camera_up.x, forward.x, eye_pos.x,
        left.y, camera_up.y, forward.y, eye_pos.y,
        left.z, camera_up.z, forward.z, eye_pos.z,
        0.0,    0.0, 0.0, 1.0,
    ).transpose();
    look_at_matrix
}
pub fn clear_translation_from_matrix(mut transform: Matrix4<f32>) -> Matrix4<f32> {
    transform.w.x = 0.0;
    transform.w.y = 0.0;
    transform.w.z = 0.0;
    transform
}

pub fn get_translation_from_matrix(transform: Matrix4<f32>) -> Vector3<f32> {
    Vector3::new(transform.w.x, transform.w.y, transform.w.z)
}

pub fn _matrix_diff(a: Matrix4<f32>, b: Matrix4<f32>) -> f32 {
    let diff: [[f32; 4]; 4] = (b - a).into();
    let mut total = 0.0;
    for column in diff {
        for val in column {
            total += val;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use cgmath::Vector4;

    use super::*;

    #[test]
    fn should_i_exist() {
        // println!(
        //     "{:?}",
        //     make_rotation_matrix(make_quat_from_axis_angle(
        //         Vector3::new(1.0, 0.0, 0.0),
        //         cgmath::Deg(-90.0).into()
        //     ))
        // );
        // println!(
        //     "{:?}",
        //     make_rotation_matrix(make_quat_from_axis_angle(
        //         Vector3::new(0.0, 1.0, 0.0),
        //         cgmath::Deg(-90.0).into()
        //     ))
        // );
        // println!(
        //     "{:?}",
        //     make_rotation_matrix(
        //         make_quat_from_axis_angle(Vector3::new(1.0, 0.0, 0.0), cgmath::Deg(-90.0).into())
        //             * make_quat_from_axis_angle(
        //                 Vector3::new(0.0, 1.0, 0.0),
        //                 cgmath::Deg(-90.0).into()
        //             )
        //     )
        // );
        // println!("{:?}", make_translation_matrix(Vector3::new(2.0, 3.0, 4.0)));
        let view = look_at_dir(-Vector3::new(0.0, 3.0, 4.0), Vector3::new(1.0, -1.0, 0.0));
        let proj = make_orthographic_proj_matrix(100.0, 100.0, 0.0, 100.0, false);
        let proj_rev = make_orthographic_proj_matrix(100.0, 100.0, 0.0, 100.0, true);

        let p1_proj_nopersp = proj * view * Vector4::new(-5.0, 0.0, 0.0, 1.0);
        let p1 = p1_proj_nopersp / p1_proj_nopersp.w;

        let p1_proj_rev_nopersp = proj_rev * view * Vector4::new(-5.0, 0.0, 0.0, 1.0);
        let p1_rev = p1_proj_rev_nopersp / p1_proj_rev_nopersp.w;

        let p2_proj_nopersp = proj * view * Vector4::new(-5.0, 0.0, 0.0, 1.0);
        let p2 = p2_proj_nopersp / p2_proj_nopersp.w;

        let p3_proj_nopersp = proj * view * Vector4::new(0.0, 0.0, -5.0, 1.0);
        let p3 = p3_proj_nopersp / p3_proj_nopersp.w;

        println!("proj: {:?}", proj.transpose());
        println!("view: {:?}", view.transpose());
        println!("proj*view: {:?}", (proj * view).transpose());
        println!("p1_proj: {:?}", p1);
        println!("p1_proj_rev: {:?}", p1_rev);
        println!("p2_proj: {:?}", p2);
        println!("p3_proj: {:?}", p3);
        assert_eq!(true, true);
    }
}
