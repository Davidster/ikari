use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, One, Quaternion, Rad, Vector3};
use std::ops::Mul;

use super::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimpleTransform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    position: Vector3<f32>,
    rotation: Quaternion<f32>,
    scale: Vector3<f32>,
    matrix: Matrix4<f32>,
    base_matrix: Matrix4<f32>,
    is_new: bool,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(0.0, 0.0, 1.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            matrix: Matrix4::one(),
            base_matrix: Matrix4::one(),
            is_new: true,
        }
    }

    pub fn position(&self) -> Vector3<f32> {
        self.position
    }

    pub fn rotation(&self) -> Quaternion<f32> {
        self.rotation
    }

    pub fn scale(&self) -> Vector3<f32> {
        self.scale
    }

    pub fn matrix(&self) -> Matrix4<f32> {
        if self.is_new {
            Matrix4::identity()
        } else {
            self.matrix * self.base_matrix
        }
    }

    pub fn _is_new(&self) -> bool {
        self.is_new
    }

    pub fn set_position(&mut self, new_position: Vector3<f32>) {
        self.position = new_position;
        self.matrix.w.x = new_position.x;
        self.matrix.w.y = new_position.y;
        self.matrix.w.z = new_position.z;
        self.is_new = false;
    }

    pub fn set_rotation(&mut self, new_rotation: Quaternion<f32>) {
        self.rotation = new_rotation;
        self.is_new = false;
        self.resync_matrix();
    }

    pub fn set_scale(&mut self, new_scale: Vector3<f32>) {
        self.scale = new_scale;
        self.is_new = false;
        self.resync_matrix();
    }

    pub fn _get_rotation_matrix(&self) -> Matrix4<f32> {
        make_rotation_matrix(self.rotation) * self.base_matrix
    }

    pub fn _get_rotation_matrix3(&self) -> Matrix3<f32> {
        let rotation_matrix = self._get_rotation_matrix();
        Matrix3::from_cols(
            Vector3::new(
                rotation_matrix.x.x,
                rotation_matrix.x.y,
                rotation_matrix.x.z,
            ),
            Vector3::new(
                rotation_matrix.y.x,
                rotation_matrix.y.y,
                rotation_matrix.y.z,
            ),
            Vector3::new(
                rotation_matrix.z.x,
                rotation_matrix.z.y,
                rotation_matrix.z.z,
            ),
        )
    }

    pub fn apply_isometry(&mut self, isometry: Isometry<f32>) {
        self.set_position(Vector3::new(
            isometry.translation.x,
            isometry.translation.y,
            isometry.translation.z,
        ));
        self.set_rotation(Quaternion::new(
            isometry.rotation.w,
            isometry.rotation.i,
            isometry.rotation.j,
            isometry.rotation.k,
        ));
    }

    pub fn decompose(&self) -> SimpleTransform {
        let mat = self.matrix();
        let position = Vector3::new(mat.w.x, mat.w.y, mat.w.z);
        let scale = Vector3::new(
            Vector3::new(mat.x.x, mat.x.y, mat.x.z).magnitude(),
            Vector3::new(mat.y.x, mat.y.y, mat.y.z).magnitude(),
            Vector3::new(mat.z.x, mat.z.y, mat.z.z).magnitude(),
        );
        let rotation = get_quat_from_rotation_matrix(Matrix4::new(
            mat.x.x / scale.x,
            mat.x.y / scale.x,
            mat.x.z / scale.x,
            0.0,
            mat.y.x / scale.y,
            mat.y.y / scale.y,
            mat.y.z / scale.y,
            0.0,
            mat.z.x / scale.z,
            mat.z.y / scale.z,
            mat.z.z / scale.z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ));

        SimpleTransform {
            position,
            rotation,
            scale,
        }
    }

    fn resync_matrix(&mut self) {
        self.matrix = make_translation_matrix(self.position)
            * make_rotation_matrix(self.rotation)
            * make_scale_matrix(self.scale);
    }
}

impl From<Matrix4<f32>> for Transform {
    fn from(matrix: Matrix4<f32>) -> Self {
        let mut transform = Transform::new();
        transform.base_matrix = matrix;
        transform.is_new = false;
        transform
    }
}

impl From<SimpleTransform> for Transform {
    fn from(simple_transform: SimpleTransform) -> Self {
        let mut transform = Transform::new();
        transform.set_position(simple_transform.position);
        transform.set_rotation(simple_transform.rotation);
        transform.set_scale(simple_transform.scale);
        transform.resync_matrix();
        transform
    }
}

impl From<gltf::scene::Transform> for Transform {
    fn from(gltf_transform: gltf::scene::Transform) -> Self {
        match gltf_transform {
            gltf::scene::Transform::Decomposed {
                translation,
                rotation,
                scale,
            } => TransformBuilder::new()
                .position(translation.into())
                .scale(scale.into())
                .rotation(rotation.into())
                .build(),
            gltf::scene::Transform::Matrix { matrix } => Matrix4::from(matrix).into(),
        }
    }
}

impl From<Isometry<f32>> for Transform {
    fn from(isometry: Isometry<f32>) -> Self {
        let mut transform = Transform::new();
        transform.set_position(Vector3::new(
            isometry.translation.x,
            isometry.translation.y,
            isometry.translation.z,
        ));
        transform.set_rotation(Quaternion::new(
            isometry.rotation.i,
            isometry.rotation.j,
            isometry.rotation.k,
            isometry.rotation.w,
        ));
        transform
    }
}

impl Mul for Transform {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if rhs.is_new {
            self
        } else {
            (self.matrix() * rhs.matrix()).into()
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransformBuilder {
    position: Vector3<f32>,
    rotation: Quaternion<f32>,
    scale: Vector3<f32>,
    base_matrix: Matrix4<f32>,
}

impl TransformBuilder {
    pub fn new() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(0.0, 0.0, 1.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            base_matrix: Matrix4::one(),
        }
    }

    pub fn position(mut self, position: Vector3<f32>) -> Self {
        self.position = position;
        self
    }

    pub fn rotation(mut self, rotation: Quaternion<f32>) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn scale(mut self, scale: Vector3<f32>) -> Self {
        self.scale = scale;
        self
    }

    pub fn _base_matrix(mut self, base_matrix: Matrix4<f32>) -> Self {
        self.base_matrix = base_matrix;
        self
    }

    pub fn build(self) -> Transform {
        let mut result = Transform::from(self.base_matrix);
        result.set_position(self.position);
        result.set_rotation(self.rotation);
        result.set_scale(self.scale);
        result.is_new = false;
        result
    }
}

#[cfg(test)]
mod tests {
    use cgmath::{Rad, Vector4};

    use super::*;

    #[test]
    fn decompose_transform() {
        let position = -Vector3::new(1.0, 2.0, 3.0);
        let rotation = make_quat_from_axis_angle(Vector3::new(1.0, 0.2, 3.0).normalize(), Rad(0.2));
        let scale = Vector3::new(3.0, 2.0, 1.0);

        let expected = SimpleTransform {
            position,
            rotation,
            scale,
        };

        let transform_1 = TransformBuilder::new()
            .position(position)
            .rotation(rotation)
            .scale(scale)
            .build();

        let transform_2: crate::transform::Transform = (make_translation_matrix(position)
            * make_rotation_matrix(rotation)
            * make_scale_matrix(scale))
        .into();

        // these fail but they pass in our hearts 💗
        assert_eq!(transform_1.decompose(), expected);
        assert_eq!(transform_2.decompose(), expected);
        assert_eq!(
            transform_1.matrix() * Vector4::new(2.0, -3.0, 4.0, 1.0),
            crate::transform::Transform::from(transform_1.decompose()).matrix()
                * Vector4::new(2.0, -3.0, 4.0, 1.0)
        );
    }
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

// this gives a coordinate frame for an object that points in direction dir.
// you can use it to point a model in direction dir from position eye_pos.
// to make a camera view matrix (one that transforms world space coordinates
// into the camera's view space), take the inverse of this matrix
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
