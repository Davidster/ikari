use crate::physics::*;

use glam::{
    f32::{Mat3, Mat4, Quat, Vec3},
    Affine3A,
};
use std::ops::{Deref, DerefMut, Mul};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimpleTransform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform(pub Affine3A);

impl Deref for Transform {
    type Target = Affine3A;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Transform {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Transform {
    pub const IDENTITY: Self = Self(Affine3A::IDENTITY);

    pub fn position(&self) -> Vec3 {
        let (_, _, position) = self.to_scale_rotation_translation();
        position
    }

    pub fn rotation(&self) -> Quat {
        let (_, rotation, _) = self.to_scale_rotation_translation();
        rotation
    }

    pub fn scale(&self) -> Vec3 {
        let (scale, _, _) = self.to_scale_rotation_translation();
        scale
    }

    pub fn set_position(&mut self, new_position: Vec3) {
        self.translation = new_position.into();
    }

    pub fn set_rotation(&mut self, new_rotation: Quat) {
        let (scale, _, position) = self.to_scale_rotation_translation();
        self.0 = Affine3A::from_scale_rotation_translation(scale, new_rotation, position);
    }

    pub fn set_scale(&mut self, new_scale: Vec3) {
        let (_, rotation, position) = self.to_scale_rotation_translation();
        self.0 = Affine3A::from_scale_rotation_translation(new_scale, rotation, position);
    }

    pub fn _get_rotation_matrix3(&self) -> Mat3 {
        let rotation_matrix = Mat4::from_quat(self.rotation());
        Mat3::from_cols(
            Vec3::new(
                rotation_matrix.x_axis.x,
                rotation_matrix.x_axis.y,
                rotation_matrix.x_axis.z,
            ),
            Vec3::new(
                rotation_matrix.y_axis.x,
                rotation_matrix.y_axis.y,
                rotation_matrix.y_axis.z,
            ),
            Vec3::new(
                rotation_matrix.z_axis.x,
                rotation_matrix.z_axis.y,
                rotation_matrix.z_axis.z,
            ),
        )
    }

    pub fn apply_isometry(&mut self, isometry: Isometry<f32>) {
        self.set_position(Vec3::new(
            isometry.translation.x,
            isometry.translation.y,
            isometry.translation.z,
        ));
        self.set_rotation(Quat::from_xyzw(
            isometry.rotation.i,
            isometry.rotation.j,
            isometry.rotation.k,
            isometry.rotation.w,
        ));
    }

    pub fn decompose(&self) -> SimpleTransform {
        let (scale, rotation, position) = self.to_scale_rotation_translation();

        SimpleTransform {
            position,
            rotation,
            scale,
        }
    }
}

impl From<Transform> for Mat4 {
    fn from(transform: Transform) -> Self {
        transform.0.into()
    }
}

impl From<Affine3A> for Transform {
    fn from(matrix: Affine3A) -> Self {
        Self(matrix)
    }
}

impl From<[[f32; 4]; 4]> for Transform {
    fn from(matrix: [[f32; 4]; 4]) -> Self {
        let (scale, rotation, position) =
            Mat4::from_cols_array_2d(&matrix).to_scale_rotation_translation();
        Self(Affine3A::from_scale_rotation_translation(
            scale, rotation, position,
        ))
    }
}

impl From<SimpleTransform> for Transform {
    fn from(simple_transform: SimpleTransform) -> Self {
        let mut transform = Transform::IDENTITY;
        transform.set_position(simple_transform.position);
        transform.set_rotation(simple_transform.rotation);
        transform.set_scale(simple_transform.scale);
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
                .rotation(Quat::from_array(rotation))
                .build(),
            gltf::scene::Transform::Matrix { matrix } => {
                let mat4 = Mat4::from_cols_array_2d(&matrix);
                if mat4.eq(&Mat4::IDENTITY) {
                    return Transform::IDENTITY;
                }
                let (scale, rotation, position) = mat4.to_scale_rotation_translation();
                Affine3A::from_scale_rotation_translation(scale, rotation, position).into()
            }
        }
    }
}

impl From<Isometry<f32>> for Transform {
    fn from(isometry: Isometry<f32>) -> Self {
        let mut transform = Transform::IDENTITY;
        transform.set_position(Vec3::new(
            isometry.translation.x,
            isometry.translation.y,
            isometry.translation.z,
        ));
        transform.set_rotation(Quat::from_xyzw(
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
        (self.0 * rhs.0).into()
    }
}

#[derive(Clone, Debug)]
pub struct TransformBuilder {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}

impl TransformBuilder {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn position(mut self, position: Vec3) -> Self {
        self.position = position;
        self
    }

    pub fn rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    pub fn build(self) -> Transform {
        let mut result = Transform::IDENTITY;
        result.set_position(self.position);
        result.set_rotation(self.rotation);
        result.set_scale(self.scale);
        result
    }
}

pub fn make_quat_from_axis_angle(axis: Vec3, angle: f32) -> Quat {
    Quat::from_axis_angle(axis, angle)
}

/// from https://vincent-p.github.io/posts/vulkan_perspective_matrix/ and https://thxforthefish.com/posts/reverse_z/
pub fn make_perspective_proj_matrix(
    near_plane_distance: f32,
    far_plane_distance: f32,
    vertical_fov: f32,
    aspect_ratio: f32,
    reverse_z: bool,
) -> Mat4 {
    let n = near_plane_distance;
    let f = far_plane_distance;
    let cot = 1.0 / (vertical_fov / 2.0).tan();
    let ar = aspect_ratio;
    #[rustfmt::skip]
    let persp_matrix = Mat4::from_cols_array(&[
        cot/ar, 0.0, 0.0,     0.0,
        0.0,    cot, 0.0,     0.0,
        0.0,    0.0, f/(n-f), n*f/(n-f),
        0.0,    0.0, -1.0,     0.0,
    ]).transpose();
    if !reverse_z {
        persp_matrix
    } else {
        #[rustfmt::skip]
        let reverse_z = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0,  0.0,
            0.0, 1.0, 0.0,  0.0,
            0.0, 0.0, -1.0, 1.0,
            0.0, 0.0, 0.0,  1.0,
        ]).transpose();
        reverse_z * persp_matrix
    }
}

pub fn make_orthographic_proj_matrix(
    width: f32,
    height: f32,
    near_plane: f32,
    far_plane: f32,
    reverse_z: bool,
) -> Mat4 {
    let l = -width / 2.0;
    let r = width / 2.0;
    let t = height / 2.0;
    let b = -height / 2.0;
    let n = near_plane;
    let f = far_plane;
    #[rustfmt::skip]
    let orth_matrix =  Mat4::from_cols_array(&[
        2.0/(r-l), 0.0,       0.0,       -(r+l)/(r-l),
        0.0,       2.0/(t-b), 0.0,       -(t+b)/(t-b),
        0.0,       0.0,       1.0/(n-f), n/(n-f),
        0.0,       0.0,       0.0,       1.0,
    ]).transpose();
    if !reverse_z {
        orth_matrix
    } else {
        #[rustfmt::skip]
        let reverse_z =  Mat4::from_cols_array(&[
            1.0, 0.0, 0.0,  0.0,
            0.0, 1.0, 0.0,  0.0,
            0.0, 0.0, -1.0, 1.0,
            0.0, 0.0, 0.0,  1.0,
        ]).transpose();
        reverse_z * orth_matrix
    }
}

pub fn direction_vector_to_coordinate_frame_matrix(dir: Vec3) -> Mat4 {
    look_at_dir(Vec3::new(0.0, 0.0, 0.0), dir)
}

pub fn _look_at(eye_pos: Vec3, dst_pos: Vec3) -> Mat4 {
    look_at_dir(eye_pos, (dst_pos - eye_pos).normalize())
}

/// this gives a coordinate frame for an object that points in direction dir.
/// you can use it to point a model in direction dir from position eye_pos.
/// to make a camera view matrix (one that transforms world space coordinates
/// into the camera's view space), take the inverse of this matrix
/// warning: fails if pointing directly upward or downward
/// a.k.a. if dir.normalize() is approximately (0, 1, 0) or (0, -1, 0)
pub fn look_at_dir(eye_pos: Vec3, dir: Vec3) -> Mat4 {
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let forward = dir.normalize();
    let left = world_up.cross(forward).normalize();
    let camera_up = forward.cross(left).normalize();
    #[rustfmt::skip]
    let look_at_matrix = Mat4::from_cols_array(&[
        left.x, camera_up.x, forward.x, eye_pos.x,
        left.y, camera_up.y, forward.y, eye_pos.y,
        left.z, camera_up.z, forward.z, eye_pos.z,
        0.0,    0.0, 0.0, 1.0,
    ]).transpose();
    look_at_matrix
}
pub fn clear_translation_from_matrix(mut transform: Mat4) -> Mat4 {
    transform.w_axis.x = 0.0;
    transform.w_axis.y = 0.0;
    transform.w_axis.z = 0.0;
    transform
}

pub fn get_translation_from_matrix(transform: Mat4) -> Vec3 {
    let (_, _, position) = transform.to_scale_rotation_translation();
    position
}

pub fn _matrix_diff(a: Mat4, b: Mat4) -> f32 {
    let diff: [[f32; 4]; 4] = (b - a).to_cols_array_2d();
    let mut total = 0.0;
    for column in diff {
        for val in column {
            total += val;
        }
    }
    total
}
