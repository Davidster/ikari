use crate::physics::rapier3d_f64::prelude::*;

use glam::{
    f32::{Mat3, Mat4, Quat, Vec3},
    Affine3A,
};
use rapier3d_f64::na::{Quaternion, UnitQuaternion};
use std::ops::{Deref, DerefMut, Mul};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimpleTransform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Default)]
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
        self.translation.into()
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

    pub fn apply_isometry(&mut self, isometry: Isometry<f64>) {
        self.set_position(Vec3::new(
            isometry.translation.x as f32,
            isometry.translation.y as f32,
            isometry.translation.z as f32,
        ));
        self.set_rotation(Quat::from_xyzw(
            isometry.rotation.i as f32,
            isometry.rotation.j as f32,
            isometry.rotation.k as f32,
            isometry.rotation.w as f32,
        ));
    }

    pub fn as_isometry(&self) -> Isometry<f64> {
        let (_scale, rotation, position) = self.to_scale_rotation_translation();
        Isometry::from_parts(
            Translation::from(position.as_dvec3().to_array()),
            UnitQuaternion::from_quaternion(Quaternion::from(rotation.as_f64().to_array())),
        )
    }

    // TODO: remove this, just use to_scale_rotation_translation()
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

impl Default for TransformBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// from https://vincent-p.github.io/posts/vulkan_perspective_matrix/ and https://thxforthefish.com/posts/reverse_z/
pub fn make_perspective_proj_matrix(
    near_plane_distance: f32,
    far_plane_distance: f32,
    fov_x: f32,
    aspect_ratio: f32,
    reverse_z: bool,
) -> Mat4 {
    let n = near_plane_distance;
    let f = far_plane_distance;
    let fov_y = fov_x / aspect_ratio;
    let cot = 1.0 / (fov_y / 2.0).tan();
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
        0.0,       0.0,       1.0/(f-n), -n/(f-n),
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

pub fn _look_at(eye_pos: Vec3, dst_pos: Vec3) -> Transform {
    look_in_dir(eye_pos, dst_pos - eye_pos)
}

/// this gives a coordinate frame for an object that points in direction dir.
/// you can use it to point a model in direction dir from position eye_pos.
/// to make a camera view matrix (one that transforms world space coordinates
/// into the camera's view space), take the inverse of this matrix
/// warning: fails if pointing directly upward or downward
/// a.k.a. if dir.normalize() is approximately (0, 1, 0) or (0, -1, 0)
pub fn look_in_dir(eye_pos: Vec3, dir: Vec3) -> Transform {
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let forward = dir.normalize();
    let left = world_up.cross(forward).normalize();
    let camera_up = forward.cross(left).normalize();
    #[rustfmt::skip]
    let mat3 = Mat3::from_cols_array(&[
        left.x, camera_up.x, forward.x,
        left.y, camera_up.y, forward.y, 
        left.z, camera_up.z, forward.z,
    ]).transpose();
    Transform(Affine3A::from_mat3_translation(mat3, eye_pos))
}

pub fn clear_translation_from_matrix(mut transform: Mat4) -> Mat4 {
    transform.w_axis.x = 0.0;
    transform.w_axis.y = 0.0;
    transform.w_axis.z = 0.0;
    transform
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
