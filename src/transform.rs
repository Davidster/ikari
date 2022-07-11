use std::ops::Mul;

use cgmath::{Matrix3, Matrix4, One, Quaternion, Vector3};

use super::*;

#[derive(Copy, Clone, Debug)]
pub struct Transform {
    position: Vector3<f32>,
    rotation: Quaternion<f32>,
    scale: Vector3<f32>,
    matrix: Matrix4<f32>,
    base_matrix: Matrix4<f32>,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(0.0, 0.0, 1.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            matrix: Matrix4::one(),
            base_matrix: Matrix4::one(),
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
        self.matrix * self.base_matrix
    }

    pub fn set_position(&mut self, new_position: Vector3<f32>) {
        self.position = new_position;
        self.matrix.w.x = new_position.x;
        self.matrix.w.y = new_position.y;
        self.matrix.w.z = new_position.z;
    }

    pub fn set_rotation(&mut self, new_rotation: Quaternion<f32>) {
        self.rotation = new_rotation;
        self.resync_matrix();
    }

    pub fn set_scale(&mut self, new_scale: Vector3<f32>) {
        self.scale = new_scale;
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
        (self.matrix() * rhs.matrix()).into()
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
        result
    }
}
