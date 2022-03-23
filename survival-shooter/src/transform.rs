use std::cell::Cell;

use cgmath::{Matrix3, Matrix4, One, Quaternion, Rad, Vector3};

use super::*;

#[derive(Clone, Debug)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>, // euler angles
    pub scale: Vector3<f32>,
    pub matrix: Matrix4<f32>,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(0.0, 0.0, 1.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            matrix: Matrix4::one(),
        }
    }

    pub fn _position(&self) -> Vector3<f32> {
        self.position
    }

    pub fn _rotation(&self) -> Quaternion<f32> {
        self.rotation
    }

    pub fn _scale(&self) -> Vector3<f32> {
        self.scale
    }

    pub fn _matrix(&self) -> Matrix4<f32> {
        self.matrix
    }

    pub fn set_position(&mut self, new_position: Vector3<f32>) {
        self.position = new_position;
        let mut matrix = self.matrix;
        matrix.w.x = new_position.x;
        matrix.w.y = new_position.y;
        matrix.w.z = new_position.z;
        self.matrix = matrix;
    }

    pub fn set_rotation(&mut self, new_rotation: Quaternion<f32>) {
        self.rotation = new_rotation;
        self.resync_matrix();
    }

    pub fn rotate_around_axis(&self, axis: Vector3<f32>, angle: Rad<f32>) {}

    pub fn set_scale(&mut self, new_scale: Vector3<f32>) {
        self.scale = new_scale;
        self.resync_matrix();
    }

    pub fn get_rotation_matrix(&self) -> Matrix4<f32> {
        let rotation = self.rotation;
        make_rotation_matrix(self.rotation)
    }

    pub fn _get_rotation_matrix3(&self) -> Matrix3<f32> {
        let rotation_matrix = self.get_rotation_matrix();
        return Matrix3::from_cols(
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
        );
    }

    fn resync_matrix(&mut self) {
        self.matrix = make_translation_matrix(self.position)
            * make_rotation_matrix(self.rotation)
            * make_scale_matrix(self.scale);
    }
}
