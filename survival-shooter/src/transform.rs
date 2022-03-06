use std::cell::Cell;

use cgmath::{Matrix3, Matrix4, One, Vector3};

use super::*;

#[derive(Clone, Debug)]
pub struct Transform {
    pub position: Cell<Vector3<f32>>,
    pub rotation: Cell<Vector3<f32>>, // euler angles
    pub scale: Cell<Vector3<f32>>,
    pub matrix: Cell<Matrix4<f32>>,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: Cell::new(Vector3::new(0.0, 0.0, 0.0)),
            rotation: Cell::new(Vector3::new(0.0, 0.0, 0.0)),
            scale: Cell::new(Vector3::new(1.0, 1.0, 1.0)),
            matrix: Cell::new(Matrix4::one()),
        }
    }

    pub fn position(&self) -> Vector3<f32> {
        self.position.get()
    }

    pub fn rotation(&self) -> Vector3<f32> {
        self.rotation.get()
    }

    pub fn scale(&self) -> Vector3<f32> {
        self.scale.get()
    }

    pub fn matrix(&self) -> Matrix4<f32> {
        self.matrix.get()
    }

    pub fn set_position(&self, new_position: Vector3<f32>) {
        self.position.set(new_position);
        let mut matrix = self.matrix.get();
        matrix.x.w = new_position.x;
        matrix.y.w = new_position.x;
        matrix.z.w = new_position.x;
        self.matrix.set(matrix);
    }

    pub fn set_rotation(&self, new_rotation: Vector3<f32>) {
        self.rotation.set(new_rotation);
        self.resync_matrix();
    }

    pub fn set_scale(&self, new_scale: Vector3<f32>) {
        self.scale.set(new_scale);
        self.resync_matrix();
    }

    pub fn get_rotation_matrix(&self) -> Matrix4<f32> {
        let rotation = self.rotation.get();
        make_rotation_matrix(rotation.x, rotation.y, rotation.z)
    }

    pub fn get_rotation_matrix3(&self) -> Matrix3<f32> {
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

    fn resync_matrix(&self) {
        let rotation = self.rotation.get();
        self.matrix.set(
            make_translation_matrix(self.position.get())
                * make_rotation_matrix(rotation.x, rotation.y, rotation.z)
                * make_scale_matrix(self.scale.get()),
        );
    }
}
