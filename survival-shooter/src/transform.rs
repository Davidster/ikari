use std::cell::Cell;

use cgmath::{Matrix3, Matrix4, One, Quaternion, Rad, Vector3};

use super::*;

#[derive(Clone, Debug)]
pub struct Transform {
    pub position: Cell<Vector3<f32>>,
    pub rotation: Cell<Quaternion<f32>>, // euler angles
    pub scale: Cell<Vector3<f32>>,
    pub matrix: Cell<Matrix4<f32>>,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: Cell::new(Vector3::new(0.0, 0.0, 0.0)),
            rotation: Cell::new(Quaternion::new(0.0, 0.0, 1.0, 0.0)),
            scale: Cell::new(Vector3::new(1.0, 1.0, 1.0)),
            matrix: Cell::new(Matrix4::one()),
        }
    }

    pub fn _position(&self) -> Vector3<f32> {
        self.position.get()
    }

    pub fn _rotation(&self) -> Quaternion<f32> {
        self.rotation.get()
    }

    pub fn _scale(&self) -> Vector3<f32> {
        self.scale.get()
    }

    pub fn _matrix(&self) -> Matrix4<f32> {
        self.matrix.get()
    }

    pub fn set_position(&self, new_position: Vector3<f32>) {
        self.position.set(new_position);
        let mut matrix = self.matrix.get();
        matrix.w.x = new_position.x;
        matrix.w.y = new_position.y;
        matrix.w.z = new_position.z;
        self.matrix.set(matrix);
    }

    pub fn set_rotation(&self, new_rotation: Quaternion<f32>) {
        self.rotation.set(new_rotation);
        self.resync_matrix();
    }

    pub fn rotate_around_axis(&self, axis: Vector3<f32>, angle: Rad<f32>) {}

    pub fn set_scale(&self, new_scale: Vector3<f32>) {
        self.scale.set(new_scale);
        self.resync_matrix();
    }

    pub fn get_rotation_matrix(&self) -> Matrix4<f32> {
        let rotation = self.rotation.get();
        make_rotation_matrix(self.rotation.get())
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

    fn resync_matrix(&self) {
        self.matrix.set(
            make_translation_matrix(self.position.get())
                * make_rotation_matrix(self.rotation.get())
                * make_scale_matrix(self.scale.get()),
        );
    }
}
