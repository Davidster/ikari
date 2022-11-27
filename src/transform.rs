use std::ops::Mul;

use cgmath::{Matrix3, Matrix4, One, Quaternion, Vector3};

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

    pub fn is_new(&self) -> bool {
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
