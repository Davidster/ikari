use cgmath::{Rad, Vector2, Vector3};

use super::*;

#[derive(Clone, Debug)]
pub struct PointLightComponent {
    pub transform: super::transform::Transform,
    pub color: Vector3<f32>,
}
