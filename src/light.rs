use cgmath::Vector3;

#[derive(Clone, Debug)]
pub struct PointLightComponent {
    pub transform: super::transform::Transform,
    pub color: Vector3<f32>,
    pub intensity: f32,
}
