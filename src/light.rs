use cgmath::Vector3;

#[derive(Clone, Debug)]
pub struct PointLightComponent {
    pub transform: crate::transform::Transform, // TODO: remove, use a node_index instead
    pub color: Vector3<f32>,
    pub intensity: f32,
}

#[derive(Clone, Debug)]
pub struct DirectionalLightComponent {
    pub position: Vector3<f32>,
    pub direction: Vector3<f32>, // TODO: use a regular transform via node_index instead of direction vector
    pub color: Vector3<f32>,
    pub intensity: f32,
}
