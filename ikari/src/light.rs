use crate::scene::*;

use glam::f32::Vec3;

#[derive(Clone, Debug)]
pub struct PointLightComponent {
    pub node_id: GameNodeId,
    pub color: Vec3,
    pub intensity: f32,
}

#[derive(Clone, Debug)]
pub struct DirectionalLightComponent {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}
