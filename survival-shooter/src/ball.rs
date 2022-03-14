use cgmath::Vector3;

use super::*;

pub struct BallComponent {
    mesh: MeshComponent,
    direction: Vector3<f32>,
    speed: f32,
}

impl BallComponent {
    pub fn new(
        sphere_mesh: MeshComponent,
        position: Vector3<f32>,
        direction: Vector3<f32>,
        speed: f32,
    ) -> Self {
        sphere_mesh.transform.set_position(position);
        BallComponent {
            mesh: sphere_mesh,
            direction,
            speed,
        }
    }

    pub fn _update(dt: f32) {
        todo!()
    }
}
