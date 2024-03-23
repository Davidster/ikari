use crate::game::*;

use glam::{
    f32::{Vec2, Vec3},
    Quat,
};
use ikari::math::lerp;

#[derive(Clone, Debug)]
pub struct BallComponent {
    pub transform: ikari::transform::Transform,
    direction: Vec3,
    speed: f32,
    radius: f32,
}

impl BallComponent {
    pub fn new(position: Vec2, direction: Vec2, radius: f32, speed: f32) -> Self {
        let transform = ikari::transform::TransformBuilder::new()
            .position(Vec3::new(position.x, radius, position.y))
            .scale(Vec3::new(radius, radius, radius))
            .build();
        let Vec2 {
            x: direction_x,
            y: direction_z,
        } = direction;
        Self {
            transform,
            direction: Vec3::new(direction_x, 0.0, direction_z).normalize(),
            speed,
            radius,
        }
    }

    pub fn rand() -> Self {
        BallComponent::new(
            Vec2::new(
                -10.0 + rand::random::<f32>() * 20.0,
                -10.0 + rand::random::<f32>() * 20.0,
            ),
            Vec2::new(
                -1.0 + rand::random::<f32>() * 2.0,
                -1.0 + rand::random::<f32>() * 2.0,
            ),
            0.05 + (rand::random::<f32>() * 0.2),
            1.0 + (rand::random::<f32>() * 15.0),
        )
    }

    pub fn update(&mut self, dt: f64) {
        // update position
        let curr_position = self.transform.position();
        let displacement = self.direction * self.speed * dt as f32;
        let new_position = curr_position + (displacement / 1.0);
        self.transform.set_position(new_position);

        // update rotation
        let curr_rotation = self.transform.rotation();
        let up = Vec3::new(0.0, 1.0, 0.0);
        let axis_of_rotation = self.direction.cross(up);
        let circumference = 2.0 * std::f32::consts::PI * self.radius;
        let angle_of_rotation =
            1.0 * (displacement.length() / circumference) * 2.0 * std::f32::consts::PI;
        let rotational_displacement = Quat::from_axis_angle(axis_of_rotation, -angle_of_rotation);
        let new_rotation = rotational_displacement * curr_rotation;
        self.transform.set_rotation(new_rotation);

        // do collision
        let pos_2d = Vec2::new(new_position.x, new_position.z);
        let ball_edges: Vec<_> = [
            Vec2::new(self.radius, 0.0),
            Vec2::new(0.0, self.radius),
            Vec2::new(-self.radius, 0.0),
            Vec2::new(0.0, -self.radius),
        ]
        .iter()
        .map(|pos| *pos + pos_2d)
        .collect();

        ball_edges.iter().for_each(|edge| {
            let Vec2 { x, y: z } = edge;
            if *x > ARENA_SIDE_LENGTH || *x < -ARENA_SIDE_LENGTH {
                // reflect on x
                self.direction.x = -self.direction.x;
            }
            if *z > ARENA_SIDE_LENGTH || *z < -ARENA_SIDE_LENGTH {
                // reflect on z
                self.direction.z = -self.direction.z;
            }
        });
    }

    pub fn lerp(&self, other: &Self, alpha: f32) -> Self {
        let transform = ikari::transform::TransformBuilder::new()
            .position(
                self.transform
                    .position()
                    .lerp(other.transform.position(), alpha),
            )
            .scale(self.transform.scale().lerp(other.transform.scale(), alpha))
            .rotation(
                self.transform
                    .rotation()
                    .lerp(other.transform.rotation(), alpha),
            )
            .build();

        BallComponent {
            transform,
            direction: self.direction.lerp(other.direction, alpha),
            speed: lerp(self.speed, other.speed, alpha),
            radius: lerp(self.radius, other.radius, alpha),
        }
    }
}
