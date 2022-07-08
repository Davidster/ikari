use cgmath::{Rad, Vector2, Vector3};

use super::*;

#[derive(Clone, Debug)]
pub struct BallComponent {
    pub transform: crate::transform::Transform,
    direction: Vector3<f32>,
    speed: f32,
    radius: f32,
}

impl BallComponent {
    pub fn new(position: Vector2<f32>, direction: Vector2<f32>, radius: f32, speed: f32) -> Self {
        let transform = TransformBuilder::new()
            .position(Vector3::new(position.x, radius, position.y))
            .scale(Vector3::new(radius, radius, radius))
            .build();
        let Vector2 {
            x: direction_x,
            y: direction_z,
        } = direction;
        Self {
            transform,
            direction: Vector3::new(direction_x, 0.0, direction_z).normalize(),
            speed,
            radius,
        }
    }

    pub fn rand() -> Self {
        BallComponent::new(
            Vector2::new(
                -10.0 + rand::random::<f32>() * 20.0,
                -10.0 + rand::random::<f32>() * 20.0,
            ),
            Vector2::new(
                -1.0 + rand::random::<f32>() * 2.0,
                -1.0 + rand::random::<f32>() * 2.0,
            ),
            0.5 + (rand::random::<f32>() * 0.75),
            1.0 + (rand::random::<f32>() * 15.0),
        )
    }

    pub fn update(&mut self, dt: f32, _logger: &mut Logger) {
        // update position
        let curr_position = self.transform.position();
        let displacement = self.direction * self.speed * dt;
        let new_position = curr_position + (displacement / 1.0);
        self.transform.set_position(new_position);

        // update rotation
        let curr_rotation = self.transform.rotation();
        let up = Vector3::new(0.0, 1.0, 0.0);
        let axis_of_rotation = self.direction.cross(up);
        let circumference = 2.0 * std::f32::consts::PI * self.radius;
        let angle_of_rotation =
            1.0 * (displacement.magnitude() / circumference) * 2.0 * std::f32::consts::PI;
        let rotational_displacement =
            make_quat_from_axis_angle(axis_of_rotation, Rad(-angle_of_rotation));
        let new_rotation = rotational_displacement * curr_rotation;
        self.transform.set_rotation(new_rotation);

        // nice log thingy
        // logger.log(&format!(
        //     "[{}:{}] {} = {:#?}",
        //     file!(),
        //     line!(),
        //     "direction",
        //     self.direction
        // ));

        // do collision
        let pos_2d = Vector2::new(new_position.x, new_position.z);
        let ball_edges: Vec<_> = vec![
            Vector2::new(self.radius, 0.0),
            Vector2::new(0.0, self.radius),
            Vector2::new(-self.radius, 0.0),
            Vector2::new(0.0, -self.radius),
        ]
        .iter()
        .map(|pos| *pos + pos_2d)
        .collect();
        // for edge in
        ball_edges.iter().for_each(|edge| {
            let Vector2 { x, y: z } = edge;
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
        let transform = TransformBuilder::new()
            .position(lerp_vec(
                self.transform.position(),
                other.transform.position(),
                alpha,
            ))
            .scale(lerp_vec(
                self.transform.scale(),
                other.transform.scale(),
                alpha,
            ))
            .rotation(
                self.transform
                    .rotation()
                    .nlerp(other.transform.rotation(), alpha),
            )
            .build();

        BallComponent {
            transform,
            direction: lerp_vec(self.direction, other.direction, alpha),
            speed: lerp(self.speed, other.speed, alpha),
            radius: lerp(self.radius, other.radius, alpha),
        }
    }
}
