use cgmath::{Vector2, Vector3};

use super::*;

#[derive(Clone, Debug)]
pub struct BallComponent {
    pub transform: super::transform::Transform,
    direction: Vector3<f32>,
    speed: f32,
    radius: f32,
}

impl BallComponent {
    pub fn new(position: Vector2<f32>, direction: Vector2<f32>, radius: f32, speed: f32) -> Self {
        let transform = super::transform::Transform::new();
        transform.set_position(Vector3::new(position.x, radius, position.y));
        transform.set_scale(Vector3::new(radius, radius, radius));
        let Vector2 {
            x: direction_x,
            y: direction_z,
        } = direction;
        BallComponent {
            transform,
            direction: Vector3::new(direction_x, 0.0, direction_z).normalize(),
            speed,
            radius,
        }
    }

    pub fn update(&mut self, dt: f32, _logger: &mut Logger) {
        // update position
        let curr_position = self.transform.position.get();
        let displacement = self.direction * self.speed * dt;
        let new_position = curr_position + (displacement / 1.0);
        self.transform.set_position(new_position);

        // update rotation
        let curr_rotation = self.transform.rotation.get();
        let up = Vector3::new(0.0, 1.0, 0.0);
        let axis_of_rotation = self.direction.cross(up);
        let circumference = 2.0 * std::f32::consts::PI * self.radius;
        let angle_of_rotation =
            1.0 * (displacement.magnitude() / circumference) * 2.0 * std::f32::consts::PI;
        let rotational_displacement =
            make_quat_from_axis_angle(axis_of_rotation, -angle_of_rotation);
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
        let lerpf = |a: f32, b: f32| b * alpha + a * (1.0 - alpha);
        let lerpvec = |a: Vector3<f32>, b: Vector3<f32>| b * alpha + a * (1.0 - alpha);

        let transform = super::transform::Transform::new();
        transform.set_position(lerpvec(
            self.transform.position.get(),
            other.transform.position.get(),
        ));
        transform.set_scale(lerpvec(
            self.transform.scale.get(),
            other.transform.scale.get(),
        ));
        transform.set_rotation(
            self.transform
                .rotation
                .get()
                .nlerp(other.transform.rotation.get(), alpha),
        );

        BallComponent {
            transform,
            direction: lerpvec(self.direction, other.direction),
            speed: lerpf(self.speed, other.speed),
            radius: lerpf(self.radius, other.radius),
        }
    }
}
