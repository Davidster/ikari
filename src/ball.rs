use cgmath::{Rad, Vector2, Vector3};

use super::*;

#[derive(Clone, Debug)]
pub struct BallComponent {
    pub instance: MeshInstance,
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
            instance: MeshInstance {
                transform,
                base_material: Default::default(),
            },
            direction: Vector3::new(direction_x, 0.0, direction_z).normalize(),
            speed,
            radius,
        }
    }

    pub fn update(&mut self, dt: f32, _logger: &mut Logger) {
        // update position
        let curr_position = self.instance.transform.position.get();
        let displacement = self.direction * self.speed * dt;
        let new_position = curr_position + (displacement / 1.0);
        self.instance.transform.set_position(new_position);

        // update rotation
        let curr_rotation = self.instance.transform.rotation.get();
        let up = Vector3::new(0.0, 1.0, 0.0);
        let axis_of_rotation = self.direction.cross(up);
        let circumference = 2.0 * std::f32::consts::PI * self.radius;
        let angle_of_rotation =
            1.0 * (displacement.magnitude() / circumference) * 2.0 * std::f32::consts::PI;
        let rotational_displacement =
            make_quat_from_axis_angle(axis_of_rotation, Rad(-angle_of_rotation));
        let new_rotation = rotational_displacement * curr_rotation;
        self.instance.transform.set_rotation(new_rotation);

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
        let transform = super::transform::Transform::new();
        transform.set_position(lerp_vec(
            self.instance.transform.position.get(),
            other.instance.transform.position.get(),
            alpha,
        ));
        transform.set_scale(lerp_vec(
            self.instance.transform.scale.get(),
            other.instance.transform.scale.get(),
            alpha,
        ));
        transform.set_rotation(
            self.instance
                .transform
                .rotation
                .get()
                .nlerp(other.instance.transform.rotation.get(), alpha),
        );

        BallComponent {
            instance: MeshInstance {
                transform,
                base_material: self.instance.base_material,
            },
            direction: lerp_vec(self.direction, other.direction, alpha),
            speed: lerp_f32(self.speed, other.speed, alpha),
            radius: lerp_f32(self.radius, other.radius, alpha),
        }
    }
}
