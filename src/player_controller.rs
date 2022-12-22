use std::time::Instant;

use cgmath::{Deg, Euler, Quaternion, Rad, Vector3};
use winit::window::CursorGrabMode;
use winit::{
    dpi::PhysicalPosition,
    event::{
        DeviceEvent, ElementState, KeyboardInput, MouseScrollDelta, VirtualKeyCode, WindowEvent,
    },
    window::Window,
};

use super::*;

#[derive(Clone, Debug)]
pub struct PlayerController {
    unprocessed_delta: Option<(f64, f64)>,
    window_focused: bool,

    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,

    pub view_direction: ControlledViewDirection,
    pub speed: f32,
    pub rigid_body_handle: RigidBodyHandle,

    pub last_jump_time: Option<Instant>,
}

#[derive(Copy, Clone, Debug)]
pub struct ControlledViewDirection {
    pub horizontal: Rad<f32>,
    pub vertical: Rad<f32>,
}

impl ControlledViewDirection {
    pub fn to_quat(self) -> Quaternion<f32> {
        Quaternion::from(Euler::new(Rad(0.0), self.horizontal, Rad(0.0)))
            * Quaternion::from(Euler::new(self.vertical, Rad(0.0), Rad(0.0)))
    }

    pub fn to_direction_vector(self) -> Vector3<f32> {
        let horizontal_scale = self.vertical.0.cos();
        Vector3::new(
            (self.horizontal.0 + std::f32::consts::PI).sin() * horizontal_scale,
            self.vertical.0.sin(),
            (self.horizontal.0 + std::f32::consts::PI).cos() * horizontal_scale,
        )
        .normalize()
    }
}

impl PlayerController {
    pub fn new(
        physics_state: &mut PhysicsState,
        speed: f32,
        position: Vector3<f32>,
        view_direction: ControlledViewDirection,
    ) -> Self {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .lock_rotations()
            .build();
        let collider = ColliderBuilder::capsule_y(0.5, 0.25)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .friction_combine_rule(CoefficientCombineRule::Min)
            .collision_groups(
                InteractionGroups::all().with_memberships(COLLISION_GROUP_PLAYER_UNSHOOTABLE),
            )
            .friction(0.0)
            .restitution(0.0)
            .build();
        let rigid_body_handle = physics_state.rigid_body_set.insert(rigid_body);
        physics_state.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics_state.rigid_body_set,
        );

        Self {
            unprocessed_delta: None,
            window_focused: false,

            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,

            view_direction,
            speed,
            rigid_body_handle,
            last_jump_time: None,
        }
    }

    pub fn process_device_events(&mut self, event: &DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta: (d_x, d_y) } if self.window_focused => {
                self.unprocessed_delta = match self.unprocessed_delta {
                    Some((x, y)) => Some((x + d_x, y + d_y)),
                    None => Some((*d_x, *d_y)),
                };
            }
            DeviceEvent::MouseWheel { delta } if self.window_focused => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => *y as f32,
                };
                let scroll_direction = if scroll_amount > 0.0 { 1.0 } else { -1.0 };
                let scroll_speed = 1.0;
                self.speed = (self.speed - (scroll_direction * scroll_speed)).clamp(0.5, 300.0);
                logger_log(&format!("Speed: {:?}", self.speed));
            }
            _ => {}
        };
    }

    pub fn process_window_events(&mut self, event: &WindowEvent, window: &mut Window) {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W => {
                        self.is_forward_pressed = is_pressed;
                    }
                    VirtualKeyCode::A => {
                        self.is_left_pressed = is_pressed;
                    }
                    VirtualKeyCode::S => {
                        self.is_backward_pressed = is_pressed;
                    }
                    VirtualKeyCode::D => {
                        self.is_right_pressed = is_pressed;
                    }
                    VirtualKeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                    }
                    VirtualKeyCode::LControl => {
                        self.is_down_pressed = is_pressed;
                    }
                    _ => {}
                }
            }
            WindowEvent::Focused(focused) => {
                if *focused {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                logger_log(&format!("Window focused: {:?}", focused));
                if let Err(err) = window.set_cursor_grab(if *focused {
                    CursorGrabMode::Confined
                } else {
                    CursorGrabMode::None
                }) {
                    logger_log(&format!(
                        "Couldn't {:?} cursor: {:?}",
                        if *focused { "grab" } else { "release" },
                        err
                    ))
                }
                window.set_cursor_visible(!*focused);
                self.window_focused = *focused;
            }
            _ => {}
        };
    }

    pub fn update(&mut self, physics_state: &mut PhysicsState) {
        if let Some((d_x, d_y)) = self.unprocessed_delta {
            let mouse_sensitivity = 0.002;

            self.view_direction.horizontal += Rad(-d_x as f32 * mouse_sensitivity);
            self.view_direction.vertical = Rad((self.view_direction.vertical.0
                + Rad(-d_y as f32 * mouse_sensitivity).0).clamp(Rad::from(Deg(-90.0)).0, Rad::from(Deg(90.0)).0));
        }
        self.unprocessed_delta = None;

        let forward_direction = self.view_direction.to_direction_vector();
        let up_direction = Vector3::new(0.0, 1.0, 0.0);
        let right_direction = forward_direction.cross(up_direction);

        let new_linear_velocity = {
            let mut res: Option<Vector3<f32>> = None;

            let mut add_movement = |movement: Vector3<f32>| {
                res = match res {
                    Some(res) => Some(res + movement),
                    None => Some(movement),
                }
            };

            if self.is_forward_pressed {
                add_movement(forward_direction);
            } else if self.is_backward_pressed {
                add_movement(-forward_direction);
            }

            if self.is_right_pressed {
                add_movement(right_direction);
            } else if self.is_left_pressed {
                add_movement(-right_direction);
            }

            res.map(|res| res.normalize() * self.speed)
                .unwrap_or(Vector3::new(0.0, 0.0, 0.0))
        };

        let rigid_body = physics_state
            .rigid_body_set
            .get_mut(self.rigid_body_handle)
            .unwrap();
        let current_linear_velocity = rigid_body.linvel();
        rigid_body.set_linvel(
            vector![
                new_linear_velocity.x,
                current_linear_velocity.y, // preserve effect of gravity
                new_linear_velocity.z
            ],
            true,
        );

        let can_jump = || {
            let jump_cooldown_seconds = 1.25;
            match self.last_jump_time {
                Some(last_jump_time) => {
                    Instant::now().duration_since(last_jump_time).as_secs_f32()
                        > jump_cooldown_seconds
                }
                None => true,
            }
        };
        if self.is_up_pressed && can_jump() {
            rigid_body.apply_impulse(vector![0.0, 3.0, 0.0], true);
            self.last_jump_time = Some(Instant::now());
        }
    }

    pub fn transform(&self, physics_state: &PhysicsState) -> crate::transform::Transform {
        TransformBuilder::new()
            .position(self.position(physics_state))
            .rotation(self.view_direction.to_quat())
            .build()
    }

    pub fn position(&self, physics_state: &PhysicsState) -> Vector3<f32> {
        let position = physics_state
            .rigid_body_set
            .get(self.rigid_body_handle)
            .unwrap()
            .translation();
        Vector3::new(position.x, position.y, position.z)
    }
}
