use crate::collisions::*;
use crate::game::*;
use crate::logger::*;
use crate::math::*;
use crate::physics::*;
use crate::renderer::*;
use crate::transform::*;
use crate::ui_overlay::*;

use std::time::Instant;

use glam::f32::{Quat, Vec3};
use glam::EulerRot;
use winit::event::MouseButton;
use winit::window::CursorGrabMode;
use winit::{
    dpi::PhysicalPosition,
    event::{
        DeviceEvent, ElementState, KeyboardInput, MouseScrollDelta, VirtualKeyCode, WindowEvent,
    },
    window::Window,
};

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

    pub mouse_button_pressed: bool,

    pub view_direction: ControlledViewDirection,
    pub speed: f32,
    pub rigid_body_handle: RigidBodyHandle,

    pub last_jump_time: Option<Instant>,
}

#[derive(Copy, Clone, Debug)]
pub struct ControlledViewDirection {
    pub horizontal: f32,
    pub vertical: f32,
}

impl ControlledViewDirection {
    pub fn to_quat(self) -> Quat {
        Quat::from_euler(EulerRot::XYZ, 0.0, self.horizontal, 0.0)
            * Quat::from_euler(EulerRot::XYZ, self.vertical, 0.0, 0.0)
    }

    pub fn to_vector(self) -> Vec3 {
        let horizontal_scale = self.vertical.cos();
        Vec3::new(
            (self.horizontal + std::f32::consts::PI).sin() * horizontal_scale,
            self.vertical.sin(),
            (self.horizontal + std::f32::consts::PI).cos() * horizontal_scale,
        )
        .normalize()
    }
}

impl PlayerController {
    pub fn new(
        physics_state: &mut PhysicsState,
        speed: f32,
        position: Vec3,
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

            mouse_button_pressed: false,

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

    pub fn process_device_events(&mut self, event: &DeviceEvent, ui_overlay: &mut IkariUiOverlay) {
        if ui_overlay.get_state().is_showing_options_menu {
            return;
        }
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
                // logger_log(&format!("Speed: {:?}", self.speed));
            }
            _ => {}
        };
    }

    pub fn process_window_events(
        &mut self,
        event: &WindowEvent,
        window: &mut Window,
        ui_overlay: &mut IkariUiOverlay,
    ) {
        let is_showing_options_menu = ui_overlay.get_state().is_showing_options_menu;

        let set_cursor_grab = |grab| {
            if let Err(err) = window.set_cursor_grab(if grab {
                CursorGrabMode::Confined
            } else {
                CursorGrabMode::None
            }) {
                logger_log(&format!(
                    "Couldn't {:?} cursor: {:?}",
                    if grab { "grab" } else { "release" },
                    err
                ))
            }
            window.set_cursor_visible(!grab);
        };

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
                if *state == ElementState::Pressed && *keycode == VirtualKeyCode::Escape {
                    let new_is_showing_options_menu = !is_showing_options_menu;
                    set_cursor_grab(!new_is_showing_options_menu);

                    ui_overlay.send_message(Message::TogglePopupMenu);
                }
                // TODO: join two if is_viewing_options_menu checks into one block
                if !is_showing_options_menu {
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
            }
            WindowEvent::Focused(focused) => {
                if *focused {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }

                logger_log(&format!("Window focused: {:?}", focused));
                self.window_focused = *focused;
                set_cursor_grab(!is_showing_options_menu);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                // TODO: join two if is_viewing_options_menu checks into one block
                if !is_showing_options_menu {
                    self.mouse_button_pressed = *state == ElementState::Pressed;
                }

                set_cursor_grab(!is_showing_options_menu);
            }
            _ => {}
        };
    }

    pub fn update(&mut self, physics_state: &mut PhysicsState) {
        if let Some((d_x, d_y)) = self.unprocessed_delta {
            let mouse_sensitivity = 0.002;

            self.view_direction.horizontal += -d_x as f32 * mouse_sensitivity;
            self.view_direction.vertical = (self.view_direction.vertical
                + (-d_y as f32 * mouse_sensitivity))
                .clamp(deg_to_rad(-90.0), deg_to_rad(90.0));
        }
        self.unprocessed_delta = None;

        let forward_direction = self.view_direction.to_vector();
        let up_direction = Vec3::new(0.0, 1.0, 0.0);
        let right_direction = forward_direction.cross(up_direction);

        let new_linear_velocity = {
            let mut res: Option<Vec3> = None;

            let mut add_movement = |movement: Vec3| {
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
                .unwrap_or(Vec3::new(0.0, 0.0, 0.0))
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

    pub fn position(&self, physics_state: &PhysicsState) -> Vec3 {
        let position = physics_state
            .rigid_body_set
            .get(self.rigid_body_handle)
            .unwrap()
            .translation();
        Vec3::new(position.x, position.y, position.z)
    }

    pub fn view_forward_vector(&self) -> Vec3 {
        self.view_direction.to_vector()
    }

    #[allow(dead_code)]
    pub fn view_frustum(&self, physics_state: &PhysicsState, aspect_ratio: f32) -> Frustum {
        self.view_frustum_with_position(aspect_ratio, self.position(physics_state))
    }

    pub fn view_frustum_with_position(&self, aspect_ratio: f32, camera_position: Vec3) -> Frustum {
        let camera_forward = self.view_direction.to_vector();
        let camera_right = camera_forward.cross(Vec3::new(0.0, 1.0, 0.0)).normalize();

        Frustum::from_camera_params(
            camera_position,
            camera_forward,
            camera_right,
            aspect_ratio,
            NEAR_PLANE_DISTANCE,
            FAR_PLANE_DISTANCE,
            FOV_Y_DEG,
        )
    }
}
