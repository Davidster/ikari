use super::*;
use cgmath::{Deg, InnerSpace, Rad, Vector3};
use winit::{
    dpi::PhysicalPosition,
    event::{
        DeviceEvent, ElementState, KeyboardInput, MouseScrollDelta, VirtualKeyCode, WindowEvent,
    },
    window::Window,
};

#[derive(Debug)]
pub struct CameraController {
    unprocessed_delta: Option<(f64, f64)>,
    window_focused: bool,

    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,

    pub current_pose: Camera,
    target_pose: Camera,
    pub speed: f32,
}

impl CameraController {
    pub fn new(speed: f32, camera: Camera) -> Self {
        Self {
            unprocessed_delta: None,
            window_focused: false,

            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,

            current_pose: camera,
            target_pose: camera,

            speed,
        }
    }

    pub fn process_device_events(&mut self, event: &DeviceEvent, logger: &mut Logger) {
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
                self.speed = (self.speed - (scroll_direction * scroll_speed))
                    .max(0.5)
                    .min(300.0);
                logger.log(&format!("Speed: {:?}", self.speed));
            }
            _ => {}
        };
    }

    pub fn process_window_events(
        &mut self,
        event: &WindowEvent,
        window: &mut Window,
        logger: &mut Logger,
    ) {
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
                logger.log(&format!("Window focused: {:?}", focused));
                if let Err(err) = window.set_cursor_grab(*focused) {
                    logger.log(&format!(
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

    pub fn update(&mut self, dt: f32) {
        if let Some((d_x, d_y)) = self.unprocessed_delta {
            let mouse_sensitivity = 0.002;

            self.target_pose.horizontal_rotation += Rad(-d_x as f32 * mouse_sensitivity);
            self.target_pose.vertical_rotation = Rad((self.target_pose.vertical_rotation.0
                + Rad(-d_y as f32 * mouse_sensitivity).0)
                .min(Rad::from(Deg(90.0)).0)
                .max(Rad::from(Deg(-90.0)).0));
        }
        self.unprocessed_delta = None;

        let forward_direction = self.current_pose.get_direction_vector();
        let up_direction = Vector3::new(0.0, 1.0, 0.0);
        let right_direction = forward_direction.cross(up_direction);

        let movement_vector = {
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

            if self.is_up_pressed {
                add_movement(up_direction);
            } else if self.is_down_pressed {
                add_movement(-up_direction);
            }

            res
        };

        if let Some(movement_vector) = movement_vector {
            self.target_pose.position += movement_vector.normalize() * self.speed * dt;
        }

        let min_y = 0.5;
        if self.target_pose.position.y < min_y {
            self.target_pose.position.y = min_y;
        }

        let max_y = 1.0;
        if self.target_pose.position.y > max_y {
            self.target_pose.position.y = max_y;
        }

        let ema_adjusted_dt = dt * 60.0;
        let pos_lerp_factor = (0.3 * ema_adjusted_dt).min(1.0);
        // let rot_lerp_factor = (0.5 * ema_adjusted_dt).min(1.0);
        let rot_lerp_factor = 1.0;

        self.current_pose.position = self
            .current_pose
            .position
            .lerp(self.target_pose.position, pos_lerp_factor);

        self.current_pose.vertical_rotation = Rad(lerp(
            self.current_pose.vertical_rotation.0,
            self.target_pose.vertical_rotation.0,
            rot_lerp_factor,
        ));

        self.current_pose.horizontal_rotation = Rad(lerp(
            self.current_pose.horizontal_rotation.0,
            self.target_pose.horizontal_rotation.0,
            rot_lerp_factor,
        ));
    }
}
