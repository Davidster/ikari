use super::*;
use cgmath::{Deg, InnerSpace, Matrix4, Rad, Vector3};
use winit::{
    dpi::PhysicalPosition,
    event::{
        DeviceEvent, ElementState, KeyboardInput, MouseScrollDelta, VirtualKeyCode, WindowEvent,
    },
    window::Window,
};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const Z_NEAR: f32 = 0.01;
const Z_FAR: f32 = 10000.0;
const FOV_Y: Deg<f32> = Deg(45.0);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CameraPose {
    horizontal_rotation: Rad<f32>,
    vertical_rotation: Rad<f32>,
    position: Vector3<f32>,
}

pub struct Camera {
    pose: CameraPose,
}

impl Camera {
    pub fn new(initial_position: Vector3<f32>) -> Self {
        Camera {
            pose: CameraPose {
                horizontal_rotation: Rad(0.0),
                vertical_rotation: Rad(0.0),
                position: initial_position,
            },
        }
    }

    pub fn build_view_projection_matrix(&self, window: &winit::window::Window) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX
            * make_perspective_matrix(
                Z_NEAR,
                Z_FAR,
                FOV_Y.into(),
                window.inner_size().width as f32 / window.inner_size().height as f32,
            )
            * make_rotation_matrix(0.0, 0.0, -self.pose.vertical_rotation.0)
            * make_rotation_matrix(-self.pose.horizontal_rotation.0, 0.0, 0.0)
            * make_translation_matrix(-self.pose.position)
    }

    pub fn get_direction_vector(&self) -> Vector3<f32> {
        let horizontal_scale = self.pose.vertical_rotation.0.cos();
        Vector3::new(
            (self.pose.horizontal_rotation.0 + std::f32::consts::PI).sin() * horizontal_scale,
            self.pose.vertical_rotation.0.sin(),
            (self.pose.horizontal_rotation.0 + std::f32::consts::PI).cos() * horizontal_scale,
        )
        .normalize()
    }
}

pub struct CameraController {
    // mouse_state: MouseState,
    unprocessed_delta: Option<(f64, f64)>,
    cursor_in_window: bool,

    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,

    target_pose: CameraPose,

    pub speed: f32,
}

impl CameraController {
    pub fn new(speed: f32, camera: &Camera) -> Self {
        Self {
            unprocessed_delta: None,
            cursor_in_window: false,

            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,

            target_pose: camera.pose,

            speed,
        }
    }

    pub fn process_device_events(
        &mut self,
        event: &DeviceEvent,
        _window: &mut Window,
        logger: &mut Logger,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta: (d_x, d_y) } if self.cursor_in_window => {
                self.unprocessed_delta = match self.unprocessed_delta {
                    Some((x, y)) => Some((x + d_x, y + d_y)),
                    None => Some((*d_x, *d_y)),
                };
            }
            DeviceEvent::MouseWheel { delta } if self.cursor_in_window => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => *y as f32,
                };
                self.speed = (self.speed - (scroll_amount * 0.0001)).max(0.01).min(5.0);
                logger.log(&format!("Speed: {:?}", self.speed));
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
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
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
            // TODO: use window focus event instead of cursor events
            WindowEvent::CursorLeft { .. } => {
                window.set_cursor_visible(true);
                self.cursor_in_window = false;
                window
                    .set_cursor_grab(false)
                    .expect("Couldn't release cursor");
            }
            WindowEvent::CursorEntered { .. } => {
                window.set_cursor_visible(false);
                self.cursor_in_window = true;
                window.set_cursor_grab(true).expect("Couldn't grab cursor");
            }
            _ => {}
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        if let Some((d_x, d_y)) = self.unprocessed_delta {
            let mouse_sensitivity = 0.003;

            self.target_pose.horizontal_rotation += Rad(-d_x as f32 * mouse_sensitivity);
            self.target_pose.vertical_rotation = Rad((self.target_pose.vertical_rotation.0
                + Rad(-d_y as f32 * mouse_sensitivity).0)
                .min(Rad::from(Deg(90.0)).0)
                .max(Rad::from(Deg(-90.0)).0));
        }
        self.unprocessed_delta = None;

        let forward_direction = camera.get_direction_vector();
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
            self.target_pose.position += movement_vector.normalize() * self.speed;
        }

        if self.target_pose.position.y < 0.0 {
            self.target_pose.position.y = 0.1;
        }

        let pos_lerp_factor = 0.3;
        let rot_lerp_factor = 0.5;

        camera.pose.position = camera
            .pose
            .position
            .lerp(self.target_pose.position, pos_lerp_factor);

        camera.pose.vertical_rotation = Rad(lerp_f32(
            camera.pose.vertical_rotation.0,
            self.target_pose.vertical_rotation.0,
            rot_lerp_factor,
        ));

        camera.pose.horizontal_rotation = Rad(lerp_f32(
            camera.pose.horizontal_rotation.0,
            self.target_pose.horizontal_rotation.0,
            rot_lerp_factor,
        ));
    }
}
