use super::*;
use cgmath::{Deg, Matrix4, Rad, Vector3};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent},
};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const Z_NEAR: f32 = 0.01;
const Z_FAR: f32 = 100.0;
const FOV_Y: Deg<f32> = Deg(45.0);
const ASPECT_RATION: f32 = FRAME_WIDTH as f32 / FRAME_HEIGHT as f32;

pub struct Camera {
    horizontal_rotation: Rad<f32>,
    vertical_rotation: Rad<f32>,
    position: Vector3<f32>,
}

impl Camera {
    pub fn new(initial_position: Vector3<f32>) -> Self {
        Camera {
            horizontal_rotation: Rad(0.0),
            vertical_rotation: Rad(0.0),
            position: initial_position,
        }
    }

    pub fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX
            * make_perspective_matrix(Z_NEAR, Z_FAR, FOV_Y.into(), ASPECT_RATION)
            // * make_translation_matrix(self.position) //TODO: try to re-add the position here, might need to apply it in a weird order doe
            // * make_rotation_matrix(0.0, Rad::from(Deg(180.0)).0, 0.0)
        * make_rotation_matrix(0.0, 0.0, self.vertical_rotation.0)
        * make_rotation_matrix(self.horizontal_rotation.0, 0.0, 0.0)
    }
}

#[derive(Clone, Debug)]
enum MouseState {
    NEVER_ENTERED,
    RESET,
    ACTIVE(MousePositionState),
}

#[derive(Clone, Debug)]
struct MousePositionState {
    last_processed: PhysicalPosition<f64>,
    current: PhysicalPosition<f64>,
}

pub struct CameraController {
    mouse_state: MouseState,
    pub speed: f32,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            mouse_state: MouseState::NEVER_ENTERED,
            speed,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved {
                position: new_position,
                ..
            } => {
                self.mouse_state = match self.mouse_state {
                    MouseState::NEVER_ENTERED => MouseState::NEVER_ENTERED,
                    MouseState::RESET => MouseState::ACTIVE(MousePositionState {
                        current: *new_position,
                        last_processed: *new_position,
                    }),
                    MouseState::ACTIVE(MousePositionState { last_processed, .. }) => {
                        MouseState::ACTIVE(MousePositionState {
                            current: *new_position,
                            last_processed,
                        })
                    }
                };
            }
            WindowEvent::CursorLeft { .. } => {
                println!("Cursor left");
                self.mouse_state = MouseState::RESET;
            }
            WindowEvent::CursorEntered { .. } => {
                println!("Cursor entered");
                self.mouse_state = MouseState::RESET;
            }
            _ => {}
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        self.mouse_state = if let MouseState::ACTIVE(mut mouse_state) = self.mouse_state.clone() {
            if !mouse_state.current.eq(&mouse_state.last_processed) {
                let mouse_sensitivity = 0.01;
                let d_x =
                    (mouse_state.current.x - mouse_state.last_processed.x) * mouse_sensitivity;
                let d_y =
                    (mouse_state.current.y - mouse_state.last_processed.y) * mouse_sensitivity;

                camera.horizontal_rotation += Rad(d_x as f32);
                camera.vertical_rotation = Rad((camera.vertical_rotation.0 + Rad(d_y as f32).0)
                    .min(Rad::from(Deg(90.0)).0)
                    .max(Rad::from(Deg(-90.0)).0));
                mouse_state.last_processed = mouse_state.current;
                MouseState::ACTIVE(mouse_state)
            } else {
                self.mouse_state.clone()
            }
        } else {
            self.mouse_state.clone()
        }
    }
}
