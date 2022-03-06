mod camera;
mod gameloop;
mod helpers;
mod renderer;
mod texture;
mod transform;

use camera::*;
use gameloop::*;
use helpers::*;
use renderer::*;
use texture::*;
use transform::*;

const FRAME_WIDTH: i64 = 1000;
const FRAME_HEIGHT: i64 = 1000;

async fn start() {
    // let mut state = State::new(&window).await;

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(
            FRAME_WIDTH as f64,
            FRAME_HEIGHT as f64,
        ))
        .with_title("David's window name")
        .build(&event_loop)
        .unwrap();

    let renderer_state = RendererState::new(&window)
        .await
        .expect("Failed to create renderer state");

    run(window, event_loop, renderer_state).await;
}

fn main() {
    env_logger::init();
    pollster::block_on(start());
}
