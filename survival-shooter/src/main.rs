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

use cgmath::prelude::*;

async fn start() {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
        .with_inner_size(winit::dpi::PhysicalSize::new(1000.0, 1000.0))
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
