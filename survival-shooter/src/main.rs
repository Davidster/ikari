mod camera;
mod gameloop;
mod helpers;
mod logger;
mod renderer;
mod texture;
mod transform;

use std::{thread, time::Duration};

use camera::*;
use gameloop::*;
use helpers::*;
use logger::*;
use renderer::*;
use texture::*;

use cgmath::prelude::*;

async fn start() {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
        .with_inner_size(winit::dpi::PhysicalSize::new(1000.0, 1000.0))
        // .with_inner_size(winit::dpi::LogicalSize::new(1000.0, 1000.0))
        .with_title("David's window name")
        .build(&event_loop)
        .unwrap();

    let renderer_state = RendererState::new(&window)
        .await
        .expect("Failed to create renderer state");

    run(window, event_loop, renderer_state).await;
}

fn main() {
    // let term = console::Term::stdout();

    // let mut i = 1;

    // loop {
    //     term.clear_screen().unwrap();
    //     term.write_line(&format!("Interation {:?}", i)).unwrap();
    //     term.write_line("Hello World a").unwrap();
    //     term.write_line("Hello World b").unwrap();
    //     term.write_line("Hello World c").unwrap();
    //     thread::sleep(Duration::from_millis(1000));
    //     i += 1;
    // }
    env_logger::init();
    pollster::block_on(start());
}
