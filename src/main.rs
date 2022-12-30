mod animation;
mod audio;
mod ball;
mod buffer;
mod camera;
mod character;
mod collisions;
mod default_textures;
mod game;
mod game_state;
mod gameloop;
mod gltf_loader;
mod light;
mod logger;
mod math;
mod mesh;
mod physics;
mod physics_ball;
mod player_controller;
mod renderer;
mod revolver;
mod scene;
mod scene_tree;
mod skinning;
mod texture;
mod time_tracker;
mod transform;

use animation::*;
use audio::*;
use ball::*;
use buffer::*;
use camera::*;
use cgmath::prelude::*;
use character::*;
use collisions::*;
use default_textures::*;
use game::*;
use game_state::*;
use gltf_loader::*;
use light::*;
use logger::*;
use math::*;
use mesh::*;
use physics::*;
use physics_ball::*;
use player_controller::*;
use rapier3d::prelude::*;
use renderer::*;
use revolver::*;
use scene::*;
use skinning::*;
use texture::*;
use time_tracker::*;
use transform::*;

async fn start() {
    let event_loop = winit::event_loop::EventLoop::new();

    let window = {
        let (width, height) = match event_loop.primary_monitor() {
            None => (1920, 1080), // Most widespread resolution in 2022.
            Some(handle) => (handle.size().width, handle.size().height),
        };
        let inner_size = winit::dpi::PhysicalSize::new(width * 3 / 4, height * 3 / 4);
        let title = format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        winit::window::WindowBuilder::new()
            //.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
            .with_inner_size(inner_size)
            .with_title(title)
            .with_maximized(false)
            // .with_visible(false)
            .build(&event_loop)
            .expect("Failed to create window")
    };

    let base_render_state = {
        let backends = if cfg!(target_os = "linux") {
            wgpu::Backends::from(wgpu::Backend::Vulkan)
        } else {
            wgpu::Backends::all()
        };
        BaseRendererState::new(&window, backends, wgpu::PresentMode::AutoVsync).await
    };

    let run_result = async {
        let game_scene = Scene::default();
        let render_buffers = RenderBuffers::default();
        let mut renderer_state = RendererState::new(render_buffers, base_render_state).await?;
        let game_state = init_game_state(game_scene, &mut renderer_state)?;
        gameloop::run(window, event_loop, game_state, renderer_state); // this will block while the game is running
        anyhow::Ok(())
    }
    .await;

    if let Err(err) = run_result {
        eprintln!(
            "Error setting up game / render state: {}\n{}",
            err,
            err.backtrace()
        )
    }
}

fn main() {
    env_logger::init();

    #[cfg(feature = "profile-with-tracy")]
    profiling::tracy_client::Client::start();

    pollster::block_on(start());
}
