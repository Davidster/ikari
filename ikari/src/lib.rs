/* #[cfg(target_arch = "wasm32")]
use crate::game::*;
#[cfg(target_arch = "wasm32")]
use crate::renderer::*;
#[cfg(target_arch = "wasm32")]
use crate::scene::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*; */

pub mod animation;
pub mod asset_loader;
pub mod audio;
pub mod ball;
pub mod buffer;
pub mod camera;
pub mod character;
pub mod collisions;
pub mod file_loader;
pub mod game;
pub mod game_state;
pub mod gameloop;
pub mod gltf_loader;
pub mod light;
pub mod logger;
pub mod math;
pub mod mesh;
pub mod physics;
pub mod physics_ball;
pub mod player_controller;
pub mod profile_dump;
pub mod renderer;
pub mod revolver;
pub mod sampler_cache;
pub mod scene;
pub mod scene_tree;
pub mod skinning;
pub mod texture;
#[cfg(not(target_arch = "wasm32"))]
pub mod texture_compression;
pub mod time;
pub mod time_tracker;
pub mod transform;
pub mod ui_overlay;

/* #[cfg(target_arch = "wasm32")]
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
        } else if cfg!(target_arch = "wasm32") {
            log::info!("wtf m8");
            wgpu::Backends::from(wgpu::Backend::BrowserWebGpu)
            // wgpu::Backends::PRIMARY
        } else {
            wgpu::Backends::from(wgpu::Backend::Dx12)
            // wgpu::Backends::PRIMARY
        };
        log::info!("backends={:?}", backends);
        BaseRenderer::new(&window, backends, wgpu::PresentMode::AutoNoVsync).await
    };

    let run_result = async {
        let game_scene = Scene::default();

        let mut renderer = Renderer::new(base_render_state, &window).await?;

        let game_state = init_game_state(game_scene, &mut renderer).await?;

        crate::gameloop::run(window, event_loop, game_state, renderer); // this will block while the game is running
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

#[cfg(target_arch = "wasm32")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
    start().await;
}
 */
