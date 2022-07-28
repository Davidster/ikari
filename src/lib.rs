mod animation;
mod audio;
mod ball;
mod buffer;
mod camera;
mod character;
mod file_loader;
mod game;
mod game_state;
mod gameloop;
mod gltf_loader;
mod helpers;
mod light;
mod logger;
mod mesh;
mod physics;
mod physics_ball;
mod player_controller;
mod renderer;
mod revolver;
mod scene;
mod skinning;
mod texture;
mod time;
mod time_tracker;
mod transform;

use animation::*;
use audio::*;
use ball::*;
use buffer::*;
use camera::*;
use character::*;
use file_loader::*;
use game::*;
use game_state::*;
use gltf_loader::*;
use helpers::*;
use light::*;
use logger::*;
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
use time::*;
use time_tracker::*;
use transform::*;

use cgmath::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

async fn start() {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = {
        log::info!("1");
        let window = winit::window::WindowBuilder::new()
            .with_title("David's window name")
            // .with_visible(false)
            .build(&event_loop)
            .expect("Failed to create window");

        if !cfg!(target_arch = "wasm32") {
            // window.set_inner_size(winit::dpi::LogicalSize::new(1000.0f32, 1000.0f32));
            // window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            window.set_inner_size(winit::dpi::PhysicalSize::new(1920.0, 1080.0));
        }

        log::info!("2");

        #[cfg(target_arch = "wasm32")]
        {
            log::info!("3");
            // Winit prevents sizing with CSS, so we have to set
            // the size manually when on web.
            use winit::dpi::PhysicalSize;
            window.set_inner_size(PhysicalSize::new(450, 400));

            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|document| document.body())
                .and_then(|body| {
                    let canvas = web_sys::Element::from(window.canvas());
                    body.append_child(&canvas).ok()?;
                    Some(())
                })
                .expect("Couldn't append canvas to document body.");

            log::info!("4");
        }

        Some(window)
        // for selecting video modes!
        // if cfg!(target_os = "macos") {
        //     Some(window)
        // } else {
        //     let monitor = window.current_monitor().unwrap();
        //     let video_modes: Vec<winit::monitor::VideoMode> = monitor.video_modes().collect();
        //     let video_mode_labels: Vec<String> = video_modes
        //         .iter()
        //         .map(|video_mode| format!("{:}", video_mode))
        //         .collect();
        //     // println!("{:}", video_modes[0]);

        //     let selected_video_mode_index = dialoguer::Select::new()
        //         .items(&video_mode_labels)
        //         .default(0)
        //         .interact_opt()
        //         .expect("Dialoguer failed");

        //     match selected_video_mode_index {
        //         Some(selected_video_mode_index) => {
        //             window.set_fullscreen(Some(winit::window::Fullscreen::Exclusive(
        //                 video_modes[selected_video_mode_index].clone(),
        //             )));
        //             Some(window)
        //         }
        //         None => {
        //             println!("No video mode selected");
        //             None
        //         }
        //     }
        // }
    };
    log::info!("5");
    if let Some(window) = window {
        let mut logger = Logger::new();
        log::info!("6");
        let mut base_render_state = BaseRendererState::new(&window).await;
        log::info!("7");
        let run_result = async {
            let (game_scene, render_buffers) =
                init_scene(&mut base_render_state, &mut logger).await?;
            log::info!("8");
            let mut renderer_state =
                RendererState::new(render_buffers, base_render_state, &mut logger).await?;
            log::info!("9");
            let game_state = init_game_state(game_scene, &mut renderer_state, &mut logger).await?;
            log::info!("10");
            gameloop::run(window, event_loop, game_state, renderer_state, logger); // this will block while the game is running
            anyhow::Ok(())
        }
        .await;

        if let Err(err) = run_result {
            log::error!(
                "Error setting up game / render state: {}\n{}",
                err,
                err.backtrace()
            );
            eprintln!(
                "Error setting up game / render state: {}\n{}",
                err,
                err.backtrace()
            )
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }
    start().await;
}
