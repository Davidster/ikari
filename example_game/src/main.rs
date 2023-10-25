mod ball;
mod character;
mod game;
mod game_state;
mod physics_ball;
mod revolver;
mod ui_overlay;

use crate::game::handle_window_resize;
use crate::game::init_game_state;
use crate::game::process_device_input;
use crate::game::process_window_input;
use crate::game::update_game_state;

use ikari::engine_state::EngineState;
use ikari::renderer::BaseRenderer;
use ikari::renderer::Renderer;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const DXC_PATH: &str = "dxc/";

async fn start() {
    let run_result = async {
        let application_start_time = ikari::time::Instant::now();

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

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;

            let dom_window = web_sys::window().unwrap();
            let document = dom_window.document().unwrap();

            let canvas_container = document
                .get_element_by_id("canvas_container")
                .unwrap()
                .dyn_into::<web_sys::HtmlElement>()
                .unwrap();
            window.set_inner_size(winit::dpi::LogicalSize::new(
                (canvas_container.offset_width() as f64 * window.scale_factor()) as u32,
                (canvas_container.offset_height() as f64 * window.scale_factor()) as u32,
            ));

            let canvas = web_sys::Element::from(window.canvas());
            canvas_container.append_child(&canvas).unwrap();
        }

        log::debug!("window: {:?}", application_start_time.elapsed());

        let (base_renderer, mut surface_data) = {
            let backends = if cfg!(target_os = "windows") {
                wgpu::Backends::from(wgpu::Backend::Vulkan)
                // wgpu::Backends::PRIMARY
            } else {
                wgpu::Backends::PRIMARY
            };
            BaseRenderer::with_window(backends, Some(DXC_PATH.into()), &window).await?
        };

        log::debug!("base render: {:?}", application_start_time.elapsed());

        let mut renderer = Renderer::new(
            base_renderer,
            surface_data.surface_config.format,
            (
                surface_data.surface_config.width,
                surface_data.surface_config.height,
            ),
        )
        .await?;

        log::debug!("renderer: {:?}", application_start_time.elapsed());

        let mut engine_state = EngineState::new()?;

        let game_state =
            init_game_state(&mut engine_state, &mut renderer, &mut surface_data, &window).await?;

        log::debug!("game state: {:?}", application_start_time.elapsed());

        // this will block while the game is running
        ikari::gameloop::run(
            window,
            event_loop,
            game_state,
            engine_state,
            renderer,
            surface_data,
            |game_context| {
                update_game_state(game_context);
            },
            |game_context, window_event| {
                process_window_input(game_context, window_event);
            },
            |game_context, device_event| {
                process_device_input(game_context, device_event);
            },
            |game_context, new_size| {
                handle_window_resize(game_context, new_size);
            },
            application_start_time,
        );

        anyhow::Ok(())
    }
    .await;

    if let Err(err) = run_result {
        log::error!(
            "Error setting up game / render state: {}\n{}",
            err,
            err.backtrace()
        );
        #[cfg(target_arch = "wasm32")]
        show_error_div();
    }
}

fn env_var_is_defined(var: &str) -> bool {
    match std::env::var(var) {
        Ok(val) => !val.is_empty(),
        Err(_) => false,
    }
}

fn main() {
    if !env_var_is_defined("RUST_BACKTRACE") {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    if env_var_is_defined("RUST_LOG") {
        env_logger::init();
    } else {
        env_logger::builder()
            .filter(Some(env!("CARGO_PKG_NAME")), log::LevelFilter::Info)
            .filter(Some(env!("CARGO_BIN_NAME")), log::LevelFilter::Info)
            .filter(Some("ikari"), log::LevelFilter::Info)
            .filter(Some("wgpu"), log::LevelFilter::Warn)
            .init();
    }

    #[cfg(feature = "tracy-n-alloc")]
    {
        use profiling::tracy_client::ProfiledAllocator;
        #[global_allocator]
        static GLOBAL: ProfiledAllocator<std::alloc::System> =
            ProfiledAllocator::new(std::alloc::System, 100);
    }
    #[cfg(feature = "tracy")]
    profiling::tracy_client::Client::start();

    ikari::block_on(start());
}

#[cfg(target_arch = "wasm32")]
pub fn show_error_div() {
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|document| {
            let div = document
                .create_element("div")
                .expect("Couldn't create div element");
            div.set_class_name("fatalerror");
            div.set_text_content(Some("Fatal error occured. See console log for details"));
            document.body().unwrap().append_child(&div).ok()
        })
        .expect("Couldn't append error message to document body.");
}

#[cfg(target_arch = "wasm32")]
fn panic_hook(info: &std::panic::PanicInfo) {
    console_error_panic_hook::hook(info);
    show_error_div();
}

#[cfg(target_arch = "wasm32")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    std::panic::set_hook(Box::new(panic_hook));
    console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
    start().await;
}
