#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use ikari::game::*;
use ikari::renderer::*;
use ikari::scene::*;

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
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("Couldn't append canvas to document body.");

        log::info!("4");
    }

    let base_render_state = {
        let backends = if cfg!(target_os = "linux") {
            wgpu::Backends::from(wgpu::Backend::Vulkan)
        } else {
            wgpu::Backends::from(wgpu::Backend::Dx12)
            // wgpu::Backends::PRIMARY
        };
        BaseRenderer::new(&window, backends, wgpu::PresentMode::AutoNoVsync).await
    };

    let run_result = async {
        let game_scene = Scene::default();

        let mut renderer = Renderer::new(base_render_state, &window).await?;

        let game_state = init_game_state(game_scene, &mut renderer).await?;

        ikari::gameloop::run(window, event_loop, game_state, renderer); // this will block while the game is running
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

    #[cfg(feature = "tracy-n-alloc")]
    {
        use profiling::tracy_client::ProfiledAllocator;
        #[global_allocator]
        static GLOBAL: ProfiledAllocator<std::alloc::System> =
            ProfiledAllocator::new(std::alloc::System, 100);
    }
    #[cfg(feature = "tracy")]
    profiling::tracy_client::Client::start();

    pollster::block_on(start());
}

#[cfg(target_arch = "wasm32")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
    start().await;
}
