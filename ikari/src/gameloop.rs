use std::sync::Arc;

use crate::engine_state::EngineState;
use crate::renderer::{Renderer, SurfaceData};
use crate::time::Instant;
use crate::ui::{IkariUiContainer, UiProgram, UiProgramEvents};
use crate::web_canvas_manager::WebCanvasManager;

use wgpu::PresentMode;
use winit::application::ApplicationHandler;
use winit::event_loop::ActiveEventLoop;
use winit::{
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowId},
};

pub trait GameState<UiOverlay>
where
    UiOverlay: UiProgram + UiProgramEvents + 'static,
{
    fn get_ui_container(&mut self) -> &mut IkariUiContainer<UiOverlay>;
}

pub struct GameContext<'a, GameState> {
    pub game_state: &'a mut GameState,
    pub engine_state: &'a mut EngineState,
    pub renderer: &'a mut Renderer,
    pub surface_data: &'a mut SurfaceData,
    pub window: &'a winit::window::Window,
    pub elwt: &'a ActiveEventLoop,
}

/// Holds everything the old event-loop closure used to capture. winit 0.30 replaced
/// the closure-based `EventLoop::run` with the `ApplicationHandler` trait, so this
/// state has to live in a struct rather than in a closure's environment.
struct IkariApp<
    OnUpdateFunction,
    OnWindowEventFunction,
    OnDeviceEventFunction,
    OnSurfaceResizeFunction,
    GameStateType,
    UiOverlay,
> {
    window: Arc<Window>,
    game_state: GameStateType,
    engine_state: EngineState,
    renderer: Renderer,
    surface_data: SurfaceData,
    on_update: OnUpdateFunction,
    on_window_event: OnWindowEventFunction,
    on_device_event: OnDeviceEventFunction,
    on_surface_resize: OnSurfaceResizeFunction,
    application_start_time: Instant,
    web_canvas_manager: WebCanvasManager,
    logged_start_time: bool,
    pending_resize_event: Option<winit::dpi::PhysicalSize<u32>>,
    force_reconfigure_surface: bool,
    /// `UiOverlay` only appears in `GameStateType: GameState<UiOverlay>`, which isn't
    /// enough to constrain it on the impls, so carry it explicitly.
    _ui_overlay: std::marker::PhantomData<UiOverlay>,
}

impl<
        OnUpdateFunction,
        OnWindowEventFunction,
        OnDeviceEventFunction,
        OnSurfaceResizeFunction,
        GameStateType,
        UiOverlay,
    >
    IkariApp<
        OnUpdateFunction,
        OnWindowEventFunction,
        OnDeviceEventFunction,
        OnSurfaceResizeFunction,
        GameStateType,
        UiOverlay,
    >
where
    OnUpdateFunction: FnMut(GameContext<GameStateType>) + 'static,
    OnWindowEventFunction: FnMut(GameContext<GameStateType>, &WindowEvent) + 'static,
    OnDeviceEventFunction: FnMut(GameContext<GameStateType>, &DeviceEvent) + 'static,
    OnSurfaceResizeFunction:
        FnMut(GameContext<GameStateType>, winit::dpi::PhysicalSize<u32>) + 'static,
    UiOverlay: UiProgram + UiProgramEvents + 'static,
    GameStateType: GameState<UiOverlay> + 'static,
{
    fn on_redraw_requested(&mut self, elwt: &ActiveEventLoop) {
        self.window.request_redraw();

        let is_vsync_is_on = !matches!(
            self.surface_data.surface_config.present_mode,
            PresentMode::Immediate | PresentMode::AutoNoVsync
        );
        let should_sleep = self.engine_state.framerate_limiter.update_and_sleep(
            &self.engine_state.time_tracker,
            is_vsync_is_on,
            true,
        );
        if should_sleep {
            return;
        }

        self.engine_state
            .time_tracker
            .on_sleep_and_inputs_completed();

        (self.on_update)(GameContext {
            game_state: &mut self.game_state,
            engine_state: &mut self.engine_state,
            renderer: &mut self.renderer,
            surface_data: &mut self.surface_data,
            window: &self.window,
            elwt,
        });

        self.engine_state.time_tracker.on_update_completed();

        self.engine_state.asset_binder.update(
            self.renderer.base.clone(),
            self.renderer.constant_data.clone(),
            self.engine_state.asset_loader.clone(),
        );

        let resized = self
            .renderer
            .reconfigure_surface_if_needed(&mut self.surface_data, self.force_reconfigure_surface);
        if resized {
            self.pending_resize_event = Some(winit::dpi::PhysicalSize::new(
                self.surface_data.surface_config.width,
                self.surface_data.surface_config.height,
            ));
        }

        let surface_texture_result = self.surface_data.surface.get_current_texture();

        self.engine_state.time_tracker.on_get_surface_completed();

        if let Some(new_size) = self.pending_resize_event.take() {
            (self.on_surface_resize)(
                GameContext {
                    game_state: &mut self.game_state,
                    engine_state: &mut self.engine_state,
                    renderer: &mut self.renderer,
                    surface_data: &mut self.surface_data,
                    window: &self.window,
                    elwt,
                },
                new_size,
            );
        }

        self.force_reconfigure_surface = false;
        match surface_texture_result {
            Ok(surface_texture) => {
                if let Err(err) = self.renderer.render(
                    &mut self.engine_state,
                    &self.surface_data,
                    surface_texture,
                    self.game_state.get_ui_container(),
                ) {
                    log::error!("{err:?}");
                }
            }
            Err(err) => match err {
                wgpu::SurfaceError::OutOfMemory => {
                    log::error!("Received surface error: {err:?}. Application will exit");
                    elwt.exit();
                }
                wgpu::SurfaceError::Timeout => {
                    log::warn!("Received surface error: {err:?}. Frame will be skipped");
                }
                wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                    self.force_reconfigure_surface = true;
                    log::warn!("Received surface error: {err:?}");
                }
                other => {
                    log::warn!("Received surface error: {other:?}");
                }
            },
        };

        self.engine_state.time_tracker.on_render_completed();

        // start the frame right away so that input processing gets tracked by the time tracker
        self.engine_state.time_tracker.on_frame_started();
        if !self.logged_start_time {
            log::debug!(
                "Took {:?} from process startup till first frame",
                self.application_start_time.elapsed()
            );
            self.logged_start_time = true;
        }
    }
}

impl<
        OnUpdateFunction,
        OnWindowEventFunction,
        OnDeviceEventFunction,
        OnSurfaceResizeFunction,
        GameStateType,
        UiOverlay,
    > ApplicationHandler
    for IkariApp<
        OnUpdateFunction,
        OnWindowEventFunction,
        OnDeviceEventFunction,
        OnSurfaceResizeFunction,
        GameStateType,
        UiOverlay,
    >
where
    OnUpdateFunction: FnMut(GameContext<GameStateType>) + 'static,
    OnWindowEventFunction: FnMut(GameContext<GameStateType>, &WindowEvent) + 'static,
    OnDeviceEventFunction: FnMut(GameContext<GameStateType>, &DeviceEvent) + 'static,
    OnSurfaceResizeFunction:
        FnMut(GameContext<GameStateType>, winit::dpi::PhysicalSize<u32>) + 'static,
    UiOverlay: UiProgram + UiProgramEvents + 'static,
    GameStateType: GameState<UiOverlay> + 'static,
{
    /// The window is created before the loop starts, so there is nothing to do here.
    /// TODO: window/renderer creation should happen here instead, see example_game main.rs
    fn resumed(&mut self, _elwt: &ActiveEventLoop) {}

    fn window_event(&mut self, elwt: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        self.web_canvas_manager.on_update();

        if window_id != self.window.id() {
            return;
        }

        if matches!(event, WindowEvent::RedrawRequested) {
            self.on_redraw_requested(elwt);
            return;
        }

        match &event {
            WindowEvent::Resized(size) => {
                self.renderer.resize_surface(&self.surface_data, *size);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.renderer
                    .resize_surface(&self.surface_data, self.window.inner_size());
            }
            WindowEvent::CloseRequested => {
                elwt.exit();
            }
            WindowEvent::Moved(_) => {
                self.engine_state
                    .framerate_limiter
                    .set_monitor_refresh_rate(
                        self.window
                            .current_monitor()
                            .and_then(|window| window.refresh_rate_millihertz())
                            .map(|millihertz| millihertz as f32 / 1000.0),
                    );
            }
            _ => {}
        };

        (self.on_window_event)(
            GameContext {
                game_state: &mut self.game_state,
                engine_state: &mut self.engine_state,
                renderer: &mut self.renderer,
                surface_data: &mut self.surface_data,
                window: &self.window,
                elwt,
            },
            &event,
        );
    }

    fn device_event(&mut self, elwt: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        self.web_canvas_manager.on_update();

        (self.on_device_event)(
            GameContext {
                game_state: &mut self.game_state,
                engine_state: &mut self.engine_state,
                renderer: &mut self.renderer,
                surface_data: &mut self.surface_data,
                window: &self.window,
                elwt,
            },
            &event,
        );
    }

    fn exiting(&mut self, _elwt: &ActiveEventLoop) {
        self.web_canvas_manager.on_exiting();
        self.engine_state.asset_loader.exit();
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run<
    OnUpdateFunction,
    OnWindowEventFunction,
    OnDeviceEventFunction,
    OnSurfaceResizeFunction,
    GameStateType,
    UiOverlay,
>(
    window: Arc<Window>,
    event_loop: EventLoop<()>,
    game_state: GameStateType,
    mut engine_state: EngineState,
    renderer: Renderer,
    surface_data: SurfaceData,
    on_update: OnUpdateFunction,
    on_window_event: OnWindowEventFunction,
    on_device_event: OnDeviceEventFunction,
    on_surface_resize: OnSurfaceResizeFunction,
    application_start_time: Instant,
) -> anyhow::Result<()>
where
    OnUpdateFunction: FnMut(GameContext<GameStateType>) + 'static,
    OnWindowEventFunction: FnMut(GameContext<GameStateType>, &WindowEvent) + 'static,
    OnDeviceEventFunction: FnMut(GameContext<GameStateType>, &DeviceEvent) + 'static,
    OnSurfaceResizeFunction:
        FnMut(GameContext<GameStateType>, winit::dpi::PhysicalSize<u32>) + 'static,
    UiOverlay: UiProgram + UiProgramEvents + 'static,
    GameStateType: GameState<UiOverlay> + 'static,
{
    let web_canvas_manager = WebCanvasManager::new(window.clone());

    engine_state.framerate_limiter.set_monitor_refresh_rate(
        window
            .current_monitor()
            .and_then(|window| window.refresh_rate_millihertz())
            .map(|millihertz| millihertz as f32 / 1000.0),
    );

    engine_state.time_tracker.on_frame_started();

    let mut app = IkariApp {
        window,
        game_state,
        engine_state,
        renderer,
        surface_data,
        on_update,
        on_window_event,
        on_device_event,
        on_surface_resize,
        application_start_time,
        web_canvas_manager,
        logged_start_time: false,
        pending_resize_event: None,
        force_reconfigure_surface: false,
        _ui_overlay: std::marker::PhantomData,
    };

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run_app(&mut app)?;

    Ok(())
}
