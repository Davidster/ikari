use std::sync::Arc;

use crate::engine_state::EngineState;
use crate::renderer::{Renderer, SurfaceData};
use crate::time::Instant;
use crate::ui::{IkariUiContainer, UiProgramEvents};
use crate::web_canvas_manager::WebCanvasManager;

use winit::event_loop::EventLoopWindowTarget;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

pub trait GameState<UiOverlay>
where
    UiOverlay: iced_winit::runtime::Program<Renderer = iced::Renderer> + UiProgramEvents + 'static,
{
    fn get_ui_container(&mut self) -> &mut IkariUiContainer<UiOverlay>;
}

pub struct GameContext<'a, GameState> {
    pub game_state: &'a mut GameState,
    pub engine_state: &'a mut EngineState,
    pub renderer: &'a mut Renderer,
    pub surface_data: &'a mut SurfaceData,
    pub window: &'a winit::window::Window,
    pub elwt: &'a winit::event_loop::EventLoopWindowTarget<()>,
}

#[allow(clippy::too_many_arguments)]
pub fn run<
    OnUpdateFunction,
    OnWindowEventFunction,
    OnDeviceEventFunction,
    OnWindowResizeFunction,
    GameStateType,
    UiOverlay,
>(
    mut window: Arc<Window>,
    event_loop: EventLoop<()>,
    mut game_state: GameStateType,
    mut engine_state: EngineState,
    mut renderer: Renderer,
    mut surface_data: SurfaceData,
    mut on_update: OnUpdateFunction,
    mut on_window_event: OnWindowEventFunction,
    mut on_device_event: OnDeviceEventFunction,
    mut on_window_resize: OnWindowResizeFunction,
    application_start_time: Instant,
) where
    OnUpdateFunction: FnMut(GameContext<GameStateType>) + 'static,
    OnWindowEventFunction: FnMut(GameContext<GameStateType>, &winit::event::WindowEvent) + 'static,
    OnDeviceEventFunction: FnMut(GameContext<GameStateType>, &winit::event::DeviceEvent) + 'static,
    OnWindowResizeFunction:
        FnMut(GameContext<GameStateType>, winit::dpi::PhysicalSize<u32>) + 'static,
    UiOverlay: iced_winit::runtime::Program<Renderer = iced::Renderer> + UiProgramEvents + 'static,
    GameStateType: GameState<UiOverlay> + 'static,
{
    let mut logged_start_time = false;
    let web_canvas_manager = WebCanvasManager::new(window.clone());

    engine_state.framerate_limiter.set_monitor_refresh_rate(
        window
            .current_monitor()
            .and_then(|window| window.refresh_rate_millihertz())
            .map(|millihertz| millihertz as f32 / 1000.0),
    );

    let mut latest_surface_texture_result = None;

    let mut pending_resize_event: Option<winit::dpi::PhysicalSize<u32>> = None;

    engine_state.time_tracker.on_frame_started();

    let handler = move |event: Event<()>, elwt: &EventLoopWindowTarget<()>| {
        web_canvas_manager.on_update(&event);

        match event {
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                // TODO: the time tracker currently doesn't process any of the events happening outside this match block!
                let sleeping = engine_state
                    .framerate_limiter
                    .update(&engine_state.time_tracker);
                if sleeping {
                    window.request_redraw();
                    return;
                }

                engine_state.time_tracker.on_sleep_completed();

                engine_state.asset_binder.update(
                    renderer.base.clone(),
                    renderer.constant_data.clone(),
                    engine_state.asset_loader.clone(),
                );

                if let Some(latest_size) = pending_resize_event.take() {
                    on_window_resize(
                        GameContext {
                            game_state: &mut game_state,
                            engine_state: &mut engine_state,
                            renderer: &mut renderer,
                            surface_data: &mut surface_data,
                            window: &mut window,
                            elwt,
                        },
                        latest_size,
                    );
                }

                on_update(GameContext {
                    game_state: &mut game_state,
                    engine_state: &mut engine_state,
                    renderer: &mut renderer,
                    surface_data: &mut surface_data,
                    window: &mut window,
                    elwt,
                });

                engine_state.time_tracker.on_update_completed();

                // on the first frame we get the surface texture before rendering
                let surface_texture_result = latest_surface_texture_result
                    .take()
                    .unwrap_or_else(|| surface_data.surface.get_current_texture());

                let force_reconfigure_surface = matches!(
                    surface_texture_result,
                    Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated)
                );
                match surface_texture_result {
                    Ok(surface_texture) => {
                        if let Err(err) = renderer.render(
                            &mut engine_state,
                            &surface_data,
                            surface_texture,
                            game_state.get_ui_container(),
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
                            log::warn!("Received surface error: {err:?}");
                        }
                    },
                };

                engine_state.time_tracker.on_render_completed();

                let resized = renderer
                    .reconfigure_surface_if_needed(&mut surface_data, force_reconfigure_surface);
                if resized {
                    pending_resize_event = Some(winit::dpi::PhysicalSize::new(
                        surface_data.surface_config.width,
                        surface_data.surface_config.height,
                    ));
                }

                latest_surface_texture_result = Some(surface_data.surface.get_current_texture());

                engine_state.time_tracker.on_get_surface_completed();

                profiling::finish_frame!();

                engine_state.time_tracker.on_frame_started();
                if !logged_start_time {
                    log::debug!(
                        "Took {:?} from process startup till first frame",
                        application_start_time.elapsed()
                    );
                    logged_start_time = true;
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::LoopExiting => {
                engine_state.asset_loader.exit();
            }
            Event::DeviceEvent { event, .. } => {
                on_device_event(
                    GameContext {
                        game_state: &mut game_state,
                        engine_state: &mut engine_state,
                        renderer: &mut renderer,
                        surface_data: &mut surface_data,
                        window: &mut window,
                        elwt,
                    },
                    &event,
                );
            }
            Event::WindowEvent {
                event, window_id, ..
            } if window_id == window.id() => {
                match &event {
                    WindowEvent::Resized(size) => {
                        if size.width > 0 && size.height > 0 {
                            renderer.resize_surface(&surface_data, *size);
                        }
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        let size = window.inner_size();
                        if size.width > 0 && size.height > 0 {
                            renderer.resize_surface(&surface_data, size);
                        }
                    }
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    }
                    WindowEvent::Moved(_) => {
                        engine_state.framerate_limiter.set_monitor_refresh_rate(
                            window
                                .current_monitor()
                                .and_then(|window| window.refresh_rate_millihertz())
                                .map(|millihertz| millihertz as f32 / 1000.0),
                        );
                    }
                    _ => {}
                };

                on_window_event(
                    GameContext {
                        game_state: &mut game_state,
                        engine_state: &mut engine_state,
                        renderer: &mut renderer,
                        surface_data: &mut surface_data,
                        window: &mut window,
                        elwt,
                    },
                    &event,
                );
            }
            _ => {}
        };
    };
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn(handler);
    }
    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run(handler).unwrap();
}
