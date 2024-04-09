use std::sync::Arc;

use crate::engine_state::EngineState;
use crate::renderer::{Renderer, SurfaceData};
use crate::time::Instant;
use crate::ui::IkariUiContainer;
use crate::web_canvas_manager::WebCanvasManager;

use winit::event_loop::EventLoopWindowTarget;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

pub trait GameState<UiOverlay>
where
    UiOverlay: iced_winit::runtime::Program<Renderer = iced::Renderer> + 'static,
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
    UiOverlay: iced_winit::runtime::Program<Renderer = iced::Renderer> + 'static,
    GameStateType: GameState<UiOverlay> + 'static,
{
    let mut logged_start_time = false;
    let _last_frame_start_time: Option<Instant> = None;
    let web_canvas_manager = WebCanvasManager::new(window.clone());

    let mut monitor_refresh_rate = window
        .current_monitor()
        .and_then(|window| window.refresh_rate_millihertz());

    let mut latest_surface_texture_result = None;

    let mut sleep_start = None;

    let handler = move |event: Event<()>, elwt: &EventLoopWindowTarget<()>| {
        web_canvas_manager.on_update(&event);

        match event {
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                // TODO: the time tracker currently doesn't process any of the events happening outside this match block!
                // TODO: the real 'frame start' time should come immediately after get_current_texture call, because this ends right
                //       after the vblank under vsync. on_frame_started and profiling::finish_frame! should be moved around accordingly.
                if sleep_start.is_none() {
                    dbg!("on frame start");
                    engine_state.time_tracker.on_frame_started();

                    if !logged_start_time && engine_state.time_tracker.last_frame_times().is_some()
                    {
                        log::debug!(
                            "Took {:?} from process startup till first frame",
                            application_start_time.elapsed()
                        );
                        logged_start_time = true;
                    }
                }

                const INTELLIGENT_SLEEP: bool = true;

                let sleeping =
                    if let (Some(last_frame_busy_time_secs), Some(monitor_refresh_rate)) = (
                        engine_state.time_tracker.last_frame_busy_time_secs(),
                        monitor_refresh_rate,
                    ) {
                        profiling::scope!("Sleep");
                        let refresh_rate_period_secs = 1000.0 / monitor_refresh_rate as f64;
                        let sleep_time =
                            (refresh_rate_period_secs - last_frame_busy_time_secs).max(0.0) * 1.01;
                        let sleep_time = 0.0;
                        // dbg!(
                        //     refresh_rate_period_secs * 1000.0,
                        //     last_frame_busy_time_secs * 1000.0,
                        //     delta * 1000.0
                        // );

                        if sleep_time > 0.0 {
                            let sleep_start_value =
                                sleep_start.get_or_insert_with(|| crate::time::Instant::now());

                            if INTELLIGENT_SLEEP {
                                if sleep_start_value.elapsed().as_secs_f64() < sleep_time {
                                    // TODO: yield here?
                                    // std::thread::yield_now();
                                    window.request_redraw();
                                    true
                                } else {
                                    false
                                }
                            } else {
                                loop {
                                    if sleep_start_value.elapsed().as_secs_f64() >= sleep_time {
                                        break false;
                                    }
                                    std::thread::yield_now();
                                }
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                // let actual_sleep_time = sleep_start.elapsed().as_secs_f64();
                // dbg!(expected_sleep_time * 1000.0);
                // dbg!(actual_sleep_time * 1000.0);
                // dbg!((expected_sleep_time - actual_sleep_time).abs() * 1000.0);

                if !sleeping {
                    sleep_start = None;

                    engine_state.time_tracker.on_sleep_completed();

                    engine_state.asset_binder.update(
                        renderer.base.clone(),
                        renderer.constant_data.clone(),
                        engine_state.asset_loader.clone(),
                    );

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
                        .unwrap_or_else(|| dbg!(surface_data.surface.get_current_texture()));

                    // TODO: merge into a match
                    if let Err(err) = &surface_texture_result {
                        match err {
                            // Reconfigure the surface if lost
                            wgpu::SurfaceError::Lost => {
                                let size = window.inner_size();

                                renderer.resize_surface(&mut surface_data, size);
                                on_window_resize(
                                    GameContext {
                                        game_state: &mut game_state,
                                        engine_state: &mut engine_state,
                                        renderer: &mut renderer,
                                        surface_data: &mut surface_data,
                                        window: &mut window,
                                        elwt,
                                    },
                                    size,
                                );
                            }
                            wgpu::SurfaceError::OutOfMemory => elwt.exit(),
                            _ => log::error!("{err:?}"),
                        }
                    }

                    // TODO: merge into a match
                    if let Ok(surface_texture) = surface_texture_result {
                        if let Err(err) = renderer.render(
                            &mut engine_state,
                            &surface_data,
                            surface_texture,
                            game_state.get_ui_container(),
                        ) {
                            log::error!("{err:?}");
                        }
                    }

                    engine_state.time_tracker.on_render_completed();

                    latest_surface_texture_result =
                        Some(surface_data.surface.get_current_texture());

                    engine_state.time_tracker.on_get_surface_completed();

                    profiling::finish_frame!();
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
                            renderer.resize_surface(&mut surface_data, *size);
                            on_window_resize(
                                GameContext {
                                    game_state: &mut game_state,
                                    engine_state: &mut engine_state,
                                    renderer: &mut renderer,
                                    surface_data: &mut surface_data,
                                    window: &mut window,
                                    elwt,
                                },
                                *size,
                            );
                        }
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        let new_inner_size = window.inner_size();
                        if new_inner_size.width > 0 && new_inner_size.height > 0 {
                            renderer.resize_surface(&mut surface_data, new_inner_size);
                            on_window_resize(
                                GameContext {
                                    game_state: &mut game_state,
                                    engine_state: &mut engine_state,
                                    renderer: &mut renderer,
                                    surface_data: &mut surface_data,
                                    window: &mut window,
                                    elwt,
                                },
                                new_inner_size,
                            );
                        }
                    }
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Moved(_) => {
                        monitor_refresh_rate = window
                            .current_monitor()
                            .and_then(|window| window.refresh_rate_millihertz())
                    }
                    _ => {}
                };

                // dbg!(sleep_start.is_some());
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
