use std::sync::Arc;

use crate::engine_state::EngineState;
use crate::renderer::*;
use crate::time::*;
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

    let handler = move |event: Event<()>, elwt: &EventLoopWindowTarget<()>| {
        web_canvas_manager.on_update(&event);

        match event {
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                engine_state.time_tracker.on_frame_started();
                profiling::finish_frame!();

                if !logged_start_time && engine_state.time_tracker.last_frame_times().is_some() {
                    log::debug!(
                        "Took {:?} from process startup till first frame",
                        application_start_time.elapsed()
                    );
                    logged_start_time = true;
                }

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

                if let Err(err) = renderer.render(
                    &mut engine_state,
                    &surface_data,
                    game_state.get_ui_container(),
                ) {
                    match err.downcast_ref::<wgpu::SurfaceError>() {
                        // Reconfigure the surface if lost
                        Some(wgpu::SurfaceError::Lost) => {
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
                        Some(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        _ => log::error!("{err:?}"),
                    }
                }

                engine_state.time_tracker.on_render_completed();
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::LoopExiting => {
                log::info!("gameloop LoopExiting");
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
