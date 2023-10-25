use crate::engine_state::EngineState;
use crate::renderer::*;
use crate::time::*;
use crate::ui::IkariUiContainer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
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
    pub window: &'a mut winit::window::Window,
    pub control_flow: &'a mut winit::event_loop::ControlFlow,
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
    mut window: Window,
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

    #[cfg(target_arch = "wasm32")]
    let canvas_container;
    #[cfg(target_arch = "wasm32")]
    {
        let dom_window = web_sys::window().unwrap();
        let document = dom_window.document().unwrap();
        canvas_container = document
            .get_element_by_id("canvas_container")
            .unwrap()
            .dyn_into::<web_sys::HtmlElement>()
            .unwrap();
    }

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::RedrawRequested(_) => {
                if !logged_start_time && engine_state.time_tracker.is_some() {
                    log::debug!(
                        "Took {:?} from process startup till first frame",
                        application_start_time.elapsed()
                    );
                    logged_start_time = true;
                }

                engine_state.on_frame_started();
                profiling::finish_frame!();

                on_update(GameContext {
                    game_state: &mut game_state,
                    engine_state: &mut engine_state,
                    renderer: &mut renderer,
                    surface_data: &mut surface_data,
                    window: &mut window,
                    control_flow,
                });

                #[cfg(target_arch = "wasm32")]
                {
                    let new_size = winit::dpi::PhysicalSize::new(
                        (canvas_container.offset_width() as f64 * window.scale_factor()) as u32,
                        (canvas_container.offset_height() as f64 * window.scale_factor()) as u32,
                    );
                    if window.inner_size() != new_size {
                        window.set_inner_size(new_size);
                    }
                }

                match renderer.render(
                    &mut engine_state,
                    &surface_data,
                    game_state.get_ui_container(),
                ) {
                    Ok(_) => {}
                    Err(err) => match err.downcast_ref::<wgpu::SurfaceError>() {
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
                                    control_flow,
                                },
                                size,
                            );
                        }
                        // The system is out of memory, we should probably quit
                        Some(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                        _ => log::error!("{err:?}"),
                    },
                }
            }
            Event::LoopDestroyed => {
                #[cfg(target_arch = "wasm32")]
                {
                    let dom_window = web_sys::window().unwrap();
                    let document = dom_window.document().unwrap();
                    let canvas_container = document
                        .get_element_by_id("canvas_container")
                        .unwrap()
                        .dyn_into::<web_sys::HtmlElement>()
                        .unwrap();
                    canvas_container.remove();
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            Event::DeviceEvent { event, .. } => {
                on_device_event(
                    GameContext {
                        game_state: &mut game_state,
                        engine_state: &mut engine_state,
                        renderer: &mut renderer,
                        surface_data: &mut surface_data,
                        window: &mut window,
                        control_flow,
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
                                    control_flow,
                                },
                                *size,
                            );
                        }
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        if new_inner_size.width > 0 && new_inner_size.height > 0 {
                            renderer.resize_surface(&mut surface_data, **new_inner_size);
                            on_window_resize(
                                GameContext {
                                    game_state: &mut game_state,
                                    engine_state: &mut engine_state,
                                    renderer: &mut renderer,
                                    surface_data: &mut surface_data,
                                    window: &mut window,
                                    control_flow,
                                },
                                **new_inner_size,
                            );
                        }
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => {}
                };

                on_window_event(
                    GameContext {
                        game_state: &mut game_state,
                        engine_state: &mut engine_state,
                        renderer: &mut renderer,
                        surface_data: &mut surface_data,
                        window: &mut window,
                        control_flow,
                    },
                    &event,
                );
            }
            _ => {}
        }
    });
}
