use crate::game::*;
use crate::game_state::*;
use crate::renderer::*;
use crate::time::*;
use crate::ui_overlay::AudioSoundStats;
use crate::ui_overlay::IkariUiOverlay;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub fn resize_window(
    renderer: &mut Renderer,
    ui_overlay: &mut IkariUiOverlay,
    surface_data: &mut SurfaceData,
    window: &winit::window::Window,
    new_size: (u32, u32),
) {
    renderer.resize_surface(new_size, surface_data);
    renderer.resize(new_size);
    ui_overlay.resize(new_size, window.scale_factor());
}

pub fn run(
    window: Window,
    event_loop: EventLoop<()>,
    mut game_state: GameState,
    mut renderer: Renderer,
    mut surface_data: SurfaceData,
    application_start_time: Instant,
) {
    let mut logged_start_time = false;
    let mut last_frame_start_time: Option<Instant> = None;

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
                game_state.on_frame_started();
                profiling::finish_frame!();
                let frame_duration = last_frame_start_time.map(|time| time.elapsed());
                last_frame_start_time = Some(Instant::now());

                update_game_state(&mut game_state, &mut renderer, &surface_data);

                {
                    // sync UI
                    // TODO: move this into a function in game module?

                    let mut renderer_data_guard = renderer.data.lock().unwrap();

                    if let Some(frame_duration) = frame_duration {
                        if !logged_start_time {
                            log::debug!(
                                "Took {:?} from process startup till first frame",
                                application_start_time.elapsed()
                            );
                            logged_start_time = true;
                        }

                        game_state.ui_overlay.send_message(
                            crate::ui_overlay::Message::FrameCompleted(frame_duration),
                        );
                        if let Some(gpu_timing_info) = renderer.process_profiler_frame() {
                            game_state.ui_overlay.send_message(
                                crate::ui_overlay::Message::GpuFrameCompleted(gpu_timing_info),
                            );
                        }
                    }

                    let camera_position = game_state
                        .player_controller
                        .position(&game_state.physics_state);
                    let camera_view_direction = game_state.player_controller.view_direction;
                    game_state.ui_overlay.send_message(
                        crate::ui_overlay::Message::CameraPoseChanged((
                            camera_position,
                            camera_view_direction,
                        )),
                    );

                    {
                        let audio_manager_guard = game_state.audio_manager.lock().unwrap();
                        for sound_index in audio_manager_guard.sound_indices() {
                            let file_path = audio_manager_guard
                                .get_sound_file_path(sound_index)
                                .unwrap();
                            let length_seconds = audio_manager_guard
                                .get_sound_length_seconds(sound_index)
                                .unwrap();
                            let pos_seconds = audio_manager_guard
                                .get_sound_pos_seconds(sound_index)
                                .unwrap();
                            let buffered_to_pos_seconds = audio_manager_guard
                                .get_sound_buffered_to_pos_seconds(sound_index)
                                .unwrap();

                            game_state.ui_overlay.send_message(
                                crate::ui_overlay::Message::AudioSoundStatsChanged((
                                    file_path.clone(),
                                    AudioSoundStats {
                                        length_seconds,
                                        pos_seconds,
                                        buffered_to_pos_seconds,
                                    },
                                )),
                            );
                        }
                    }

                    let ui_state = game_state.ui_overlay.get_state().clone();

                    renderer_data_guard.enable_soft_shadows = ui_state.enable_soft_shadows;
                    renderer_data_guard.soft_shadow_factor = ui_state.soft_shadow_factor;
                    renderer_data_guard.shadow_bias = ui_state.shadow_bias;
                    renderer_data_guard.enable_shadow_debug = ui_state.enable_shadow_debug;
                    renderer_data_guard.soft_shadow_grid_dims = ui_state.soft_shadow_grid_dims;
                    renderer_data_guard.draw_culling_frustum = ui_state.draw_culling_frustum;
                    renderer_data_guard.draw_point_light_culling_frusta =
                        ui_state.draw_point_light_culling_frusta;
                    renderer.set_vsync(ui_state.enable_vsync, &mut surface_data);

                    renderer.set_culling_frustum_lock(
                        &game_state,
                        (
                            surface_data.surface_config.width,
                            surface_data.surface_config.height,
                        ),
                        ui_state.culling_frustum_lock_mode,
                    );
                }

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

                game_state.ui_overlay.update(&window, control_flow);

                match renderer.render(&mut game_state, &surface_data) {
                    Ok(_) => {}
                    Err(err) => match err.downcast_ref::<wgpu::SurfaceError>() {
                        // Reconfigure the surface if lost
                        Some(wgpu::SurfaceError::Lost) => {
                            resize_window(
                                &mut renderer,
                                &mut game_state.ui_overlay,
                                &mut surface_data,
                                &window,
                                window.inner_size().into(),
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
                process_device_input(&mut game_state, &event);
            }
            Event::WindowEvent {
                event, window_id, ..
            } if window_id == window.id() => {
                match &event {
                    WindowEvent::Resized(size) => {
                        if size.width > 0 && size.height > 0 {
                            resize_window(
                                &mut renderer,
                                &mut game_state.ui_overlay,
                                &mut surface_data,
                                &window,
                                (*size).into(),
                            );
                        }
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        if new_inner_size.width > 0 && new_inner_size.height > 0 {
                            resize_window(
                                &mut renderer,
                                &mut game_state.ui_overlay,
                                &mut surface_data,
                                &window,
                                (**new_inner_size).into(),
                            );
                        }
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => {}
                };

                game_state.ui_overlay.handle_window_event(&window, &event);

                process_window_input(
                    &mut game_state,
                    &mut renderer,
                    &mut surface_data,
                    &event,
                    &window,
                );
            }
            _ => {}
        }
    });
}
