use crate::game::*;
use crate::game_state::*;
use crate::logger::*;
use crate::renderer::*;
use crate::time::*;
use crate::ui_overlay::AudioSoundStats;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

const MAX_LOG_RATE: i64 = 24; // 24 logs per second

pub fn run(
    mut window: Window,
    event_loop: EventLoop<()>,
    mut game_state: GameState,
    renderer: Renderer,
) {
    let mut last_log_time: Option<Instant> = None;
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

                update_game_state(&mut game_state, &renderer.base, renderer.data.clone());

                let last_log_time_clone = last_log_time;
                let mut write_logs = || {
                    if let Err(err) = LOGGER.lock().unwrap().write_to_term() {
                        eprintln!("Error writing to terminal: {}", err);
                    }
                    last_log_time = Some(Instant::now());
                };

                match last_log_time_clone {
                    Some(last_log_time)
                        if last_log_time.elapsed()
                            > Duration::from_millis((1000.0 / MAX_LOG_RATE as f32) as u64) =>
                    {
                        write_logs()
                    }
                    None => write_logs(),
                    _ => {}
                }

                {
                    // sync UI
                    // TODO: move this into a function in game module?

                    let mut renderer_data_guard = renderer.data.lock().unwrap();

                    if let Some(frame_duration) = frame_duration {
                        renderer_data_guard.ui_overlay.send_message(
                            crate::ui_overlay::Message::FrameCompleted(frame_duration),
                        );
                        if let Some(gpu_timing_info) = renderer.process_profiler_frame() {
                            renderer_data_guard.ui_overlay.send_message(
                                crate::ui_overlay::Message::GpuFrameCompleted(gpu_timing_info),
                            );
                        }
                    }

                    let camera_position = game_state
                        .player_controller
                        .position(&game_state.physics_state);
                    let camera_view_direction = game_state.player_controller.view_direction;
                    renderer_data_guard.ui_overlay.send_message(
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

                            renderer_data_guard.ui_overlay.send_message(
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

                    let ui_state = renderer_data_guard.ui_overlay.get_state().clone();

                    renderer_data_guard.enable_soft_shadows = ui_state.enable_soft_shadows;
                    renderer_data_guard.soft_shadow_factor = ui_state.soft_shadow_factor;
                    renderer_data_guard.shadow_bias = ui_state.shadow_bias;
                    renderer_data_guard.enable_shadow_debug = ui_state.enable_shadow_debug;
                    renderer_data_guard.soft_shadow_grid_dims = ui_state.soft_shadow_grid_dims;
                    renderer_data_guard.draw_culling_frustum = ui_state.draw_culling_frustum;
                    renderer_data_guard.draw_point_light_culling_frusta =
                        ui_state.draw_point_light_culling_frusta;
                    renderer.set_vsync(ui_state.enable_vsync);
                    renderer
                        .set_culling_frustum_lock(&game_state, ui_state.culling_frustum_lock_mode);
                }

                #[cfg(target_arch = "wasm32")]
                {
                    let new_size = winit::dpi::PhysicalSize::new(
                        canvas_container.offset_width() as u32,
                        canvas_container.offset_height() as u32,
                    );
                    if window.inner_size() != new_size {
                        window.set_inner_size(new_size);
                    }
                }

                match renderer.render(&mut game_state, &window, control_flow) {
                    Ok(_) => {}
                    Err(err) => match err.downcast_ref::<wgpu::SurfaceError>() {
                        // Reconfigure the surface if lost
                        Some(wgpu::SurfaceError::Lost) => {
                            renderer.resize(window.inner_size(), window.scale_factor())
                        }
                        // The system is out of memory, we should probably quit
                        Some(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                        _ => logger_log(&format!("{:?}", err)),
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
                process_device_input(&mut game_state, &renderer, &event);
            }
            Event::WindowEvent {
                event, window_id, ..
            } if window_id == window.id() => {
                match &event {
                    WindowEvent::Resized(size) => {
                        if size.width > 0 && size.height > 0 {
                            renderer.resize(*size, window.scale_factor());
                        }
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        if new_inner_size.width > 0 && new_inner_size.height > 0 {
                            renderer.resize(**new_inner_size, window.scale_factor());
                        }
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => {}
                };

                renderer
                    .data
                    .lock()
                    .unwrap()
                    .ui_overlay
                    .handle_window_event(&window, &event);

                process_window_input(&mut game_state, &renderer, &event, &mut window);
            }
            _ => {}
        }
    });
}
