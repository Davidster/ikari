use crate::game::*;
use crate::game_state::*;
use crate::logger::*;
use crate::renderer::*;

use std::time::{Duration, Instant};

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

                    renderer_data_guard.enable_soft_shadows = renderer_data_guard
                        .ui_overlay
                        .get_state()
                        .enable_soft_shadows;
                    renderer_data_guard.soft_shadow_factor = renderer_data_guard
                        .ui_overlay
                        .get_state()
                        .soft_shadow_factor;
                    renderer_data_guard.shadow_bias =
                        renderer_data_guard.ui_overlay.get_state().shadow_bias;
                    renderer_data_guard.enable_shadow_debug = renderer_data_guard
                        .ui_overlay
                        .get_state()
                        .enable_shadow_debug;
                    renderer_data_guard.soft_shadow_grid_dims = renderer_data_guard
                        .ui_overlay
                        .get_state()
                        .soft_shadow_grid_dims;
                    renderer_data_guard.draw_culling_frustum = renderer_data_guard
                        .ui_overlay
                        .get_state()
                        .draw_culling_frustum;
                    renderer_data_guard.draw_point_light_culling_frusta = renderer_data_guard
                        .ui_overlay
                        .get_state()
                        .draw_point_light_culling_frusta;

                    renderer.set_culling_frustum_lock(
                        &game_state,
                        renderer_data_guard
                            .ui_overlay
                            .get_state()
                            .culling_frustum_lock_mode,
                    )
                }

                match renderer.render(&mut game_state, &window, control_flow) {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => {
                        renderer.resize(window.inner_size(), window.scale_factor())
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => logger_log(&format!("{:?}", e)),
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
