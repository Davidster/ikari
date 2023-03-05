use std::time::{Duration, Instant};

use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use super::*;

const MAX_LOG_RATE: i64 = 24; // 24 logs per second

pub fn run(
    mut window: Window,
    event_loop: EventLoop<()>,
    mut game_state: GameState,
    mut renderer_state: RendererState,
) {
    let mut last_log_time: Option<Instant> = None;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => {
                game_state.on_frame_started();
                profiling::finish_frame!();
                LOGGER.lock().unwrap().on_frame_completed();

                update_game_state(&mut game_state, &renderer_state.base.lock().unwrap());

                /* let last_log_time_clone = last_log_time;
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
                } */

                match renderer_state.render(&mut game_state) {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => {
                        renderer_state.resize(renderer_state.base.lock().unwrap().window_size)
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
                process_device_input(&mut game_state, &event);
            }
            Event::WindowEvent {
                event, window_id, ..
            } if window_id == window.id() => {
                process_window_input(&mut game_state, &mut renderer_state, &event, &mut window);
                match event {
                    WindowEvent::Resized(size) => {
                        if size.width > 0 && size.height > 0 {
                            renderer_state.resize(size);
                        }
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        if new_inner_size.width > 0 && new_inner_size.height > 0 {
                            renderer_state.resize(*new_inner_size);
                        }
                    }
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    _ => {}
                };
            }
            _ => {}
        }
    });
}
