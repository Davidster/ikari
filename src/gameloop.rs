use std::time::{Duration, Instant};

use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use super::*;

const MAX_LOG_RATE: i64 = 24; // 24 logs per second

pub async fn run<'a>(
    mut window: Window,
    event_loop: EventLoop<()>,
    mut renderer_state: RendererState,
) {
    let mut last_log_time: Option<Instant> = None;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => {
                renderer_state.update(&window);
                renderer_state.logger.on_frame_completed();

                let last_log_time_clone = last_log_time;
                let mut write_logs = || {
                    renderer_state.logger.write_to_term();
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

                match renderer_state.render() {
                    Ok(_) => {
                        if !renderer_state.rendered_first_frame {
                            renderer_state.rendered_first_frame = true;
                            window.set_visible(true);
                        }
                    }
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => {
                        renderer_state.resize(renderer_state.current_window_size)
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        *control_flow = winit::event_loop::ControlFlow::Exit
                    }
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => renderer_state.logger.log(&format!("{:?}", e)),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            Event::DeviceEvent { event, .. } => {
                renderer_state.process_device_input(&event, &mut window);
            }
            Event::WindowEvent {
                event, window_id, ..
            } if window_id == window.id() => {
                renderer_state.process_window_input(&event, &mut window);
                match event {
                    WindowEvent::Resized(size) => {
                        renderer_state.resize(size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        renderer_state.resize(*new_inner_size);
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
                    } => *control_flow = winit::event_loop::ControlFlow::Exit,
                    _ => {}
                };
            }
            _ => {}
        }
    });
}
