use std::time::Duration;

use iced_wgpu::Renderer;
use iced_winit::widget::{Container, Row};
use iced_winit::{Command, Element, Length, Program};
use winit::{event::WindowEvent, window::Window};

const FRAME_TIME_HISTORY_SIZE: usize = 5000;

#[derive(Debug)]
pub struct UiState {
    recent_frame_times: Vec<Duration>,
}

#[derive(Debug)]
pub struct IcedProgram {
    state: UiState,
}

#[derive(Debug, Clone)]
pub enum Message {
    FrameCompleted(Duration),
}

impl UiState {
    pub fn new() -> Self {
        Self {
            recent_frame_times: vec![],
        }
    }
}

impl Default for UiState {
    fn default() -> Self {
        Self::new()
    }
}

// the iced ui
impl Program for IcedProgram {
    type Renderer = Renderer;
    type Message = Message;

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::FrameCompleted(frame_duration) => {
                self.state.recent_frame_times.push(frame_duration);
                if self.state.recent_frame_times.len() > FRAME_TIME_HISTORY_SIZE {
                    self.state.recent_frame_times.remove(0);
                }
            }
        }

        Command::none()
    }

    fn view(&self) -> Element<Message, Renderer> {
        if self.state.recent_frame_times.is_empty() {
            return Row::new().into();
        }

        let avg_frame_time_millis = {
            let alpha = 0.1;
            let mut frame_times_iterator = self
                .state
                .recent_frame_times
                .iter()
                .map(|frame_time| frame_time.as_nanos() as f64 / 1_000_000.0);
            let mut res = frame_times_iterator.next().unwrap(); // checked that length isn't 0
            for frame_time in frame_times_iterator {
                res = (1.0 - alpha) * res + (alpha * frame_time);
            }
            res
        };

        let framerate_msg = String::from(&format!(
            "Frametime: {:.2}ms ({:.2}fps)",
            avg_frame_time_millis,
            1_000.0 / avg_frame_time_millis
        ));
        Row::new()
            .width(Length::Shrink)
            .height(Length::Shrink)
            .padding(5)
            .push(
                Container::new(
                    Row::new()
                        .width(Length::Shrink)
                        .height(Length::Shrink)
                        .push(iced_winit::widget::text(framerate_msg.as_str())),
                )
                .padding(5)
                .style(iced::theme::Container::Box),
            )
            .into()
    }
}

// integrates iced into ikari
// based off of https://github.com/iced-rs/iced/tree/master/examples/integration_wgpu
pub struct UiOverlay {
    debug: iced_winit::Debug,
    renderer: iced_wgpu::Renderer,
    staging_belt: wgpu::util::StagingBelt,
    viewport: iced_wgpu::Viewport,
    clipboard: iced_winit::clipboard::Clipboard,
    program_container: iced_winit::program::State<IcedProgram>,
    cursor_position: winit::dpi::PhysicalPosition<f64>,
    modifiers: winit::event::ModifiersState,
}

impl UiOverlay {
    pub fn new(
        window: &Window,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let viewport = iced_wgpu::Viewport::with_physical_size(
            iced::Size::new(window.inner_size().width, window.inner_size().height),
            window.scale_factor(),
        );

        let staging_belt = wgpu::util::StagingBelt::new(5 * 1024);

        let state = IcedProgram {
            state: Default::default(),
        };

        let mut debug = iced_winit::Debug::new();
        let mut renderer = iced_wgpu::Renderer::new(iced_wgpu::Backend::new(
            device,
            iced_wgpu::Settings::default(),
            surface_format,
        ));

        let program_container = iced_winit::program::State::new(
            state,
            viewport.logical_size(),
            &mut renderer,
            &mut debug,
        );

        let cursor_position = winit::dpi::PhysicalPosition::new(-1.0, -1.0);
        let modifiers = winit::event::ModifiersState::default();
        let clipboard = iced_winit::clipboard::Clipboard::connect(window);

        Self {
            debug,
            renderer,
            staging_belt,
            program_container,
            cursor_position,
            modifiers,
            viewport,
            clipboard,
        }
    }

    pub fn resize(&mut self, window_size: winit::dpi::PhysicalSize<u32>, scale_factor: f64) {
        self.viewport = iced_winit::Viewport::with_physical_size(
            iced::Size::new(window_size.width, window_size.height),
            scale_factor,
        );
    }

    pub fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = *position;
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = *new_modifiers;
            }
            _ => {}
        }

        if let Some(event) =
            iced_winit::conversion::window_event(event, window.scale_factor(), self.modifiers)
        {
            self.program_container.queue_event(event);
        }
    }

    pub fn update(&mut self, window: &Window) {
        if !self.program_container.is_queue_empty() {
            let _ = self.program_container.update(
                self.viewport.logical_size(),
                iced_winit::conversion::cursor_position(
                    self.cursor_position,
                    self.viewport.scale_factor(),
                ),
                &mut self.renderer,
                &iced_wgpu::Theme::Dark,
                &iced_winit::renderer::Style {
                    text_color: iced::Color::WHITE,
                },
                &mut self.clipboard,
                &mut self.debug,
            );
        }

        window.set_cursor_icon(iced_winit::conversion::mouse_interaction(
            self.program_container.mouse_interaction(),
        ));
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
    ) {
        self.staging_belt.recall();

        self.renderer.with_primitives(|backend, primitive| {
            backend.present(
                device,
                &mut self.staging_belt,
                encoder,
                texture_view,
                primitive,
                &self.viewport,
                &self.debug.overlay(),
            );
        });

        self.staging_belt.finish();
    }

    pub fn get_state(&self) -> &UiState {
        &self.program_container.program().state
    }

    pub fn send_message(&mut self, message: Message) {
        self.program_container.queue_message(message);
    }
}
