use std::borrow::Cow;

use iced::font::Family;
use iced::widget::Row;
use iced::{window, Command, Element, Font, Size};

use iced_wgpu::graphics::Viewport;
use iced_winit::{runtime, Clipboard};

use winit::event::{DeviceEvent, WindowEvent};
use winit::window::Window;

pub trait UiProgramEvents: runtime::Program {
    fn handle_window_event(&self, _window: &Window, _event: &WindowEvent) -> Vec<Self::Message> {
        vec![]
    }

    fn handle_device_event(&self, _window: &Window, _event: &DeviceEvent) -> Vec<Self::Message> {
        vec![]
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmptyUiOverlay;

// integrates iced into ikari
// based off of https://github.com/iced-rs/iced/tree/master/examples/integration_wgpu
pub struct IkariUiContainer<UiOverlay>
where
    UiOverlay: runtime::Program + UiProgramEvents + 'static,
{
    debug: runtime::Debug,
    renderer: iced::Renderer,
    viewport: Viewport,
    clipboard: Clipboard,
    program_container: runtime::program::State<UiOverlay>,
    pub cursor_position: winit::dpi::PhysicalPosition<f64>,
    modifiers: winit::keyboard::ModifiersState,
    last_cursor_icon: Option<winit::window::CursorIcon>,
    theme: UiOverlay::Theme,
    surface_format: wgpu::TextureFormat,
}

impl<UiOverlay> IkariUiContainer<UiOverlay>
where
    UiOverlay: runtime::Program<Renderer = iced::Renderer> + UiProgramEvents,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        state: UiOverlay,
        // TODO: this only works if the font matches the weight, stretch, style and monospacedness
        // of iced_core::Font::DEFAULT
        default_font_name: Option<&'static str>,
        load_fonts: Vec<&'static [u8]>,
        theme: UiOverlay::Theme,
    ) -> Self {
        let viewport = Viewport::with_physical_size(
            Size::new(window.inner_size().width, window.inner_size().height),
            window.scale_factor(),
        );

        let cursor_position = winit::dpi::PhysicalPosition::new(-1.0, -1.0);

        let settings = iced_wgpu::Settings {
            default_font: Font {
                family: default_font_name
                    .map(Family::Name)
                    .unwrap_or(Font::DEFAULT.family),
                ..Default::default()
            },
            ..Default::default()
        };

        let surface_format = surface_format.add_srgb_suffix();

        let mut debug = runtime::Debug::new();
        let wgpu_renderer = iced_wgpu::Renderer::new(
            iced_wgpu::Backend::new(device, queue, settings, surface_format),
            settings.default_font,
            settings.default_text_size,
        );

        let mut renderer = iced::Renderer::Wgpu(wgpu_renderer);

        {
            use iced_wgpu::core::text::Renderer;

            for font_bytes in load_fonts {
                renderer.load_font(Cow::from(font_bytes));
            }
        }

        let program_container = iced_winit::runtime::program::State::new(
            state,
            viewport.logical_size(),
            &mut renderer,
            &mut debug,
        );

        let modifiers = winit::keyboard::ModifiersState::default();
        let clipboard = Clipboard::connect(window);

        Self {
            debug,
            renderer,
            program_container,
            cursor_position,
            modifiers,
            viewport,
            clipboard,
            last_cursor_icon: None,
            theme,
            surface_format,
        }
    }

    pub fn resize(&mut self, framebuffer_size: winit::dpi::PhysicalSize<u32>, scale_factor: f64) {
        self.viewport = Viewport::with_physical_size(
            Size::new(framebuffer_size.width, framebuffer_size.height),
            scale_factor,
        );
    }

    pub fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = *position;
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers.state();
            }
            _ => {}
        }

        if let Some(event) = iced_winit::conversion::window_event(
            window::Id::MAIN,
            event.clone(),
            window.scale_factor(),
            self.modifiers,
        ) {
            self.program_container.queue_event(event);
        }

        let messages = self
            .program_container
            .program()
            .handle_window_event(window, event);
        for message in messages {
            self.program_container.queue_message(message);
        }
    }

    pub fn handle_device_event(&mut self, window: &Window, event: &DeviceEvent) {
        let messages = self
            .program_container
            .program()
            .handle_device_event(window, event);
        for message in messages {
            self.program_container.queue_message(message);
        }
    }
    #[profiling::function]
    pub fn update(&mut self, window: &Window) {
        if !self.program_container.is_queue_empty() {
            let _ = self.program_container.update(
                self.viewport.logical_size(),
                iced_winit::core::mouse::Cursor::Available(
                    iced_winit::conversion::cursor_position(
                        self.cursor_position,
                        self.viewport.scale_factor(),
                    ),
                ),
                &mut self.renderer,
                &self.theme,
                &iced_winit::core::renderer::Style {
                    text_color: iced::Color::WHITE,
                },
                &mut self.clipboard,
                &mut self.debug,
            );
        }

        let cursor_icon =
            iced_winit::conversion::mouse_interaction(self.program_container.mouse_interaction());
        if self.last_cursor_icon != Some(cursor_icon) {
            window.set_cursor_icon(cursor_icon);
            self.last_cursor_icon = Some(cursor_icon);
        }
    }

    pub(crate) fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
    ) {
        if let iced::Renderer::Wgpu(renderer) = &mut self.renderer {
            renderer.with_primitives(|backend, primitive| {
                backend.present(
                    device,
                    queue,
                    encoder,
                    None,
                    self.surface_format,
                    texture_view,
                    primitive,
                    &self.viewport,
                    &self.debug.overlay(),
                );
            });
        }
    }

    pub fn queue_message(&mut self, message: UiOverlay::Message) {
        self.program_container.queue_message(message);
    }

    pub fn get_state(&self) -> &UiOverlay {
        self.program_container.program()
    }
}

impl runtime::Program for EmptyUiOverlay {
    type Message = ();
    type Theme = iced::Theme;
    type Renderer = iced::Renderer;

    fn update(&mut self, _message: Self::Message) -> Command<Self::Message> {
        Command::none()
    }

    fn view(&self) -> Element<Self::Message, iced::Theme, iced::Renderer> {
        Row::new().into()
    }
}
