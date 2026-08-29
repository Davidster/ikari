use std::borrow::Cow;

use iced::widget::Row;
use iced::{Element, Font, Pixels, Size, Task};

use iced_wgpu::graphics::{Shell, Viewport};
use iced_wgpu::{Engine, Renderer as WgpuRenderer};
use iced_winit::core::{mouse, renderer as core_renderer, Event as IcedEvent};
use iced_winit::runtime::user_interface::{self, UserInterface};
use iced_winit::{conversion, Clipboard};

use winit::event::{DeviceEvent, WindowEvent};
use winit::window::Window;

/// ikari's stand-in for iced's `runtime::Program`, which was removed in iced 0.13.
/// iced is now driven through `UserInterface` directly, so the engine owns the
/// trait that ui overlays implement.
pub trait UiProgram {
    type Message: std::fmt::Debug + Send;
    type Theme;

    fn update(&mut self, message: Self::Message) -> Task<Self::Message>;

    fn view(&self) -> Element<'_, Self::Message, Self::Theme, iced::Renderer>;
}

pub trait UiProgramEvents: UiProgram {
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
// based off of https://github.com/iced-rs/iced/tree/0.14.0/examples/integration
pub struct IkariUiContainer<UiOverlay>
where
    UiOverlay: UiProgram + UiProgramEvents + 'static,
{
    // `iced::Renderer` is an enum over the wgpu and tiny-skia renderers. ikari always
    // uses the wgpu one, but holding the enum keeps ikari compatible with third-party
    // iced widgets, which are written against `iced::Renderer` rather than
    // `iced_wgpu::Renderer`.
    renderer: iced::Renderer,
    viewport: Viewport,
    clipboard: Clipboard,
    state: UiOverlay,
    cache: user_interface::Cache,
    queued_events: Vec<IcedEvent>,
    queued_messages: Vec<UiOverlay::Message>,
    pub cursor_position: winit::dpi::PhysicalPosition<f64>,
    modifiers: winit::keyboard::ModifiersState,
    last_cursor_icon: Option<winit::window::CursorIcon>,
    theme: UiOverlay::Theme,
    surface_format: wgpu::TextureFormat,
}

impl<UiOverlay> IkariUiContainer<UiOverlay>
where
    UiOverlay: UiProgram + UiProgramEvents,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        window: &std::sync::Arc<Window>,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        state: UiOverlay,
        default_font: Option<Font>,
        load_fonts: Vec<&'static [u8]>,
        theme: UiOverlay::Theme,
    ) -> Self {
        let viewport = Viewport::with_physical_size(
            Size::new(window.inner_size().width, window.inner_size().height),
            window.scale_factor() as f32,
        );

        let cursor_position = winit::dpi::PhysicalPosition::new(-1.0, -1.0);

        let default_font = default_font.unwrap_or(Font::DEFAULT);
        let surface_format = surface_format.add_srgb_suffix();

        // iced 0.14 owns its own encoder and submits directly through the engine's
        // queue, so it needs owned handles rather than the caller's encoder.
        let engine = Engine::new(
            adapter,
            device.clone(),
            queue.clone(),
            surface_format,
            None,
            Shell::headless(),
        );

        let mut renderer =
            iced::Renderer::Primary(WgpuRenderer::new(engine, default_font, Pixels::from(16)));

        // iced 0.14 loads fonts into a process-global font system rather than through
        // the renderer.
        for font_bytes in load_fonts {
            iced_wgpu::graphics::text::font_system()
                .write()
                .expect("Failed to acquire the global iced font system")
                .load_font(Cow::from(font_bytes));
        }

        let clipboard = Clipboard::connect(window.clone());

        Self {
            renderer,
            viewport,
            clipboard,
            state,
            cache: user_interface::Cache::default(),
            queued_events: Vec::new(),
            queued_messages: Vec::new(),
            cursor_position,
            modifiers: winit::keyboard::ModifiersState::default(),
            last_cursor_icon: None,
            theme,
            surface_format,
        }
    }

    pub fn resize(&mut self, framebuffer_size: winit::dpi::PhysicalSize<u32>, scale_factor: f64) {
        self.viewport = Viewport::with_physical_size(
            Size::new(framebuffer_size.width, framebuffer_size.height),
            scale_factor as f32,
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

        if let Some(event) =
            conversion::window_event(event.clone(), window.scale_factor() as f32, self.modifiers)
        {
            self.queued_events.push(event);
        }

        self.queued_messages
            .extend(self.state.handle_window_event(window, event));
    }

    pub fn handle_device_event(&mut self, window: &Window, event: &DeviceEvent) {
        self.queued_messages
            .extend(self.state.handle_device_event(window, event));
    }

    #[profiling::function]
    pub fn update(&mut self, window: &Window) {
        let cursor = mouse::Cursor::Available(conversion::cursor_position(
            self.cursor_position,
            self.viewport.scale_factor(),
        ));

        let events = std::mem::take(&mut self.queued_events);
        let mut messages = std::mem::take(&mut self.queued_messages);

        // iced 0.14 rebuilds the UserInterface every frame from a persisted cache,
        // rather than holding a long-lived program::State like 0.12 did.
        {
            let mut interface = UserInterface::build(
                self.state.view(),
                self.viewport.logical_size(),
                std::mem::take(&mut self.cache),
                &mut self.renderer,
            );

            let _ = interface.update(
                &events,
                cursor,
                &mut self.renderer,
                &mut self.clipboard,
                &mut messages,
            );

            interface.draw(
                &mut self.renderer,
                &self.theme,
                &core_renderer::Style {
                    text_color: iced::Color::WHITE,
                },
                cursor,
            );

            self.cache = interface.into_cache();
        }

        for message in messages {
            let _ = self.state.update(message);
        }

        // TODO: restore cursor-icon syncing. In iced 0.12 `UserInterface::draw`
        // returned the current `mouse::Interaction`; in 0.14 it returns nothing and
        // there's no obvious replacement for a manual integration, so the cursor no
        // longer changes shape over UI elements.
        let _ = (&mut self.last_cursor_icon, window);
    }

    pub(crate) fn render(&mut self, texture_view: &wgpu::TextureView) {
        // NOTE: iced submits its own command buffer here, so the caller must have
        // already submitted the scene's encoder for the draw order to be correct.
        if let iced::Renderer::Primary(renderer) = &mut self.renderer {
            let _submission =
                renderer.present(None, self.surface_format, texture_view, &self.viewport);
        }
    }

    pub fn queue_message(&mut self, message: UiOverlay::Message) {
        self.queued_messages.push(message);
    }

    pub fn get_state(&self) -> &UiOverlay {
        &self.state
    }
}

impl UiProgram for EmptyUiOverlay {
    type Message = ();
    type Theme = iced::Theme;

    fn update(&mut self, _message: Self::Message) -> Task<Self::Message> {
        Task::none()
    }

    fn view(&self) -> Element<'_, Self::Message, iced::Theme, iced::Renderer> {
        Row::new().into()
    }
}

impl UiProgramEvents for EmptyUiOverlay {}
