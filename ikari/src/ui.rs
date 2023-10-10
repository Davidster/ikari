use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use glam::Vec3;
use iced::alignment::Horizontal;
use iced::font::Family;
use iced::widget::{
    canvas, checkbox, container, radio, scrollable, slider, text, Button, Column, Container, Row,
    Text,
};
use iced::{mouse, Background, Command, Element, Font, Length, Rectangle, Size, Theme};
use iced_aw::{floating_element, Modal};
use iced_winit::{runtime, Clipboard, Viewport};
use plotters::prelude::*;
use plotters::style::RED;
use plotters_iced::{Chart, ChartWidget, DrawingBackend};
use winit::{event::WindowEvent, window::Window};

use crate::file_loader::GameFilePath;
use crate::math::*;
use crate::player_controller::*;
use crate::profile_dump::*;
use crate::renderer::CullingFrustumLock;
use crate::time::*;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmptyUiOverlay;

// TODO: move out of ui module (but keep them in ikari?)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum CullingFrustumLockMode {
    Full,
    FocalPoint,
    #[default]
    None,
}

// TODO: move out of ui module (but keep them in ikari?)
impl CullingFrustumLockMode {
    pub const ALL: [CullingFrustumLockMode; 3] = [
        CullingFrustumLockMode::None,
        CullingFrustumLockMode::Full,
        CullingFrustumLockMode::FocalPoint,
    ];
}

// TODO: move out of ui module (but keep them in ikari?)
impl From<CullingFrustumLock> for CullingFrustumLockMode {
    fn from(value: CullingFrustumLock) -> Self {
        match value {
            CullingFrustumLock::Full(_) => CullingFrustumLockMode::Full,
            CullingFrustumLock::FocalPoint(_) => CullingFrustumLockMode::FocalPoint,
            CullingFrustumLock::None => CullingFrustumLockMode::None,
        }
    }
}

// TODO: move out of ui module (but keep them in ikari?)
impl std::fmt::Display for CullingFrustumLockMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CullingFrustumLockMode::Full => "On",
                CullingFrustumLockMode::FocalPoint => "Only FocalPoint",
                CullingFrustumLockMode::None => "Off",
            }
        )
    }
}

// TODO: move out of ui module (but keep them in ikari?)
pub struct GpuTimerScopeResultWrapper(pub wgpu_profiler::GpuTimerScopeResult);

// TODO: move out of ui module (but keep them in ikari?)
impl std::ops::Deref for GpuTimerScopeResultWrapper {
    type Target = wgpu_profiler::GpuTimerScopeResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// TODO: move out of ui module (but keep them in ikari?)
impl std::fmt::Debug for GpuTimerScopeResultWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "(TODO implement better formatter for type) {:?} -> {:?}",
            self.label, self.time
        )
    }
}

// TODO: move out of ui module (but keep them in ikari?)
impl Clone for GpuTimerScopeResultWrapper {
    fn clone(&self) -> Self {
        Self(wgpu_profiler::GpuTimerScopeResult {
            label: self.label.clone(),
            time: self.time.clone(),
            nested_scopes: clone_nested_scopes(&self.nested_scopes),
            pid: self.pid,
            tid: self.tid,
        })
    }
}

// TODO: move out of ui module (but keep them in ikari?)
fn clone_nested_scopes(
    nested_scopes: &[wgpu_profiler::GpuTimerScopeResult],
) -> Vec<wgpu_profiler::GpuTimerScopeResult> {
    nested_scopes
        .iter()
        .map(|nested_scope| wgpu_profiler::GpuTimerScopeResult {
            label: nested_scope.label.clone(),
            time: nested_scope.time.clone(),
            nested_scopes: clone_nested_scopes(&nested_scope.nested_scopes),
            pid: nested_scope.pid,
            tid: nested_scope.tid,
        })
        .collect()
}

// TODO: move out of ui module (but keep them in ikari?)
pub fn collect_frame_time_ms(frame_times: &Vec<GpuTimerScopeResultWrapper>) -> f64 {
    let mut result = 0.0;
    for frame_time in frame_times {
        result += (frame_time.time.end - frame_time.time.start) * 1000.0;
    }
    result
}

// integrates iced into ikari
// based off of https://github.com/iced-rs/iced/tree/master/examples/integration_wgpu
pub struct IkariUiContainer<UiOverlay>
where
    UiOverlay: runtime::Program + 'static,
{
    debug: runtime::Debug,
    renderer: iced::Renderer,
    viewport: Viewport,
    clipboard: Clipboard,
    program_container: runtime::program::State<UiOverlay>,
    pub cursor_position: winit::dpi::PhysicalPosition<f64>,
    modifiers: winit::event::ModifiersState,
    last_cursor_icon: Option<winit::window::CursorIcon>,
}

// pub trait IkariUiProgram: runtime::Program {
//     fn update()
// }

impl<UiOverlay> IkariUiContainer<UiOverlay>
where
    UiOverlay: runtime::Program<Renderer = iced::Renderer>,
{
    pub fn new(
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_texture_format: wgpu::TextureFormat,
        state: UiOverlay,
        default_font: Option<(&'static str, &'static [u8])>,
    ) -> Self {
        let viewport = Viewport::with_physical_size(
            Size::new(window.inner_size().width, window.inner_size().height),
            window.scale_factor(),
        );

        let cursor_position = winit::dpi::PhysicalPosition::new(-1.0, -1.0);

        let mut debug = runtime::Debug::new();
        let wgpu_renderer = iced_wgpu::Renderer::new(iced_wgpu::Backend::new(
            device,
            queue,
            iced_wgpu::Settings {
                default_font: Font {
                    family: default_font
                        .map(|(font_name, _)| Family::Name(font_name))
                        .unwrap_or(Font::DEFAULT.family),
                    ..Default::default()
                },
                ..Default::default()
            },
            output_texture_format,
        ));

        let mut renderer = iced::Renderer::Wgpu(wgpu_renderer);

        {
            use iced_wgpu::core::text::Renderer;

            if let Some((_, font_bytes)) = default_font {
                renderer.load_font(Cow::from(font_bytes));
            }
            // TODO: this should be supplied by the user. ikari shouldn't depend on iced_aw.
            renderer.load_font(Cow::from(iced_aw::graphics::icons::ICON_FONT_BYTES));
        }

        let program_container =
            runtime::program::State::new(state, viewport.logical_size(), &mut renderer, &mut debug);

        let modifiers = winit::event::ModifiersState::default();
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

    #[profiling::function]
    pub fn update(&mut self, window: &Window, control_flow: &mut winit::event_loop::ControlFlow) {
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
                &Theme::Dark,
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
    type Renderer = iced::Renderer;
    type Message = ();

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        Command::none()
    }

    fn view(&self) -> Element<Self::Message, iced::Renderer> {
        Row::new().into()
    }
}
