use std::time::Duration;

use iced::widget::Column;
use iced::Background;
use iced_wgpu::Renderer;
use iced_winit::widget::{Container, Row};
use iced_winit::{Command, Element, Length, Program};
use plotters::prelude::*;
use plotters::style::RED;
use plotters_iced::{Chart, ChartWidget, DrawingBackend};
use winit::{event::WindowEvent, window::Window};

use crate::logger::*;

const FRAME_TIME_HISTORY_SIZE: usize = 5000;

#[derive(Debug)]
pub struct IcedProgram {
    chart: FpsChart,
}

#[derive(Debug)]
pub enum Message {
    FrameCompleted(Duration),
    GpuFrameCompleted(Vec<GpuTimerScopeResultWrapper>),
}

pub struct ContainerStyle;

impl iced::widget::container::StyleSheet for ContainerStyle {
    type Style = iced_wgpu::Theme;

    fn appearance(&self, _: &Self::Style) -> iced::widget::container::Appearance {
        iced::widget::container::Appearance {
            background: Some(Background::Color(iced::Color::from_rgba(
                0.3, 0.3, 0.3, 0.6,
            ))),
            ..Default::default()
        }
    }
}

#[derive(Debug)]
struct FpsChart {
    recent_frame_times: Vec<Duration>,
    recent_gpu_frame_times: Vec<Vec<GpuTimerScopeResultWrapper>>,
}

impl Chart<Message> for FpsChart {
    type State = ();

    fn build_chart<DB: DrawingBackend>(
        &self,
        _state: &Self::State,
        mut builder: plotters_iced::ChartBuilder<DB>,
    ) {
        let result: Result<(), String> = (|| {
            let oldest_ft_age_secs = 5.0f32;
            let mut frame_times_with_ages = Vec::new();
            let mut acc = std::time::Duration::from_secs(0);
            for frame_time in self.recent_frame_times.iter().rev() {
                frame_times_with_ages.push((
                    -acc.as_secs_f32(),
                    (1.0 / frame_time.as_secs_f32()).round() as i32,
                ));
                acc += *frame_time;

                if acc.as_secs_f32() > oldest_ft_age_secs {
                    break;
                }
            }
            frame_times_with_ages.reverse();

            // round up to the nearest multiple of 30fps
            let roundup_factor = 30.0;
            let min_y_axis_height = 120;

            let max_y = frame_times_with_ages
                .iter()
                .map(|(_frame_age_secs, fps)| *fps)
                .reduce(|acc, fps| if acc > fps { acc } else { fps })
                .map(|max_fps| {
                    (((max_fps as f32 / roundup_factor).ceil() * roundup_factor) as i32)
                        .max(min_y_axis_height)
                })
                .unwrap_or(min_y_axis_height);

            let fps_grading_size = 30.0;

            let mut chart = builder
                .x_label_area_size(20)
                .y_label_area_size(52)
                .build_cartesian_2d(
                    -oldest_ft_age_secs..0.0,
                    (0..max_y).group_by(fps_grading_size as usize),
                )
                .map_err(|err| err.to_string())?;

            let mesh_line_style = ShapeStyle {
                color: RGBAColor(175, 175, 175, 1.0),
                filled: false,
                stroke_width: 1,
            };
            let axis_line_style = ShapeStyle {
                color: WHITE.to_rgba(),
                filled: false,
                stroke_width: 2,
            };
            let axis_labels_style = ("sans-serif", 18, &WHITE);

            chart
                .configure_mesh()
                .x_label_formatter(&|x| format!("{}s", x.abs().round() as i32))
                .y_label_formatter(&|y| format!("{} fps", y))
                .disable_x_mesh()
                .y_max_light_lines(1)
                .set_all_tick_mark_size(1)
                .x_labels(oldest_ft_age_secs as usize + 1)
                .y_labels(((max_y / fps_grading_size as i32) + 1).min(7) as usize)
                .light_line_style(mesh_line_style)
                .bold_line_style(mesh_line_style)
                .axis_style(axis_line_style)
                .x_label_style(axis_labels_style)
                .y_label_style(axis_labels_style)
                .draw()
                .map_err(|err| err.to_string())?;

            chart
                .configure_series_labels()
                .draw()
                .map_err(|err| err.to_string())?;

            chart
                .draw_series(plotters::series::LineSeries::new(
                    frame_times_with_ages,
                    RED.stroke_width(1),
                ))
                .map_err(|err| err.to_string())?;

            Ok(())
        })();

        if let Err(err) = result {
            logger_log(&format!("Error building fps chart: {:?}", err));
        }
    }
}

impl FpsChart {
    fn view(&self) -> iced::Element<Message> {
        ChartWidget::new(self)
            .width(iced::Length::Fixed(400.0))
            .height(iced::Length::Fixed(300.0))
            .into()
    }
}

// the iced ui
impl Program for IcedProgram {
    type Renderer = Renderer;
    type Message = Message;

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::FrameCompleted(frame_duration) => {
                self.chart.recent_frame_times.push(frame_duration);
                if self.chart.recent_frame_times.len() > FRAME_TIME_HISTORY_SIZE {
                    self.chart.recent_frame_times.remove(0);
                }
            }
            Message::GpuFrameCompleted(frames) => {
                self.chart.recent_gpu_frame_times.push(frames);
                if self.chart.recent_gpu_frame_times.len() > FRAME_TIME_HISTORY_SIZE {
                    self.chart.recent_gpu_frame_times.remove(0);
                }
            }
        }

        Command::none()
    }

    fn view(&self) -> Element<Message, Renderer> {
        if self.chart.recent_frame_times.is_empty() {
            return Row::new().into();
        }

        let (avg_frame_time_millis, avg_gpu_frame_time_millis) = {
            let alpha = 0.01;

            let mut frame_times_iterator = self
                .chart
                .recent_frame_times
                .iter()
                .map(|frame_time| frame_time.as_nanos() as f64 / 1_000_000.0);
            let mut cpu = frame_times_iterator.next().unwrap(); // checked that length isn't 0
            for frame_time in frame_times_iterator {
                cpu = (1.0 - alpha) * cpu + (alpha * frame_time);
            }

            let mut gpu_frame_times_iterator = self
                .chart
                .recent_gpu_frame_times
                .iter()
                .map(collect_frame_time_ms);
            let mut gpu = gpu_frame_times_iterator.next().unwrap_or(0.0);
            for frame_time in gpu_frame_times_iterator {
                gpu = (1.0 - alpha) * gpu + (alpha * frame_time);
            }

            (cpu, gpu)
        };

        let framerate_msg = String::from(&format!(
            "Frametime: {:.2}ms ({:.2}fps), GPU: {:.2}ms",
            avg_frame_time_millis,
            1_000.0 / avg_frame_time_millis,
            avg_gpu_frame_time_millis
        ));

        let container_style = Box::new(ContainerStyle {});
        Row::new()
            .width(Length::Shrink)
            .height(Length::Shrink)
            .padding(8)
            .push(
                Container::new(
                    Column::new()
                        .width(Length::Shrink)
                        .height(Length::Shrink)
                        .push(iced_winit::widget::text(framerate_msg.as_str()))
                        .push(Container::new(self.chart.view()).padding([16, 20, 16, 0])), // top, right, bottom, left
                )
                .padding(8)
                .style(iced::theme::Container::Custom(container_style)),
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
            chart: FpsChart {
                recent_frame_times: vec![],
                recent_gpu_frame_times: vec![],
            },
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

    pub fn send_message(&mut self, message: Message) {
        self.program_container.queue_message(message);
    }
}

pub struct GpuTimerScopeResultWrapper(pub wgpu_profiler::GpuTimerScopeResult);

impl std::ops::Deref for GpuTimerScopeResultWrapper {
    type Target = wgpu_profiler::GpuTimerScopeResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for GpuTimerScopeResultWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "(TODO implement better formatter for GpuTimerScopeResultWrapper) {:?}",
            self.label
        )
    }
}

fn collect_frame_time_ms(frame_times: &Vec<GpuTimerScopeResultWrapper>) -> f64 {
    let mut result = 0.0;
    for frame_time in frame_times {
        result += (frame_time.time.end - frame_time.time.start) * 1000.0;
    }
    result
}
