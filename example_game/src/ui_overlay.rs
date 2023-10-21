use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use glam::Vec3;
use iced::alignment::Horizontal;

use iced::widget::{
    canvas, checkbox, container, radio, scrollable, slider, text, Button, Column, Container, Row,
    Text,
};
use iced::{mouse, Background, Command, Element, Length, Rectangle, Theme};
use iced_aw::{floating_element, Modal};
use iced_winit::runtime;
use ikari::file_loader::GameFilePath;
use ikari::math::rad_to_deg;
use ikari::player_controller::ControlledViewDirection;
use ikari::profile_dump::can_generate_profile_dump;
use ikari::profile_dump::generate_profile_dump;
use ikari::profile_dump::PendingPerfDump;
use ikari::time::Instant;
use ikari::ui::collect_frame_time_ms;
use ikari::ui::CullingFrustumLockMode;
use ikari::ui::GpuTimerScopeResultWrapper;
use plotters::prelude::*;
use plotters::style::RED;
use plotters_iced::{Chart, ChartWidget, DrawingBackend};

use ikari::time::Duration;

use crate::game::INITIAL_ENABLE_CULLING_FRUSTUM_DEBUG;
use crate::game::INITIAL_ENABLE_POINT_LIGHT_CULLING_FRUSTUM_DEBUG;
use crate::game::INITIAL_ENABLE_SHADOW_DEBUG;
use crate::game::INITIAL_ENABLE_SOFT_SHADOWS;
use crate::game::INITIAL_ENABLE_VSYNC;
use crate::game::INITIAL_IS_SHOWING_CAMERA_POSE;
use crate::game::INITIAL_IS_SHOWING_CURSOR_MARKER;
use crate::game::INITIAL_SHADOW_BIAS;
use crate::game::INITIAL_SKYBOX_WEIGHT;
use crate::game::INITIAL_SOFT_SHADOW_FACTOR;
use crate::game::INITIAL_SOFT_SHADOW_GRID_DIMS;

pub const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("./fonts/Lato-Regular.ttf");
pub const DEFAULT_FONT_NAME: &str = "Lato";

const FRAME_TIME_HISTORY_SIZE: usize = 720;

#[derive(Debug, Clone)]
pub struct AudioSoundStats {
    pub length_seconds: Option<f32>,
    pub pos_seconds: f32,
    pub buffered_to_pos_seconds: f32,
}

#[derive(Debug, Clone)]
pub enum Message {
    ViewportDimsChanged((u32, u32)),
    CursorPosChanged(winit::dpi::PhysicalPosition<f64>),
    FrameCompleted(Duration),
    GpuFrameCompleted(Vec<GpuTimerScopeResultWrapper>),
    CameraPoseChanged((Vec3, ControlledViewDirection)),
    AudioSoundStatsChanged((GameFilePath, AudioSoundStats)),
    ToggleVSync(bool),
    ToggleCameraPose(bool),
    ToggleCursorMarker(bool),
    ToggleFpsChart(bool),
    ToggleGpuSpans(bool),
    ToggleSoftShadows(bool),
    ToggleDrawCullingFrustum(bool),
    ToggleDrawPointLightCullingFrusta(bool),
    ToggleShadowDebug(bool),
    ToggleAudioStats(bool),
    ShadowBiasChanged(f32),
    SkyboxWeightChanged(f32),
    SoftShadowFactorChanged(f32),
    SoftShadowGridDimsChanged(u32),
    CullingFrustumLockModeChanged(CullingFrustumLockMode),
    TogglePopupMenu,
    ClosePopupMenu,
    ExitButtonPressed,
    GenerateProfileDump,
}

#[derive(Debug)]
pub struct UiOverlay {
    clock: canvas::Cache,
    viewport_dims: (u32, u32),
    pub cursor_position: winit::dpi::PhysicalPosition<f64>,
    fps_chart: FpsChart,
    is_showing_fps_chart: bool,
    is_showing_gpu_spans: bool,
    is_showing_audio_stats: bool,
    pub is_showing_options_menu: bool,
    pub was_exit_button_pressed: bool,
    camera_pose: Option<(Vec3, ControlledViewDirection)>, // position, direction

    audio_sound_stats: BTreeMap<String, AudioSoundStats>,

    pub enable_vsync: bool,
    pub enable_soft_shadows: bool,
    pub skybox_weight: f32,
    pub shadow_bias: f32,
    pub soft_shadow_factor: f32,
    pub enable_shadow_debug: bool,
    pub draw_culling_frustum: bool,
    pub draw_point_light_culling_frusta: bool,
    pub culling_frustum_lock_mode: CullingFrustumLockMode,
    pub soft_shadow_grid_dims: u32,
    pub is_showing_camera_pose: bool,
    pub is_showing_cursor_marker: bool,

    pub pending_perf_dump: Option<PendingPerfDump>,
    perf_dump_completion_time: Option<Instant>,
}

pub struct ContainerStyle;

impl container::StyleSheet for ContainerStyle {
    type Style = Theme;

    fn appearance(&self, _: &Self::Style) -> container::Appearance {
        container::Appearance {
            background: Some(Background::Color(iced::Color::from_rgba(
                0.3, 0.3, 0.3, 0.6,
            ))),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
struct FpsChart {
    recent_frame_times: Vec<Duration>,
    recent_gpu_frame_times: Vec<Vec<GpuTimerScopeResultWrapper>>,
    avg_frame_time_millis: Option<f64>,
    avg_gpu_frame_time_millis: Option<f64>,
}

impl Chart<Message> for FpsChart {
    type State = ();

    fn build_chart<DB: DrawingBackend>(&self, _state: &Self::State, mut builder: ChartBuilder<DB>) {
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
                .x_label_area_size(24)
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
            // TODO: verify that it's actually using the correct font here
            let axis_labels_style = (DEFAULT_FONT_NAME, 16, &WHITE);

            chart
                .configure_mesh()
                .x_label_formatter(&|x| format!("{}s", x.abs().round() as i32))
                .y_label_formatter(&|y| format!("{y} fps"))
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
            log::error!("Error building fps chart: {err:?}");
        }
    }
}

impl FpsChart {
    fn view(&self) -> Element<Message, iced::Renderer> {
        ChartWidget::new(self)
            .width(Length::Fixed(400.0))
            .height(Length::Fixed(300.0))
            .into()
    }
}

impl UiOverlay {
    pub fn new(window: &winit::window::Window) -> Self {
        let cursor_position = winit::dpi::PhysicalPosition::new(-1.0, -1.0);

        Self {
            clock: Default::default(),
            viewport_dims: (window.inner_size().width, window.inner_size().height),
            cursor_position,
            fps_chart: FpsChart {
                recent_frame_times: vec![],
                recent_gpu_frame_times: vec![],
                avg_frame_time_millis: None,
                avg_gpu_frame_time_millis: None,
            },

            audio_sound_stats: BTreeMap::new(),

            camera_pose: None,
            is_showing_camera_pose: INITIAL_IS_SHOWING_CAMERA_POSE,
            is_showing_cursor_marker: INITIAL_IS_SHOWING_CURSOR_MARKER,
            is_showing_fps_chart: true,
            is_showing_gpu_spans: false,
            is_showing_options_menu: false,
            was_exit_button_pressed: false,
            is_showing_audio_stats: false,
            enable_vsync: INITIAL_ENABLE_VSYNC,
            enable_soft_shadows: INITIAL_ENABLE_SOFT_SHADOWS,
            skybox_weight: INITIAL_SKYBOX_WEIGHT,
            shadow_bias: INITIAL_SHADOW_BIAS,
            soft_shadow_factor: INITIAL_SOFT_SHADOW_FACTOR,
            enable_shadow_debug: INITIAL_ENABLE_SHADOW_DEBUG,
            draw_culling_frustum: INITIAL_ENABLE_CULLING_FRUSTUM_DEBUG,
            draw_point_light_culling_frusta: INITIAL_ENABLE_POINT_LIGHT_CULLING_FRUSTUM_DEBUG,
            culling_frustum_lock_mode: CullingFrustumLockMode::None,
            soft_shadow_grid_dims: INITIAL_SOFT_SHADOW_GRID_DIMS,
            pending_perf_dump: None,
            perf_dump_completion_time: None,
        }
    }
}

impl<Message> canvas::Program<Message, iced::Renderer> for UiOverlay {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let clock = self.clock.draw(renderer, bounds.size(), |frame| {
            let center = iced::Point::new(
                frame.width() * self.cursor_position.x as f32,
                frame.height() * self.cursor_position.y as f32,
            );
            let radius = 24.0;
            let background = canvas::Path::circle(center, radius);
            frame.fill(&background, iced::Color::from_rgba8(0x12, 0x93, 0xD8, 0.5));
        });

        vec![clock]
    }
}

// the iced ui
impl runtime::Program for UiOverlay {
    type Renderer = iced::Renderer;
    type Message = Message;

    fn update(&mut self, message: Message) -> Command<Message> {
        let moving_average_alpha = 0.01;

        match message {
            Message::ViewportDimsChanged(new_state) => {
                self.viewport_dims = new_state;
            }
            Message::CursorPosChanged(new_state) => {
                self.cursor_position = new_state;
            }
            Message::FrameCompleted(frame_duration) => {
                self.clock.clear();

                let frame_time_ms = frame_duration.as_nanos() as f64 / 1_000_000.0;
                self.fps_chart.avg_frame_time_millis =
                    Some(match self.fps_chart.avg_frame_time_millis {
                        Some(avg_frame_time_millis) => {
                            (1.0 - moving_average_alpha) * avg_frame_time_millis
                                + (moving_average_alpha * frame_time_ms)
                        }
                        None => frame_time_ms,
                    });

                self.fps_chart.recent_frame_times.push(frame_duration);
                if self.fps_chart.recent_frame_times.len() > FRAME_TIME_HISTORY_SIZE {
                    self.fps_chart.recent_frame_times.remove(0);
                }

                let mut clear = false;
                if let Some(pending_perf_dump) = &self.pending_perf_dump {
                    let pending_perf_dump_guard = pending_perf_dump.lock().unwrap();
                    if pending_perf_dump_guard.is_some() {
                        if let Some(perf_dump_completion_time) = self.perf_dump_completion_time {
                            if perf_dump_completion_time.elapsed()
                                > std::time::Duration::from_secs_f32(2.0)
                            {
                                clear = true;
                            }
                        } else {
                            self.perf_dump_completion_time = Some(Instant::now());
                        }
                    }
                }

                if clear {
                    self.pending_perf_dump = None;
                    self.perf_dump_completion_time = None;
                }
            }
            Message::GpuFrameCompleted(frames) => {
                let frame_time_ms = collect_frame_time_ms(&frames);
                self.fps_chart.avg_gpu_frame_time_millis =
                    Some(match self.fps_chart.avg_gpu_frame_time_millis {
                        Some(avg_gpu_frame_time_millis) => {
                            (1.0 - moving_average_alpha) * avg_gpu_frame_time_millis
                                + (moving_average_alpha * frame_time_ms)
                        }
                        None => frame_time_ms,
                    });

                self.fps_chart.recent_gpu_frame_times.push(frames);
                if self.fps_chart.recent_gpu_frame_times.len() > FRAME_TIME_HISTORY_SIZE {
                    self.fps_chart.recent_gpu_frame_times.remove(0);
                }
            }
            Message::AudioSoundStatsChanged((track_path, stats)) => {
                self.audio_sound_stats.insert(
                    track_path.relative_path.to_string_lossy().to_string(),
                    stats,
                );
            }
            Message::CameraPoseChanged(new_state) => {
                self.camera_pose = Some(new_state);
            }
            Message::ToggleVSync(new_state) => {
                self.enable_vsync = new_state;
            }
            Message::ToggleCameraPose(new_state) => {
                self.is_showing_camera_pose = new_state;
            }
            Message::ToggleCursorMarker(new_state) => {
                self.is_showing_cursor_marker = new_state;
            }
            Message::ToggleFpsChart(new_state) => {
                self.is_showing_fps_chart = new_state;
            }
            Message::ToggleGpuSpans(new_state) => {
                self.is_showing_gpu_spans = new_state;
            }
            Message::ToggleSoftShadows(new_state) => {
                self.enable_shadow_debug = new_state;
            }
            Message::ToggleDrawCullingFrustum(new_state) => {
                self.draw_culling_frustum = new_state;
                if !self.draw_culling_frustum {
                    self.culling_frustum_lock_mode = CullingFrustumLockMode::None;
                }
            }
            Message::ToggleDrawPointLightCullingFrusta(new_state) => {
                self.draw_point_light_culling_frusta = new_state;
            }
            Message::ToggleShadowDebug(new_state) => {
                self.enable_soft_shadows = new_state;
            }
            Message::ToggleAudioStats(new_state) => {
                self.is_showing_audio_stats = new_state;
            }
            Message::SkyboxWeightChanged(new_state) => {
                self.skybox_weight = new_state;
            }
            Message::ShadowBiasChanged(new_state) => {
                self.shadow_bias = new_state;
            }
            Message::SoftShadowFactorChanged(new_state) => {
                self.soft_shadow_factor = new_state;
            }
            Message::SoftShadowGridDimsChanged(new_state) => {
                self.soft_shadow_grid_dims = new_state;
            }
            Message::CullingFrustumLockModeChanged(new_state) => {
                self.culling_frustum_lock_mode = new_state;
            }
            Message::ClosePopupMenu => self.is_showing_options_menu = false,
            Message::ExitButtonPressed => self.was_exit_button_pressed = true,
            Message::GenerateProfileDump => {
                self.pending_perf_dump = Some(generate_profile_dump());
            }
            Message::TogglePopupMenu => {
                self.is_showing_options_menu = !self.is_showing_options_menu
            }
        }

        Command::none()
    }

    fn view(&self) -> Element<Message, iced::Renderer> {
        if self.fps_chart.recent_frame_times.is_empty() {
            return Row::new().into();
        }

        let moving_average_alpha = 0.01;

        let container_style = Box::new(ContainerStyle {});

        let mut rows = Column::new()
            .width(Length::Shrink)
            .height(Length::Shrink)
            .spacing(4);

        let has_gpu_frame_time_data = self
            .fps_chart
            .recent_gpu_frame_times
            .iter()
            .any(|list| !list.is_empty());

        if let Some(avg_frame_time_millis) = self.fps_chart.avg_frame_time_millis {
            let cpu_frametime_string = format!(
                "{:.2}ms ({:.2}fps)",
                avg_frame_time_millis,
                1_000.0 / avg_frame_time_millis,
            );
            let gpu_frametime_string = if has_gpu_frame_time_data {
                format!(
                    ", GPU: {:.2}ms",
                    self.fps_chart.avg_gpu_frame_time_millis.unwrap_or_default()
                )
            } else {
                "".into()
            };
            rows = rows.push(text(&format!(
                "Frametime: {cpu_frametime_string}{gpu_frametime_string}"
            )));
        }

        if self.is_showing_camera_pose {
            if let Some((camera_position, camera_direction)) = self.camera_pose {
                rows = rows.push(text(&format!(
                    "Camera position:  x={:.2}, y={:.2}, z={:.2}",
                    camera_position.x, camera_position.y, camera_position.z
                )));
                let two_pi = 2.0 * std::f32::consts::PI;
                let camera_horizontal = (camera_direction.horizontal + two_pi) % two_pi;
                let camera_vertical = camera_direction.vertical;
                rows = rows.push(text(&format!(
                    "Camera direction: h={:.2} ({:.2} deg), v={:.2} ({:.2} deg)",
                    camera_horizontal,
                    rad_to_deg(camera_horizontal),
                    camera_vertical,
                    rad_to_deg(camera_vertical),
                )));
                // rows = rows.push(text(&format!(
                //     "Camera direction: {:?}",
                //     camera_direction.to_vector()
                // )));
            }
        }

        if self.is_showing_audio_stats {
            for (file_path, stats) in &self.audio_sound_stats {
                let format_timestamp = |timestamp| format!("{timestamp:.2}");
                let pos = format_timestamp(stats.pos_seconds);
                let length = stats
                    .length_seconds
                    .map(|length_seconds| {
                        let length = format_timestamp(length_seconds);
                        format!("/ {length}")
                    })
                    .unwrap_or_default();
                let buffered_pos = format_timestamp(stats.buffered_to_pos_seconds);
                rows = rows.push(text(&format!(
                    "{file_path}: {pos}{length}, buffer to {buffered_pos}"
                )));
            }
        }

        if self.is_showing_gpu_spans {
            let mut span_set: HashSet<&str> = HashSet::new();
            for frame_spans in &self.fps_chart.recent_gpu_frame_times {
                // single frame
                for span in frame_spans {
                    span_set.insert(&span.label);
                }
            }
            let span_names: Vec<&str> = {
                let mut span_names: Vec<&str> = span_set.iter().copied().collect();
                span_names.sort();
                span_names
            };
            let mut avg_span_times: HashMap<&str, f64> = HashMap::new();

            for frame_spans in &self.fps_chart.recent_gpu_frame_times {
                // process all spans of a single frame

                let mut totals_by_span: HashMap<&str, f64> = HashMap::new();
                for span in frame_spans {
                    let span_time_ms = (span.time.end - span.time.start) * 1000.0;
                    let total = totals_by_span.entry(&span.label).or_default();
                    *total += span_time_ms;
                }

                for span_name in &span_names {
                    use std::collections::hash_map::Entry;

                    let frame_time = totals_by_span.entry(span_name).or_default();
                    match avg_span_times.entry(span_name) {
                        Entry::Occupied(mut entry) => {
                            *entry.get_mut() = (1.0 - moving_average_alpha) * entry.get()
                                + (moving_average_alpha * *frame_time);
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(*frame_time);
                        }
                    }
                }
            }

            let mut avg_span_times_vec: Vec<_> = avg_span_times.iter().collect();
            avg_span_times_vec.sort_by_key(|(span, _)| **span);
            avg_span_times_vec.sort_by_key(|(_span, avg_span_frame_time)| {
                (avg_span_frame_time.max(0.01) * 100.0) as u64
            });
            avg_span_times_vec.reverse();
            for (span, span_frame_time) in avg_span_times_vec {
                let msg = &format!("{span:}: {span_frame_time:.2}ms");
                rows = rows.push(text(msg).size(14));
            }
        }

        if self.is_showing_fps_chart {
            let padding = [16, 20, 16, 0]; // top, right, bottom, left
            rows = rows.push(Container::new(self.fps_chart.view()).padding(padding));
        }

        let background_content = Container::new(
            Row::new()
                .width(Length::Shrink)
                .height(Length::Shrink)
                .padding(8)
                .push(
                    Container::new(rows)
                        .padding(8)
                        .style(iced::theme::Container::Custom(container_style)),
                ),
        )
        .width(Length::Fill)
        .height(Length::Fill);

        let modal_content: Option<Element<_, _>> = self.is_showing_options_menu.then(|| {
            let separator_line = Text::new("-------------")
                .width(Length::Fill)
                .horizontal_alignment(Horizontal::Center);

            let mut options = Column::new()
                .spacing(16)
                .padding([6, 48, 6, 6])
                .width(Length::Fill);

            // vsync
            #[cfg(not(target_arch = "wasm32"))]
            {
                options = options.push(checkbox(
                    "Enable VSync",
                    self.enable_vsync,
                    Message::ToggleVSync,
                ));
            }

            // camera debug
            options = options.push(checkbox(
                "Show Camera Pose",
                self.is_showing_camera_pose,
                Message::ToggleCameraPose,
            ));

            // cursor marker debug
            options = options.push(checkbox(
                "Show Cursor Marker",
                self.is_showing_cursor_marker,
                Message::ToggleCursorMarker,
            ));

            // audio stats debug
            options = options.push(checkbox(
                "Show Audio Stats",
                self.is_showing_audio_stats,
                Message::ToggleAudioStats,
            ));

            // fps overlay
            options = options.push(checkbox(
                "Show FPS Chart",
                self.is_showing_fps_chart,
                Message::ToggleFpsChart,
            ));

            if has_gpu_frame_time_data {
                options = options.push(checkbox(
                    "Show Detailed GPU Frametimes",
                    self.is_showing_gpu_spans,
                    Message::ToggleGpuSpans,
                ));
            }

            // frustum culling debug
            options = options.push(separator_line.clone());
            options = options.push(checkbox(
                "Enable Frustum Culling Debug",
                self.draw_culling_frustum,
                Message::ToggleDrawCullingFrustum,
            ));
            if self.draw_culling_frustum {
                options = options.push(Text::new("Lock Culling Frustum"));
                for mode in CullingFrustumLockMode::ALL {
                    options = options.push(radio(
                        format!("{mode}"),
                        mode,
                        Some(self.culling_frustum_lock_mode),
                        Message::CullingFrustumLockModeChanged,
                    ));
                }
            }

            // point light frusta debug
            options = options.push(checkbox(
                "Enable Point Light Frustum Culling Debug",
                self.draw_point_light_culling_frusta,
                Message::ToggleDrawPointLightCullingFrusta,
            ));

            // shadow debug
            options = options.push(separator_line.clone());
            options = options.push(checkbox(
                "Enable Shadow Debug",
                self.enable_shadow_debug,
                Message::ToggleSoftShadows,
            ));
            options = options.push(checkbox(
                "Enable Soft Shadows",
                self.enable_soft_shadows,
                Message::ToggleShadowDebug,
            ));
            options = options.push(Text::new(format!(
                "Skybox weight: {:.5}",
                self.skybox_weight
            )));
            options = options.push(
                slider(0.0..=1.0, self.skybox_weight, Message::SkyboxWeightChanged).step(0.01),
            );
            options = options.push(Text::new(format!("Shadow Bias: {:.5}", self.shadow_bias)));
            options = options.push(
                slider(
                    0.00001..=0.005,
                    self.shadow_bias,
                    Message::ShadowBiasChanged,
                )
                .step(0.00001),
            );
            options = options.push(Text::new(format!(
                "Soft Shadow Factor: {:.4}",
                self.soft_shadow_factor
            )));
            options = options.push(
                slider(
                    0.0001..=0.005,
                    self.soft_shadow_factor,
                    Message::SoftShadowFactorChanged,
                )
                .step(0.0001),
            );
            options = options.push(Text::new(format!(
                "Soft Shadow Grid Dims: {:}",
                self.soft_shadow_grid_dims
            )));
            options = options.push(
                slider(
                    0..=16u32,
                    self.soft_shadow_grid_dims,
                    Message::SoftShadowGridDimsChanged,
                )
                .step(1),
            );

            // profile dump
            if can_generate_profile_dump() {
                if let Some(pending_perf_dump) = &self.pending_perf_dump {
                    let (message, color) = if self.perf_dump_completion_time.is_some() {
                        match *pending_perf_dump.lock().unwrap() {
                            Some(Ok(_)) => (
                                "Profile dump complete!".to_string(),
                                iced::Color::from_rgb(0.7, 1.0, 0.0),
                            ),
                            Some(Err(_)) => (
                                "Profile dump failed! See stdout for details.".to_string(),
                                iced::Color::from_rgb(0.9, 0.1, 0.2),
                            ),
                            None => {
                                unreachable!();
                            }
                        }
                    } else {
                        let time_secs = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or(Default::default())
                            .as_secs_f64();
                        let ellipsis_num = 2 + (time_secs * 4.0).sin().round() as i32;
                        let elipsis_str =
                            (0..ellipsis_num).map(|_| ".").collect::<Vec<_>>().join("");
                        (
                            format!("Generating profile dump{elipsis_str}"),
                            iced::Color::from_rgb(1.0, 0.7, 0.1),
                        )
                    };
                    options = options.push(Text::new(message).style(color));
                } else {
                    options = options.push(
                        Button::new(
                            Text::new("Generate Profile Dump")
                                .horizontal_alignment(Horizontal::Center),
                        )
                        .width(Length::Shrink)
                        .on_press(Message::GenerateProfileDump),
                    );
                }
            }

            // exit button
            options = options.push(
                Button::new(Text::new("Exit Game").horizontal_alignment(Horizontal::Center))
                    .width(Length::Shrink)
                    .on_press(Message::ExitButtonPressed),
            );

            iced_aw::Card::new(
                Text::new("Options"),
                scrollable(options)
                    .height(Length::Fixed(self.viewport_dims.1 as f32 * 0.75 - 50.0)),
            )
            .max_width(300.0)
            .on_close(Message::ClosePopupMenu)
            .into()
        });

        if self.is_showing_cursor_marker {
            let overlay = {
                let canvas: canvas::Canvas<&Self, Message, iced::Renderer> = canvas(self as &Self)
                    .width(Length::Fill)
                    .height(Length::Fill);

                Element::from(
                    container(canvas)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .padding(0),
                )
            };

            Modal::new(
                floating_element(background_content, overlay)
                    .anchor(floating_element::Anchor::NorthWest),
                modal_content,
            )
            .into()
        } else {
            Modal::new(background_content, modal_content).into()
        }
    }
}
