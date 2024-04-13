use std::collections::hash_map::Entry;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use glam::Vec3;
use iced::alignment::Horizontal;
use iced::font::Family;
use iced::font::Weight;
use iced::widget::button;

use iced::Border;
use iced::Font;

use iced::widget::{
    canvas, checkbox, container, radio, scrollable, slider, text, Column, Container, Row,
};
use iced::Length;
use iced::{mouse, Background, Command, Element, Rectangle, Theme};
use iced_aw::style::modal;
use iced_aw::{floating_element, Modal};
use iced_winit::runtime;
use ikari::file_manager::GameFilePath;
use ikari::framerate_limiter::FramerateLimit;
use ikari::framerate_limiter::FramerateLimitType;
use ikari::player_controller::ControlledViewDirection;
use ikari::profile_dump::can_generate_profile_dump;
use ikari::profile_dump::generate_profile_dump;
use ikari::profile_dump::PendingPerfDump;
use ikari::renderer::BloomType;
use ikari::renderer::CullingFrustumLockMode;
use ikari::renderer::CullingStats;
use ikari::renderer::MIN_SHADOW_MAP_BIAS;
use ikari::texture::apply_render_scale;
use ikari::time::Instant;
use ikari::time_tracker::FrameDurations;
use ikari::time_tracker::FrameInstants;
use ikari::ui::UiProgramEvents;
use plotters::prelude::*;
use plotters_iced::{Chart, ChartWidget, DrawingBackend};

use ikari::time::Duration;

use crate::game::INITIAL_BLOOM_TYPE;
use crate::game::INITIAL_ENABLE_CASCADE_DEBUG;
use crate::game::INITIAL_ENABLE_CULLING_FRUSTUM_DEBUG;
use crate::game::INITIAL_ENABLE_DEPTH_PREPASS;
use crate::game::INITIAL_ENABLE_DIRECTIONAL_LIGHT_CULLING_FRUSTUM_DEBUG;
use crate::game::INITIAL_ENABLE_POINT_LIGHT_CULLING_FRUSTUM_DEBUG;
use crate::game::INITIAL_ENABLE_SHADOW_DEBUG;
use crate::game::INITIAL_ENABLE_SOFT_SHADOWS;
use crate::game::INITIAL_ENABLE_VSYNC;
use crate::game::INITIAL_FAR_PLANE_DISTANCE;
use crate::game::INITIAL_FOV_X;
use crate::game::INITIAL_FRAMERATE_LIMIT;
use crate::game::INITIAL_IS_SHOWING_CAMERA_POSE;
use crate::game::INITIAL_IS_SHOWING_CURSOR_MARKER;
use crate::game::INITIAL_NEAR_PLANE_DISTANCE;
use crate::game::INITIAL_NEW_BLOOM_INTENSITY;
use crate::game::INITIAL_NEW_BLOOM_RADIUS;
use crate::game::INITIAL_RENDER_SCALE;
use crate::game::INITIAL_SHADOW_BIAS;
use crate::game::INITIAL_SHADOW_SMALL_OBJECT_CULLING_SIZE_PIXELS;
use crate::game::INITIAL_SKYBOX_WEIGHT;
use crate::game::INITIAL_SOFT_SHADOWS_MAX_DISTANCE;
use crate::game::INITIAL_SOFT_SHADOW_FACTOR;
use crate::game::INITIAL_SOFT_SHADOW_GRID_DIMS;

// Default
pub const LATO_FONT_BYTES: &[u8] = include_bytes!("./fonts/Lato-Regular.ttf");
pub const LATO_BOLD_FONT_BYTES: &[u8] = include_bytes!("./fonts/Lato-Bold.ttf");
pub const LATO_FONT_NAME: &str = "Lato";

pub const PACIFICO_FONT_BYTES: &[u8] = include_bytes!("./fonts/Pacifico-Regular.ttf");
pub const PACIFICO_FONT_NAME: &str = "Pacifico";

const FRAME_TIME_HISTORY_SIZE: usize = 5 * 144 + 1; // 1 more than 5 seconds of 144hz
const FRAME_TIMES_MOVING_AVERAGE_ALPHA: f64 = 0.01;
const FPS_CHART_LINE_COLORS: [RGBAColor; 4] = [
    RGBAColor(165, 242, 85, 1.0),  // total
    RGBAColor(49, 168, 224, 0.8),  // update
    RGBAColor(159, 127, 242, 0.8), // render
    RGBAColor(253, 183, 23, 0.8),  // gpu
];
const SHOW_BLOOM_TYPE: bool = false;
pub(crate) const THEME: iced::Theme = iced::Theme::TokyoNight;

#[derive(Debug, Clone)]
pub struct AudioSoundStats {
    pub length_seconds: Option<f32>,
    pub pos_seconds: f32,
    pub buffered_to_pos_seconds: f32,
}

#[derive(Debug, Clone)]
pub struct FrameStats {
    pub instants: FrameInstants,
    pub durations: FrameDurations,
    pub gpu_timer_query_results: Vec<wgpu_profiler::GpuTimerQueryResult>,
    pub culling_stats: Option<CullingStats>,
}

#[derive(Debug, Clone)]
pub enum Message {
    ViewportDimsChanged((u32, u32)),
    MonitorRefreshRateChanged(Option<f32>),
    CursorPosChanged(winit::dpi::PhysicalPosition<f64>),
    FrameCompleted(FrameStats),
    CameraPoseChanged((Vec3, ControlledViewDirection)),
    AudioSoundStatsChanged((GameFilePath, AudioSoundStats)),
    VsyncChanged(bool),
    FramerateLimitTypeChanged(FramerateLimitType),
    CustomFramerateLimitChanged(f32),
    RenderScaleChanged(f32),
    FovxChanged(f32),
    NearPlaneDistanceChanged(f32),
    FarPlaneDistanceChanged(f32),
    BloomTypeChanged(BloomType),
    NewBloomRadiusChanged(f32),
    NewBloomIntensityChanged(f32),
    ShadowSmallObjectCullingSizeChanged(f32),
    SoftShadowsMaxDistanceChanged(f32),
    ToggleDepthPrepass(bool),
    ToggleCameraPose(bool),
    ToggleCursorMarker(bool),
    ToggleFps(bool),
    ToggleFpsChart(bool),
    ToggleGpuSpans(bool),
    ToggleSoftShadows(bool),
    ToggleDrawCullingFrustum(bool),
    ToggleDrawPointLightCullingFrusta(bool),
    ToggleDrawDirectionalLightCullingFrusta(bool),
    ToggleShadowDebug(bool),
    ToggleCascadeDebug(bool),
    ToggleAudioStats(bool),
    ToggleCullingStats(bool),
    ShadowBiasChanged(f32),
    SkyboxWeightChanged(f32),
    SoftShadowFactorChanged(f32),
    SoftShadowGridDimsChanged(u32),
    CullingFrustumLockModeChanged(CullingFrustumLockMode),
    TogglePopupMenu,
    ClosePopupMenu,
    ExitButtonPressed,
    GenerateProfileDump,
    DefaultSettingsButtonPressed,
    ToggleGeneralSettingsCollapse,
    ToggleCameraSettingsCollapse,
    TogglePostEffectSettingsCollapse,
    ToggleShadowSettingsCollapse,
    ToggleDebugSettingsCollapse,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GeneralSettings {
    collapsed: bool,

    pub enable_depth_prepass: bool,
    pub enable_vsync: bool,
    pub framerate_limit_type: FramerateLimitType,
    pub custom_framerate_limit: f32,
    pub render_scale: f32,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            collapsed: true,
            enable_vsync: INITIAL_ENABLE_VSYNC,
            enable_depth_prepass: INITIAL_ENABLE_DEPTH_PREPASS,
            framerate_limit_type: INITIAL_FRAMERATE_LIMIT.into(),
            custom_framerate_limit: match INITIAL_FRAMERATE_LIMIT {
                FramerateLimit::None => 60.0,
                FramerateLimit::Monitor => 60.0,
                FramerateLimit::Custom(val) => val,
            },
            render_scale: INITIAL_RENDER_SCALE,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CameraSettings {
    collapsed: bool,

    pub fov_x: f32,
    pub near_plane_distance: f32,
    pub far_plane_distance: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            collapsed: true,

            fov_x: INITIAL_FOV_X,
            near_plane_distance: INITIAL_NEAR_PLANE_DISTANCE,
            far_plane_distance: INITIAL_FAR_PLANE_DISTANCE,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PostEffectSettings {
    collapsed: bool,

    pub bloom_type: BloomType,

    pub new_bloom_radius: f32,
    pub new_bloom_intensity: f32,

    pub skybox_weight: f32,
}

impl Default for PostEffectSettings {
    fn default() -> Self {
        Self {
            collapsed: true,

            bloom_type: INITIAL_BLOOM_TYPE,

            new_bloom_radius: INITIAL_NEW_BLOOM_RADIUS,
            new_bloom_intensity: INITIAL_NEW_BLOOM_INTENSITY,

            skybox_weight: INITIAL_SKYBOX_WEIGHT,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ShadowSettings {
    collapsed: bool,

    pub enable_soft_shadows: bool,
    pub shadow_bias: f32,
    pub soft_shadow_factor: f32,
    pub soft_shadows_max_distance: f32,
    pub soft_shadow_grid_dims: u32,
    pub shadow_small_object_culling_size_pixels: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            collapsed: true,

            enable_soft_shadows: INITIAL_ENABLE_SOFT_SHADOWS,
            shadow_bias: INITIAL_SHADOW_BIAS,
            soft_shadow_factor: INITIAL_SOFT_SHADOW_FACTOR,
            soft_shadows_max_distance: INITIAL_SOFT_SHADOWS_MAX_DISTANCE,
            soft_shadow_grid_dims: INITIAL_SOFT_SHADOW_GRID_DIMS,
            shadow_small_object_culling_size_pixels:
                INITIAL_SHADOW_SMALL_OBJECT_CULLING_SIZE_PIXELS,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DebugSettings {
    collapsed: bool,

    pub draw_culling_frustum: bool,
    pub draw_point_light_culling_frusta: bool,
    pub draw_directional_light_culling_frusta: bool,
    pub enable_shadow_debug: bool,
    pub enable_cascade_debug: bool,
    pub record_culling_stats: bool,
    pub culling_frustum_lock_mode: CullingFrustumLockMode,

    is_showing_fps: bool,
    is_showing_fps_chart: bool,
    is_showing_gpu_spans: bool,
    is_showing_audio_stats: bool,
    is_showing_camera_pose: bool,
    pub is_showing_cursor_marker: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            collapsed: true,

            draw_culling_frustum: INITIAL_ENABLE_CULLING_FRUSTUM_DEBUG,
            draw_point_light_culling_frusta: INITIAL_ENABLE_POINT_LIGHT_CULLING_FRUSTUM_DEBUG,
            draw_directional_light_culling_frusta:
                INITIAL_ENABLE_DIRECTIONAL_LIGHT_CULLING_FRUSTUM_DEBUG,
            enable_shadow_debug: INITIAL_ENABLE_SHADOW_DEBUG,
            enable_cascade_debug: INITIAL_ENABLE_CASCADE_DEBUG,
            record_culling_stats: false,
            culling_frustum_lock_mode: CullingFrustumLockMode::default(),

            is_showing_fps: true,
            is_showing_camera_pose: INITIAL_IS_SHOWING_CAMERA_POSE,
            is_showing_cursor_marker: INITIAL_IS_SHOWING_CURSOR_MARKER,
            is_showing_fps_chart: false,
            is_showing_gpu_spans: false,
            is_showing_audio_stats: false,
        }
    }
}

#[derive(Debug)]
pub struct UiOverlay {
    clock: canvas::Cache,
    viewport_dims: (u32, u32),
    monitor_refresh_rate: Option<f32>,
    cursor_position: winit::dpi::PhysicalPosition<f64>,
    fps_chart: FpsChart,
    camera_pose: Option<(Vec3, ControlledViewDirection)>, // position, direction
    audio_sound_stats: BTreeMap<String, AudioSoundStats>,
    pending_perf_dump: Option<PendingPerfDump>,
    perf_dump_completion_time: Option<Instant>,
    culling_stats: Option<CullingStats>,
    frame_counter: usize,

    pub is_showing_options_menu: bool,
    pub was_exit_button_pressed: bool,

    pub general_settings: GeneralSettings,
    pub camera_settings: CameraSettings,
    pub post_effect_settings: PostEffectSettings,
    pub shadow_settings: ShadowSettings,
    pub debug_settings: DebugSettings,
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

pub struct CollapsibleButtonStyle;

impl CollapsibleButtonStyle {
    const HOVERED_ALPHA: f32 = 0.5;
    const PRESSED_ALPHA: f32 = 0.3;
    const DISABLED_ALPHA: f32 = 0.3;

    fn transparent_color(color: iced::Color, alpha: f32) -> iced::Color {
        iced::Color::from_rgba(color.r, color.g, color.b, color.a * alpha)
    }
}

impl button::StyleSheet for CollapsibleButtonStyle {
    type Style = Theme;

    fn active(&self, style: &Self::Style) -> button::Appearance {
        button::Appearance {
            border: Border::with_radius(2),
            text_color: style.palette().text,
            ..button::Appearance::default()
        }
    }

    fn hovered(&self, style: &Self::Style) -> button::Appearance {
        let active = self.active(style);

        button::Appearance {
            text_color: Self::transparent_color(active.text_color, Self::HOVERED_ALPHA),
            ..active
        }
    }

    fn pressed(&self, style: &Self::Style) -> button::Appearance {
        let active = self.active(style);

        button::Appearance {
            text_color: Self::transparent_color(active.text_color, Self::PRESSED_ALPHA),
            ..active
        }
    }

    fn disabled(&self, style: &Self::Style) -> button::Appearance {
        let active = self.active(style);

        button::Appearance {
            text_color: Self::transparent_color(active.text_color, Self::DISABLED_ALPHA),
            ..active
        }
    }
}

pub struct ModalStyle;

impl modal::StyleSheet for ModalStyle {
    type Style = Theme;

    fn active(&self, _style: &Self::Style) -> modal::Appearance {
        modal::Appearance {
            background: iced::Color::from_rgba(0.0, 0.0, 0.0, 0.5).into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct FpsChart {
    recent_frame_times: Vec<(Instant, FrameDurations, Option<Duration>)>,
    avg_total_frame_time_millis: Option<f64>,
    avg_sleep_time_ms: Option<f64>,
    avg_get_surface_time_ms: Option<f64>,
    avg_update_time_ms: Option<f64>,
    avg_render_time_ms: Option<f64>,
    avg_gpu_frame_time_ms: Option<f64>,
    avg_gpu_frame_time_per_span: HashMap<String, f64>,
}

impl Chart<Message> for FpsChart {
    type State = ();

    fn build_chart<DB: DrawingBackend>(&self, _state: &Self::State, mut builder: ChartBuilder<DB>) {
        let result: Result<(), String> = (|| {
            let most_recent_start_time =
                self.recent_frame_times[self.recent_frame_times.len() - 1].0;

            let oldest_ft_age_secs =
                (most_recent_start_time - self.recent_frame_times[0].0).as_secs_f32();
            let mut chart_data: [Vec<(f32, i32)>; 4] = Default::default();

            for chart_data in chart_data.iter_mut() {
                chart_data.reserve_exact(self.recent_frame_times.len());
            }

            for (start_time, durations, gpu_duration) in self.recent_frame_times.iter() {
                let x = -(most_recent_start_time - *start_time).as_secs_f32();

                chart_data[0].push((x, (1.0 / durations.total.as_secs_f32()).round() as i32));

                if let Some(update_duration) = durations.update {
                    chart_data[1].push((x, (1.0 / update_duration.as_secs_f32()).round() as i32));
                }

                if let Some(render_duration) = durations.render {
                    chart_data[2].push((x, (1.0 / render_duration.as_secs_f32()).round() as i32));
                }

                if let Some(gpu_duration) = gpu_duration {
                    chart_data[3].push((x, (1.0 / gpu_duration.as_secs_f32()).round() as i32));
                }
            }

            // round up to the nearest multiple of 30fps
            let roundup_factor = 30.0;
            let min_y_axis_height = 120;

            let max_y = (0..self.recent_frame_times.len())
                .flat_map(|i| {
                    chart_data
                        .iter()
                        .filter(move |chart_data| chart_data.len() > i)
                        .map(move |chart_data| chart_data[i].1)
                })
                .max()
                .map(|max_fps| {
                    (((max_fps as f32 / roundup_factor).ceil() * roundup_factor) as i32)
                        .max(min_y_axis_height)
                })
                .unwrap_or(min_y_axis_height);

            let fps_grading_size = 30.0;

            let mut chart = builder
                .x_label_area_size(24)
                .y_label_area_size(60)
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
            let axis_labels_style = (LATO_FONT_NAME, 16, &WHITE);

            chart
                .configure_mesh()
                .x_label_formatter(&|x| format!("{}s", x.abs().round() as i32))
                .y_label_formatter(&|y| format!("{y} fps"))
                .disable_x_mesh()
                .y_max_light_lines(1)
                .set_all_tick_mark_size(1)
                .x_labels((oldest_ft_age_secs as usize + 1).min(8))
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

            for (i, chart_data) in chart_data.iter().enumerate() {
                chart
                    .draw_series(plotters::series::LineSeries::new(
                        chart_data.clone(),
                        RGBAColor(
                            FPS_CHART_LINE_COLORS[i].0,
                            FPS_CHART_LINE_COLORS[i].1,
                            FPS_CHART_LINE_COLORS[i].2,
                            if i == 0 { 1.0 } else { 0.3 },
                        )
                        .stroke_width(1),
                    ))
                    .map_err(|err| err.to_string())?;
            }

            Ok(())
        })();

        if let Err(err) = result {
            log::error!("Error building fps chart: {err:?}");
        }
    }
}

impl FpsChart {
    pub fn on_frame_completed(
        &mut self,
        instants: FrameInstants,
        durations: FrameDurations,
        gpu_timer_query_results: Vec<wgpu_profiler::GpuTimerQueryResult>,
    ) {
        let total_gpu_duration = Self::get_total_gpu_duration(&gpu_timer_query_results);
        self.recompute_avg_frametimes(&durations, total_gpu_duration);
        self.recompute_avg_frametimes_per_gpu_span(&gpu_timer_query_results);

        self.recent_frame_times
            .push((instants.start, durations, total_gpu_duration));
        if self.recent_frame_times.len() > FRAME_TIME_HISTORY_SIZE {
            self.recent_frame_times.remove(0);
        }
    }

    fn get_total_gpu_duration(
        gpu_timer_query_results: &[wgpu_profiler::GpuTimerQueryResult],
    ) -> Option<Duration> {
        if !gpu_timer_query_results.is_empty() {
            let mut total_gpu_time_seconds = 0.0;
            for query_result in gpu_timer_query_results {
                // start can occasionally be larger than end on macos :(
                // see https://github.com/Wumpf/wgpu-profiler/issues/64
                total_gpu_time_seconds +=
                    (query_result.time.end - query_result.time.start).max(0.0);
            }
            Some(Duration::from_secs_f64(total_gpu_time_seconds))
        } else {
            None
        }
    }

    fn recompute_avg_frametimes(
        &mut self,
        new_durations: &FrameDurations,
        new_total_gpu_duration: Option<Duration>,
    ) {
        let compute_new_avg_frametime =
            |current_average: Option<f64>, new_duration: Option<Duration>| {
                let new_duration_ms = new_duration
                    .as_ref()
                    .map(|duration| duration.as_nanos() as f64 / 1_000_000.0);
                match (current_average, new_duration_ms) {
                    (Some(current_average), Some(new_duration_ms)) => Some(
                        (1.0 - FRAME_TIMES_MOVING_AVERAGE_ALPHA) * current_average
                            + (FRAME_TIMES_MOVING_AVERAGE_ALPHA * new_duration_ms),
                    ),
                    _ => new_duration_ms,
                }
            };

        self.avg_total_frame_time_millis =
            compute_new_avg_frametime(self.avg_total_frame_time_millis, Some(new_durations.total));
        self.avg_sleep_time_ms =
            compute_new_avg_frametime(self.avg_sleep_time_ms, new_durations.sleep);
        self.avg_get_surface_time_ms =
            compute_new_avg_frametime(self.avg_get_surface_time_ms, new_durations.get_surface);
        self.avg_update_time_ms =
            compute_new_avg_frametime(self.avg_update_time_ms, new_durations.update);
        self.avg_render_time_ms =
            compute_new_avg_frametime(self.avg_render_time_ms, new_durations.render);
        self.avg_gpu_frame_time_ms =
            compute_new_avg_frametime(self.avg_gpu_frame_time_ms, new_total_gpu_duration);
    }

    fn recompute_avg_frametimes_per_gpu_span(
        &mut self,
        gpu_timer_query_results: &[wgpu_profiler::GpuTimerQueryResult],
    ) {
        let span_set: HashSet<_> = gpu_timer_query_results
            .iter()
            .map(|query_result| query_result.label.clone())
            .collect();

        // clean out old spans from previous frames
        let spans_to_remove: Vec<_> = self
            .avg_gpu_frame_time_per_span
            .keys()
            .filter(|key| !span_set.contains(*key))
            .cloned()
            .collect();

        for span_to_remove in spans_to_remove {
            self.avg_gpu_frame_time_per_span.remove(&span_to_remove);
        }

        let mut span_totals: HashMap<String, f64> = HashMap::new();

        for query_result in gpu_timer_query_results {
            let span_time_ms = (query_result.time.end - query_result.time.start) * 1000.0;
            match span_totals.entry(query_result.label.clone()) {
                Entry::Occupied(mut entry) => {
                    entry.insert(entry.get() + span_time_ms);
                }
                Entry::Vacant(entry) => {
                    entry.insert(span_time_ms);
                }
            }
        }

        for (span_label, span_time_ms) in &span_totals {
            match self.avg_gpu_frame_time_per_span.entry(span_label.clone()) {
                Entry::Occupied(mut entry) => {
                    entry.insert(
                        (1.0 - FRAME_TIMES_MOVING_AVERAGE_ALPHA) * entry.get()
                            + (FRAME_TIMES_MOVING_AVERAGE_ALPHA * span_time_ms),
                    );
                }
                Entry::Vacant(entry) => {
                    entry.insert(*span_time_ms);
                }
            }
        }
    }
}

impl UiOverlay {
    pub fn new(window: &winit::window::Window) -> Self {
        let cursor_position = winit::dpi::PhysicalPosition::new(-1.0, -1.0);

        Self {
            clock: Default::default(),
            viewport_dims: (window.inner_size().width, window.inner_size().height),
            // monitor_refresh_rate: None,
            monitor_refresh_rate: window
                .current_monitor()
                .and_then(|monitor| monitor.refresh_rate_millihertz())
                .map(|millihertz| millihertz as f32 / 1000.0),
            cursor_position,
            fps_chart: FpsChart::default(),
            camera_pose: None,
            audio_sound_stats: BTreeMap::new(),
            pending_perf_dump: None,
            perf_dump_completion_time: None,
            culling_stats: None,

            frame_counter: 0,

            is_showing_options_menu: false,
            was_exit_button_pressed: false,

            general_settings: GeneralSettings::default(),
            camera_settings: CameraSettings::default(),
            post_effect_settings: PostEffectSettings::default(),
            shadow_settings: ShadowSettings::default(),
            debug_settings: DebugSettings::default(),
        }
    }

    fn poll_perf_dump_state(&mut self) {
        let mut clear = false;
        if let Some(pending_perf_dump) = &self.pending_perf_dump {
            let pending_perf_dump_guard = pending_perf_dump.lock();
            if pending_perf_dump_guard.is_some() {
                if let Some(perf_dump_completion_time) = self.perf_dump_completion_time {
                    if perf_dump_completion_time.elapsed() > std::time::Duration::from_secs_f32(2.0)
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
}

impl<Message> canvas::Program<Message, iced::Theme, iced::Renderer> for UiOverlay {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &iced::Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        self.clock.clear();
        let clock = self.clock.draw(renderer, bounds.size(), |frame| {
            let colors = [
                iced::Color::from_rgba8(255, 0, 0, 0.5),
                iced::Color::from_rgba8(0, 255, 0, 0.5),
                iced::Color::from_rgba8(0, 0, 255, 0.5),
            ];
            let color = colors[self.frame_counter % 3];

            let center = iced::Point::new(
                frame.width() * self.cursor_position.x as f32,
                frame.height() * self.cursor_position.y as f32,
            );
            let radius = 24.0;
            let background = canvas::Path::circle(center, radius);
            frame.fill(&background, color);
        });

        vec![clock]
    }
}

impl UiProgramEvents for UiOverlay {
    fn handle_window_event(
        &self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> Vec<Self::Message> {
        match event {
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                vec![Message::CursorPosChanged(
                    winit::dpi::PhysicalPosition::new(
                        position.x / window.inner_size().width as f64,
                        position.y / window.inner_size().height as f64,
                    ),
                )]
            }
            winit::event::WindowEvent::Resized(size) => {
                vec![Message::ViewportDimsChanged((
                    (size.width as f64 / window.scale_factor()) as u32,
                    (size.height as f64 / window.scale_factor()) as u32,
                ))]
            }
            _ => vec![],
        }
    }
}

// the iced ui
impl runtime::Program for UiOverlay {
    type Message = Message;
    type Theme = iced::Theme;
    type Renderer = iced::Renderer;

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::ViewportDimsChanged(new_state) => {
                self.viewport_dims = new_state;
            }
            Message::MonitorRefreshRateChanged(new_state) => {
                self.monitor_refresh_rate = new_state;
            }
            Message::CursorPosChanged(new_state) => {
                self.cursor_position = new_state;
            }
            Message::FrameCompleted(frame_stats) => {
                self.fps_chart.on_frame_completed(
                    frame_stats.instants,
                    frame_stats.durations,
                    frame_stats.gpu_timer_query_results,
                );
                self.culling_stats = frame_stats.culling_stats;
                self.frame_counter += 1;
                self.poll_perf_dump_state();
            }
            Message::AudioSoundStatsChanged((track_path, stats)) => {
                self.audio_sound_stats.insert(
                    track_path.relative_path.to_string_lossy().to_string(),
                    stats,
                );
            }
            Message::ClosePopupMenu => self.is_showing_options_menu = false,
            Message::ExitButtonPressed => self.was_exit_button_pressed = true,
            Message::DefaultSettingsButtonPressed => {
                self.general_settings = GeneralSettings {
                    collapsed: self.general_settings.collapsed,
                    ..GeneralSettings::default()
                };
                self.camera_settings = CameraSettings {
                    collapsed: self.camera_settings.collapsed,
                    ..CameraSettings::default()
                };
                self.post_effect_settings = PostEffectSettings {
                    collapsed: self.post_effect_settings.collapsed,
                    ..PostEffectSettings::default()
                };
                self.shadow_settings = ShadowSettings {
                    collapsed: self.shadow_settings.collapsed,
                    ..ShadowSettings::default()
                };
                self.debug_settings = DebugSettings {
                    collapsed: self.debug_settings.collapsed,
                    ..DebugSettings::default()
                };
            }
            Message::GenerateProfileDump => {
                self.pending_perf_dump = Some(generate_profile_dump());
            }
            Message::TogglePopupMenu => {
                self.is_showing_options_menu = !self.is_showing_options_menu
            }
            Message::CameraPoseChanged(new_state) => {
                self.camera_pose = Some(new_state);
            }

            Message::ToggleGeneralSettingsCollapse => {
                self.general_settings.collapsed = !self.general_settings.collapsed;
            }
            Message::VsyncChanged(new_state) => {
                self.general_settings.enable_vsync = new_state;
            }
            Message::FramerateLimitTypeChanged(new_state) => {
                self.general_settings.framerate_limit_type = new_state;
            }
            Message::CustomFramerateLimitChanged(new_state) => {
                self.general_settings.custom_framerate_limit = new_state;
            }
            Message::RenderScaleChanged(new_state) => {
                self.general_settings.render_scale = new_state;
            }
            Message::ToggleDepthPrepass(new_state) => {
                self.general_settings.enable_depth_prepass = new_state;
            }

            Message::ToggleCameraSettingsCollapse => {
                self.camera_settings.collapsed = !self.camera_settings.collapsed;
            }
            Message::FovxChanged(new_state) => {
                self.camera_settings.fov_x = new_state.to_radians();
            }
            Message::NearPlaneDistanceChanged(new_state) => {
                self.camera_settings.near_plane_distance = new_state;
            }
            Message::FarPlaneDistanceChanged(new_state) => {
                self.camera_settings.far_plane_distance = new_state;
            }

            Message::TogglePostEffectSettingsCollapse => {
                self.post_effect_settings.collapsed = !self.post_effect_settings.collapsed;
            }
            Message::BloomTypeChanged(new_state) => {
                self.post_effect_settings.bloom_type = new_state;
            }
            Message::NewBloomRadiusChanged(new_state) => {
                self.post_effect_settings.new_bloom_radius = new_state;
            }
            Message::NewBloomIntensityChanged(new_state) => {
                self.post_effect_settings.new_bloom_intensity = new_state;
            }
            Message::SkyboxWeightChanged(new_state) => {
                self.post_effect_settings.skybox_weight = new_state;
            }

            Message::ToggleShadowSettingsCollapse => {
                self.shadow_settings.collapsed = !self.shadow_settings.collapsed;
            }
            Message::ShadowSmallObjectCullingSizeChanged(new_state) => {
                self.shadow_settings.shadow_small_object_culling_size_pixels = new_state
            }
            Message::SoftShadowsMaxDistanceChanged(new_state) => {
                self.shadow_settings.soft_shadows_max_distance = new_state
            }
            Message::ToggleSoftShadows(new_state) => {
                self.shadow_settings.enable_soft_shadows = new_state;
            }
            Message::ShadowBiasChanged(new_state) => {
                self.shadow_settings.shadow_bias = new_state;
            }
            Message::SoftShadowFactorChanged(new_state) => {
                self.shadow_settings.soft_shadow_factor = new_state;
            }
            Message::SoftShadowGridDimsChanged(new_state) => {
                self.shadow_settings.soft_shadow_grid_dims = new_state;
            }

            Message::ToggleDebugSettingsCollapse => {
                self.debug_settings.collapsed = !self.debug_settings.collapsed;
            }
            Message::ToggleDrawCullingFrustum(new_state) => {
                self.debug_settings.draw_culling_frustum = new_state;
                if !self.debug_settings.draw_culling_frustum {
                    self.debug_settings.culling_frustum_lock_mode = CullingFrustumLockMode::None;
                }
            }
            Message::ToggleDrawPointLightCullingFrusta(new_state) => {
                self.debug_settings.draw_point_light_culling_frusta = new_state;
            }
            Message::ToggleDrawDirectionalLightCullingFrusta(new_state) => {
                self.debug_settings.draw_directional_light_culling_frusta = new_state;
            }
            Message::ToggleShadowDebug(new_state) => {
                self.debug_settings.enable_shadow_debug = new_state;
            }
            Message::ToggleCascadeDebug(new_state) => {
                self.debug_settings.enable_cascade_debug = new_state;
            }
            Message::ToggleCullingStats(new_state) => {
                self.debug_settings.record_culling_stats = new_state;
            }
            Message::CullingFrustumLockModeChanged(new_state) => {
                self.debug_settings.culling_frustum_lock_mode = new_state;
            }
            Message::ToggleCameraPose(new_state) => {
                self.debug_settings.is_showing_camera_pose = new_state;
            }
            Message::ToggleCursorMarker(new_state) => {
                self.debug_settings.is_showing_cursor_marker = new_state;
            }
            Message::ToggleFps(new_state) => {
                self.debug_settings.is_showing_fps = new_state;
            }
            Message::ToggleFpsChart(new_state) => {
                self.debug_settings.is_showing_fps_chart = new_state;
            }
            Message::ToggleGpuSpans(new_state) => {
                self.debug_settings.is_showing_gpu_spans = new_state;
            }
            Message::ToggleAudioStats(new_state) => {
                self.debug_settings.is_showing_audio_stats = new_state;
            }
        }

        Command::none()
    }

    fn view(&self) -> Element<'_, Message, iced::Theme, iced::Renderer> {
        if self.fps_chart.recent_frame_times.is_empty() {
            return Row::new().into();
        }

        let container_style = Box::new(ContainerStyle {});

        let get_chart_line_color = |i: usize| {
            iced::Color::from_rgba8(
                FPS_CHART_LINE_COLORS[i].0,
                FPS_CHART_LINE_COLORS[i].1,
                FPS_CHART_LINE_COLORS[i].2,
                FPS_CHART_LINE_COLORS[i].3 as f32,
            )
        };

        let DebugSettings {
            is_showing_fps,
            is_showing_fps_chart,
            is_showing_gpu_spans,
            is_showing_audio_stats,
            is_showing_camera_pose,
            is_showing_cursor_marker,
            ..
        } = self.debug_settings;

        let mut rows: Vec<Element<_>> = vec![];

        if is_showing_fps {
            if let Some(avg_frame_time_millis) = self.fps_chart.avg_total_frame_time_millis {
                let mut text = text(format!(
                    "Frametime: {:.2}ms ({:.2}fps)",
                    avg_frame_time_millis,
                    1_000.0 / avg_frame_time_millis,
                ));

                if is_showing_fps_chart {
                    text = text.style(get_chart_line_color(0))
                }
                rows.push(text.into());
            }

            if self.general_settings.framerate_limit_type != FramerateLimitType::None {
                if let Some(millis) = self.fps_chart.avg_sleep_time_ms {
                    let text = text(&format!("Sleep: {:.2}ms", millis));
                    rows.push(text.into());
                }

                if let Some(millis) = self.fps_chart.avg_get_surface_time_ms {
                    let text = text(&format!("Get surface: {:.2}ms", millis));
                    rows.push(text.into());
                }
            }

            if let Some(millis) = self.fps_chart.avg_update_time_ms {
                let mut text = text(&format!("Update: {:.2}ms", millis));
                if is_showing_fps_chart {
                    text = text.style(get_chart_line_color(1))
                }
                rows.push(text.into());
            }

            if let Some(millis) = self.fps_chart.avg_render_time_ms {
                let mut text = text(&format!("Render: {:.2}ms", millis));
                if is_showing_fps_chart {
                    text = text.style(get_chart_line_color(2))
                }
                rows.push(text.into());
            }

            if let Some(millis) = self.fps_chart.avg_gpu_frame_time_ms {
                let mut text = text(&format!("GPU: {:.2}ms", millis));
                if is_showing_fps_chart {
                    text = text.style(get_chart_line_color(3))
                }
                rows.push(text.into());
            }
        }

        if SHOW_BLOOM_TYPE {
            rows.push(
                text(&format!(
                    "Bloom type: {}",
                    self.post_effect_settings.bloom_type
                ))
                .into(),
            );
        }

        if is_showing_camera_pose {
            if let Some((camera_position, camera_direction)) = self.camera_pose {
                rows.push(
                    text(&format!(
                        "Camera position:  x={:.2}, y={:.2}, z={:.2}",
                        camera_position.x, camera_position.y, camera_position.z
                    ))
                    .into(),
                );
                let two_pi = 2.0 * std::f32::consts::PI;
                let camera_horizontal = (camera_direction.horizontal + two_pi) % two_pi;
                let camera_vertical = camera_direction.vertical;
                rows.push(
                    text(&format!(
                        "Camera direction: h={:.2} ({:.2} deg), v={:.2} ({:.2} deg)",
                        camera_horizontal,
                        camera_horizontal.to_degrees(),
                        camera_vertical,
                        camera_vertical.to_degrees(),
                    ))
                    .into(),
                );
            }
        }

        if is_showing_audio_stats {
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
                rows.push(
                    text(&format!(
                        "{file_path}: {pos}{length}, buffer to {buffered_pos}"
                    ))
                    .into(),
                );
            }
        }

        if let Some(culling_stats) = &self.culling_stats {
            let text_size = 14;
            rows.push(text("Culling stats:").size(text_size).into());

            rows.push(
                text(&format!("  Time to cull: {:?}", culling_stats.time_to_cull))
                    .size(text_size)
                    .into(),
            );

            let total_count = culling_stats.total_count;

            rows.push(
                text(&format!("  Total objects: {}", total_count))
                    .size(text_size)
                    .into(),
            );
            rows.push(
                text(&format!(
                    "  Completely culled: {} ({:.2}%)",
                    culling_stats.completely_culled_count,
                    100.0 * culling_stats.completely_culled_count as f32 / total_count as f32
                ))
                .size(text_size)
                .into(),
            );
            rows.push(
                text(&format!(
                    "  Main camera: {} ({:.2}%)",
                    culling_stats.main_camera_culled_count,
                    100.0 * culling_stats.main_camera_culled_count as f32 / total_count as f32
                ))
                .size(text_size)
                .into(),
            );

            for (light_index, cascades) in culling_stats
                .directional_lights_culled_counts
                .iter()
                .enumerate()
            {
                rows.push(
                    text(&format!("  Directional light: {}", light_index))
                        .size(text_size)
                        .into(),
                );

                for (cascade_index, cascade_count) in cascades.iter().enumerate() {
                    rows.push(
                        text(&format!(
                            "    Cascade {}: {} ({:.2}%)",
                            cascade_index,
                            cascade_count,
                            100.0 * (*cascade_count as f32) / total_count as f32
                        ))
                        .size(text_size)
                        .into(),
                    );
                }
            }
            for (light_index, frusta) in culling_stats.point_light_culled_counts.iter().enumerate()
            {
                rows.push(
                    text(&format!("  Point light: {}", light_index))
                        .size(text_size)
                        .into(),
                );

                for (frustum_index, frustum_count) in frusta.iter().enumerate() {
                    rows.push(
                        text(&format!(
                            "    Frustum {}: {} ({:.2}%)",
                            frustum_index,
                            frustum_count,
                            100.0 * (*frustum_count as f32) / total_count as f32
                        ))
                        .size(text_size)
                        .into(),
                    );
                }
            }
        }

        if is_showing_gpu_spans {
            let mut avg_span_times_vec: Vec<_> =
                self.fps_chart.avg_gpu_frame_time_per_span.iter().collect();

            avg_span_times_vec.sort_by_key(|(span, _)| *span);
            avg_span_times_vec.sort_by_key(|(_span, avg_span_frame_time)| {
                (avg_span_frame_time.max(0.01) * 100.0) as u64
            });
            avg_span_times_vec.reverse();
            for (span, span_frame_time) in avg_span_times_vec {
                let msg = &format!("{span:}: {span_frame_time:.2}ms");
                rows.push(text(msg).size(14).into());
            }
        }

        if is_showing_fps_chart {
            let padding = [16, 20, 16, 0]; // top, right, bottom, left
            rows.push(
                Container::new(
                    ChartWidget::new(&self.fps_chart)
                        .width(Length::Fixed(400.0))
                        .height(Length::Fixed(300.0)),
                )
                .padding(padding)
                .into(),
            );
        }

        let padding = if rows.is_empty() { 0 } else { 8 };
        let background_content = Container::new(
            Row::new()
                .width(Length::Shrink)
                .height(Length::Shrink)
                .padding(padding)
                .push(
                    Container::new(
                        Column::with_children(rows)
                            .width(Length::Shrink)
                            .height(Length::Shrink)
                            .spacing(4),
                    )
                    .padding(padding)
                    .style(iced::theme::Container::Custom(container_style)),
                ),
        )
        .width(Length::Fill)
        .height(Length::Fill);

        let modal_content: Option<Element<_, _, _>> = self.is_showing_options_menu.then(|| {
            let big_text_size: u16 = 18;
            let small_text_size: u16 = 14;
            let checkbox_size: u16 = small_text_size * 4 / 3;
            let slider_size: u16 = small_text_size * 4 / 3;

            let collapse_title = |title, collapsed, on_press| {
                let arrow = if collapsed { ">" } else { "v" };
                button(
                    text(format!("{arrow}  {title}"))
                        .font(Font {
                            family: Family::Name(LATO_FONT_NAME),
                            weight: Weight::Bold,
                            ..Font::DEFAULT
                        })
                        .size(big_text_size),
                )
                .style(iced::theme::Button::Custom(Box::new(
                    CollapsibleButtonStyle,
                )))
                .on_press(on_press)
            };

            let mut options = Column::new()
                .spacing(4)
                .padding([6, 48, 6, 6])
                .width(Length::Fill);

            {
                let GeneralSettings {
                    collapsed,

                    enable_depth_prepass,
                    enable_vsync,
                    framerate_limit_type,
                    custom_framerate_limit,
                    render_scale,
                } = self.general_settings;

                options = options.push(collapse_title(
                    "General:",
                    collapsed,
                    Message::ToggleGeneralSettingsCollapse,
                ));

                if !collapsed {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        options = options.push(
                            checkbox("VSync", enable_vsync)
                                .size(checkbox_size)
                                .text_size(small_text_size)
                                .on_toggle(Message::VsyncChanged),
                        );
                    }

                    options = options.push(
                        text(format!(
                            "Render scale ({render_scale:.1}, {}x{})",
                            apply_render_scale(self.viewport_dims.0, render_scale),
                            apply_render_scale(self.viewport_dims.1, render_scale)
                        ))
                        .size(small_text_size),
                    );
                    options = options.push(
                        slider(0.1..=4.0, render_scale, Message::RenderScaleChanged)
                            .height(slider_size)
                            .step(0.1),
                    );

                    options = options.push(text("Framerate limit").size(small_text_size));

                    let mut framerate_limit_options = vec![];
                    for limit_type in FramerateLimitType::ALL {
                        if limit_type == FramerateLimitType::Monitor
                            && self.monitor_refresh_rate.is_none()
                        {
                            continue;
                        }

                        let value = match limit_type {
                            FramerateLimitType::None => None,
                            FramerateLimitType::Monitor => self.monitor_refresh_rate,
                            FramerateLimitType::Custom => Some(custom_framerate_limit),
                        };
                        let value_text = value
                            .map(|value| format!(" ({value:.2})"))
                            .unwrap_or_default();

                        framerate_limit_options.push(
                            radio(
                                format!("{limit_type}{value_text}"),
                                limit_type,
                                Some(framerate_limit_type),
                                Message::FramerateLimitTypeChanged,
                            )
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .into(),
                        );
                    }

                    options = options.push(
                        container(Column::with_children(framerate_limit_options).spacing(4))
                            .padding([0, 0, 8, 0]),
                    );

                    if framerate_limit_type == FramerateLimitType::Custom {
                        options = options.push(
                            slider(
                                1.0..=300.0,
                                custom_framerate_limit,
                                Message::CustomFramerateLimitChanged,
                            )
                            .height(slider_size)
                            .step(1.0),
                        );
                    }

                    options = options.push(
                        checkbox("Depth Pre-pass", enable_depth_prepass)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleDepthPrepass),
                    );
                }
            }

            {
                let CameraSettings {
                    collapsed,

                    fov_x,
                    near_plane_distance,
                    far_plane_distance,
                } = self.camera_settings;

                options = options.push(collapse_title(
                    "Camera:",
                    collapsed,
                    Message::ToggleCameraSettingsCollapse,
                ));

                if !collapsed {
                    options = options.push(
                        text(format!("Horizontal FOV: {:.4}", fov_x.to_degrees()))
                            .size(small_text_size),
                    );
                    options = options.push(
                        slider(10.0..=150.0, fov_x.to_degrees(), Message::FovxChanged)
                            .height(slider_size)
                            .step(0.1),
                    );

                    options = options.push(
                        text(format!("Near plane distance: {:.4}", near_plane_distance))
                            .size(small_text_size),
                    );
                    options = options.push(
                        slider(
                            0.0001..=1.0,
                            near_plane_distance,
                            Message::NearPlaneDistanceChanged,
                        )
                        .height(slider_size)
                        .step(0.0001),
                    );

                    options = options.push(
                        text(format!("Far plane distance: {:.4}", far_plane_distance))
                            .size(small_text_size),
                    );
                    options = options.push(
                        slider(
                            100.0..=100000.0,
                            far_plane_distance,
                            Message::FarPlaneDistanceChanged,
                        )
                        .height(slider_size)
                        .step(10.0),
                    );
                }
            }

            {
                let PostEffectSettings {
                    collapsed,

                    bloom_type,
                    new_bloom_radius,
                    new_bloom_intensity,
                    skybox_weight,
                    ..
                } = self.post_effect_settings;

                options = options.push(collapse_title(
                    "Post effects:",
                    collapsed,
                    Message::TogglePostEffectSettingsCollapse,
                ));

                if !collapsed {
                    options = options.push(text("Bloom Type").size(small_text_size));

                    let mut bloom_options = vec![];
                    for mode in BloomType::ALL {
                        bloom_options.push(
                            radio(
                                format!("{mode}"),
                                mode,
                                Some(bloom_type),
                                Message::BloomTypeChanged,
                            )
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .into(),
                        );
                    }

                    options = options.push(
                        container(Column::with_children(bloom_options).spacing(4))
                            .padding([0, 0, 8, 0]),
                    );

                    options = options.push(
                        text(format!("New Bloom Radius: {:.4}", new_bloom_radius))
                            .size(small_text_size),
                    );
                    options = options.push(
                        slider(
                            0.0001..=0.025,
                            new_bloom_radius,
                            Message::NewBloomRadiusChanged,
                        )
                        .step(0.0001),
                    );

                    options = options.push(
                        text(format!("New Bloom Intensity: {:.4}", new_bloom_intensity))
                            .size(small_text_size),
                    );
                    options = options.push(
                        slider(
                            0.001..=0.25,
                            new_bloom_intensity,
                            Message::NewBloomIntensityChanged,
                        )
                        .step(0.001),
                    );

                    options = options.push(
                        text(format!("Skybox weight: {:.5}", skybox_weight)).size(small_text_size),
                    );
                    options = options.push(
                        slider(0.0..=1.0, skybox_weight, Message::SkyboxWeightChanged).step(0.01),
                    );
                }
            }

            {
                let ShadowSettings {
                    collapsed,

                    enable_soft_shadows,
                    shadow_bias,
                    soft_shadow_factor,
                    soft_shadows_max_distance,
                    soft_shadow_grid_dims,
                    shadow_small_object_culling_size_pixels,
                } = self.shadow_settings;

                options = options.push(collapse_title(
                    "Shadows:",
                    collapsed,
                    Message::ToggleShadowSettingsCollapse,
                ));

                if !collapsed {
                    options = options.push(
                        checkbox("Enable Soft Shadows", enable_soft_shadows)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleSoftShadows),
                    );
                    options = options.push(
                        text(format!("Soft Shadow Factor: {:.6}", soft_shadow_factor))
                            .size(small_text_size),
                    );
                    options = options.push(
                        slider(
                            0.000001..=0.00015,
                            soft_shadow_factor,
                            Message::SoftShadowFactorChanged,
                        )
                        .step(0.000001),
                    );
                    options = options.push(
                        text(format!("Soft Shadow Grid Dims: {:}", soft_shadow_grid_dims))
                            .size(small_text_size),
                    );
                    options = options.push(
                        slider(
                            0..=16u32,
                            soft_shadow_grid_dims,
                            Message::SoftShadowGridDimsChanged,
                        )
                        .step(1u32),
                    );

                    options = options.push(
                        text(format!(
                            "Small Object Culling Size (Pixels): {:.4}",
                            shadow_small_object_culling_size_pixels
                        ))
                        .size(small_text_size),
                    );
                    options = options.push(
                        slider(
                            0.1..=50.0,
                            shadow_small_object_culling_size_pixels,
                            Message::ShadowSmallObjectCullingSizeChanged,
                        )
                        .step(0.1),
                    );

                    options = options.push(
                        text(format!(
                            "Soft Shadows Max Distance: {:.4}",
                            soft_shadows_max_distance
                        ))
                        .size(small_text_size),
                    );
                    options = options.push(
                        slider(
                            10.0..=500.0,
                            soft_shadows_max_distance,
                            Message::SoftShadowsMaxDistanceChanged,
                        )
                        .step(1.0),
                    );

                    options = options
                        .push(text(format!("Bias: {:.5}", shadow_bias)).size(small_text_size));
                    options = options.push(
                        slider(
                            MIN_SHADOW_MAP_BIAS..=0.005,
                            shadow_bias,
                            Message::ShadowBiasChanged,
                        )
                        .step(0.00001),
                    );
                }
            }

            {
                let DebugSettings {
                    collapsed,

                    draw_culling_frustum,
                    draw_point_light_culling_frusta,
                    draw_directional_light_culling_frusta,
                    enable_shadow_debug,
                    enable_cascade_debug,
                    record_culling_stats,
                    culling_frustum_lock_mode,
                    ..
                } = self.debug_settings;

                options = options.push(collapse_title(
                    "Debug:",
                    collapsed,
                    Message::ToggleDebugSettingsCollapse,
                ));

                if !collapsed {
                    options = options.push(
                        checkbox("FPS", is_showing_fps)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleFps),
                    );

                    options = options.push(
                        checkbox("FPS Chart", is_showing_fps_chart)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleFpsChart),
                    );

                    options = options.push(
                        checkbox("Culling Stats", record_culling_stats)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleCullingStats),
                    );

                    if self
                        .fps_chart
                        .recent_frame_times
                        .iter()
                        .any(|(_, _, gpu_duration)| gpu_duration.is_some())
                    {
                        options = options.push(
                            checkbox("Detailed GPU Frametimes", is_showing_gpu_spans)
                                .size(checkbox_size)
                                .text_size(small_text_size)
                                .on_toggle(Message::ToggleGpuSpans),
                        );
                    }

                    options = options.push(
                        checkbox("Camera Pose", is_showing_camera_pose)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleCameraPose),
                    );

                    options = options.push(
                        checkbox("Cascade Debug", enable_cascade_debug)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleCascadeDebug),
                    );

                    options = options.push(
                        checkbox("Shadow Debug", enable_shadow_debug)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleShadowDebug),
                    );

                    options = options.push(
                        checkbox("Frustum Culling Overlay", draw_culling_frustum)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleDrawCullingFrustum),
                    );
                    if draw_culling_frustum {
                        options = options.push(text("Lock Culling Frustum").size(small_text_size));

                        let mut culling_lock_options = vec![];
                        for mode in CullingFrustumLockMode::ALL {
                            culling_lock_options.push(
                                radio(
                                    format!("{mode}"),
                                    mode,
                                    Some(culling_frustum_lock_mode),
                                    Message::CullingFrustumLockModeChanged,
                                )
                                .size(checkbox_size)
                                .text_size(small_text_size)
                                .into(),
                            );
                        }

                        options = options.push(
                            container(Column::with_children(culling_lock_options).spacing(4))
                                .padding([0, 0, 8, 0]),
                        );
                    }

                    options = options.push(
                        checkbox(
                            "Point Light Frustum Culling Overlay",
                            draw_point_light_culling_frusta,
                        )
                        .size(checkbox_size)
                        .text_size(small_text_size)
                        .on_toggle(Message::ToggleDrawPointLightCullingFrusta),
                    );
                    options = options.push(
                        checkbox(
                            "Directional Light Frustum Culling Overlay",
                            draw_directional_light_culling_frusta,
                        )
                        .size(checkbox_size)
                        .text_size(small_text_size)
                        .on_toggle(Message::ToggleDrawDirectionalLightCullingFrusta),
                    );

                    options = options.push(
                        checkbox("Audio Stats", is_showing_audio_stats)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleAudioStats),
                    );

                    options = options.push(
                        checkbox("Cursor Marker", is_showing_cursor_marker)
                            .size(checkbox_size)
                            .text_size(small_text_size)
                            .on_toggle(Message::ToggleCursorMarker),
                    );
                }
            }

            let mut bottom_buttons = Column::new().spacing(8).padding([8, 0, 0, 0]);

            if can_generate_profile_dump() {
                if let Some(pending_perf_dump) = &self.pending_perf_dump {
                    let (message, color) = if self.perf_dump_completion_time.is_some() {
                        match *pending_perf_dump.lock() {
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
                    bottom_buttons = bottom_buttons.push(text(message).style(color));
                } else {
                    bottom_buttons = bottom_buttons.push(
                        button(
                            text("Generate Profile Dump")
                                .size(small_text_size)
                                .horizontal_alignment(Horizontal::Center),
                        )
                        .width(Length::Shrink)
                        .on_press(Message::GenerateProfileDump),
                    );
                }
            }

            bottom_buttons = bottom_buttons.push(
                button(
                    text("Reset Defaults")
                        .size(small_text_size)
                        .horizontal_alignment(Horizontal::Center),
                )
                .width(Length::Shrink)
                .on_press(Message::DefaultSettingsButtonPressed),
            );

            bottom_buttons = bottom_buttons.push(
                button(
                    text("Exit Game")
                        .size(small_text_size)
                        .horizontal_alignment(Horizontal::Center),
                )
                .width(Length::Shrink)
                .on_press(Message::ExitButtonPressed),
            );

            options = options.push(bottom_buttons);

            iced_aw::Card::new(
                text("Options").font(Font::with_name(PACIFICO_FONT_NAME)),
                scrollable(options)
                    .height(Length::Fixed(self.viewport_dims.1 as f32 * 0.75 - 50.0)),
            )
            .max_width(300.0)
            .on_close(Message::ClosePopupMenu)
            .into()
        });

        let modal_background: Element<_> = if is_showing_cursor_marker {
            let overlay = {
                let canvas: canvas::Canvas<&Self, Message, iced::Theme, iced::Renderer> =
                    canvas(self as &Self)
                        .width(Length::Fill)
                        .height(Length::Fill);

                Element::from(
                    container(canvas)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .padding(0),
                )
            };
            floating_element(background_content, overlay)
                .anchor(floating_element::Anchor::NorthWest)
                .into()
        } else {
            background_content.into()
        };

        Modal::new(modal_background, modal_content)
            .style(modal::ModalStyles::Custom(Rc::new(ModalStyle)))
            .into()
    }
}
