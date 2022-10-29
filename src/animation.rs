use std::ops::{Add, Mul};

use cgmath::{Quaternion, Vector3};

use super::*;

#[derive(Debug)]
pub struct Animation {
    pub length_seconds: f32,
    pub speed: f32,
    pub channels: Vec<Channel>,
    pub state: AnimationState,
}

#[derive(Debug)]
pub struct Channel {
    pub node_id: GameNodeId,
    pub property: gltf::animation::Property,
    pub interpolation_type: gltf::animation::Interpolation,
    pub keyframe_timings: Vec<f32>,
    pub keyframe_values_u8: Vec<u8>,
}

#[derive(Copy, Clone, Debug)]
pub struct AnimationState {
    pub current_time_seconds: f32,
    pub is_playing: bool,
    pub loop_type: LoopType,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopType {
    Once,
    Wrap,
    #[allow(dead_code)]
    PingPong,
}

impl Default for AnimationState {
    fn default() -> Self {
        Self {
            current_time_seconds: 0.0,
            is_playing: false,
            loop_type: LoopType::Once,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct KeyframeTime {
    index: usize,
    time: f32,
}

pub fn step_animations(scene: &mut Scene, delta_time_seconds: f32) {
    pub enum Op {
        Translation(Vector3<f32>),
        Scale(Vector3<f32>),
        Rotation(Quaternion<f32>),
    }

    let mut ops: Vec<(GameNodeId, Op)> = Vec::new();
    for animation in scene.animations.iter_mut() {
        let state = &mut animation.state;
        if !state.is_playing {
            continue;
        }
        state.current_time_seconds += delta_time_seconds * animation.speed;
        if state.loop_type == LoopType::Once
            && state.current_time_seconds > animation.length_seconds
        {
            state.current_time_seconds = 0.0;
            state.is_playing = false;
        }
        let animation_time_seconds = match state.loop_type {
            LoopType::PingPong => {
                let forwards =
                    (state.current_time_seconds / animation.length_seconds).floor() as i32 % 2 == 0;
                if forwards {
                    state.current_time_seconds % animation.length_seconds
                } else {
                    animation.length_seconds - state.current_time_seconds % animation.length_seconds
                }
            }
            _ => state.current_time_seconds % animation.length_seconds,
        };

        for channel in animation.channels.iter() {
            let (previous_key_frame, next_key_frame) =
                get_nearby_keyframes(&channel.keyframe_timings, animation_time_seconds);
            if let Some(op) = match channel.property {
                gltf::animation::Property::Translation => {
                    Some(Op::Translation(get_vec3_at_moment(
                        channel,
                        animation_time_seconds,
                        previous_key_frame,
                        next_key_frame,
                    )))
                }
                gltf::animation::Property::Scale => Some(Op::Scale(get_vec3_at_moment(
                    channel,
                    animation_time_seconds,
                    previous_key_frame,
                    next_key_frame,
                ))),
                gltf::animation::Property::Rotation => Some(Op::Rotation(get_quat_at_moment(
                    channel,
                    animation_time_seconds,
                    previous_key_frame,
                    next_key_frame,
                ))),
                _ => None,
            } {
                ops.push((channel.node_id, op));
            }
        }
    }
    for (node_id, op) in ops {
        if let Some(node) = scene.get_node_mut(node_id) {
            let transform = &mut node.transform;
            match op {
                Op::Translation(translation) => {
                    transform.set_position(translation);
                }
                Op::Scale(scale) => {
                    transform.set_scale(scale);
                }
                Op::Rotation(rotation) => {
                    transform.set_rotation(rotation);
                }
            }
        }
    }
}

fn get_vec3_at_moment(
    channel: &Channel,
    animation_time_seconds: f32,
    previous_keyframe: Option<KeyframeTime>,
    next_keyframe: Option<KeyframeTime>,
) -> Vector3<f32> {
    let get_basic_keyframe_values = || {
        bytemuck::cast_slice::<_, [f32; 3]>(&channel.keyframe_values_u8)
            .to_vec()
            .iter()
            .copied()
            .map(Vector3::from)
            .collect::<Vec<_>>()
    };
    let get_cubic_keyframe_values = || {
        bytemuck::cast_slice::<_, [[f32; 3]; 3]>(&channel.keyframe_values_u8)
            .to_vec()
            .iter()
            .copied()
            .map(|kf| {
                [
                    Vector3::from(kf[0]), // in-tangent
                    Vector3::from(kf[1]), // value
                    Vector3::from(kf[2]), // out-tangent
                ]
            })
            .collect::<Vec<_>>()
    };

    match previous_keyframe {
        Some(previous_keyframe) => {
            let (next_keyframe, interpolation_factor) = match next_keyframe {
                Some(next_keyframe) => (
                    next_keyframe,
                    (animation_time_seconds - previous_keyframe.time)
                        / (next_keyframe.time - previous_keyframe.time),
                ),
                None => (previous_keyframe, 1.0),
            };

            match channel.interpolation_type {
                gltf::animation::Interpolation::Linear => {
                    let keyframe_values = get_basic_keyframe_values();
                    let previous_keyframe_value = keyframe_values[previous_keyframe.index];
                    let next_keyframe_value = keyframe_values[next_keyframe.index];
                    lerp_vec(
                        previous_keyframe_value,
                        next_keyframe_value,
                        interpolation_factor,
                    )
                }
                gltf::animation::Interpolation::Step => {
                    let keyframe_values = get_basic_keyframe_values();
                    keyframe_values[previous_keyframe.index]
                }
                gltf::animation::Interpolation::CubicSpline => {
                    let keyframe_values = get_cubic_keyframe_values();
                    let previous_keyframe_value = keyframe_values[previous_keyframe.index];
                    let next_keyframe_value = keyframe_values[next_keyframe.index];
                    let keyframe_length = next_keyframe.time - previous_keyframe.time;

                    do_cubic_interpolation(
                        previous_keyframe_value,
                        next_keyframe_value,
                        keyframe_length,
                        interpolation_factor,
                    )
                }
            }
        }
        None => match channel.interpolation_type {
            gltf::animation::Interpolation::Linear => get_basic_keyframe_values()[0],
            gltf::animation::Interpolation::Step => get_basic_keyframe_values()[0],
            gltf::animation::Interpolation::CubicSpline => get_cubic_keyframe_values()[0][1],
        },
    }
}

fn get_quat_at_moment(
    channel: &Channel,
    animation_time_seconds: f32,
    previous_keyframe: Option<KeyframeTime>,
    next_keyframe: Option<KeyframeTime>,
) -> Quaternion<f32> {
    let get_basic_keyframe_values = || {
        bytemuck::cast_slice::<_, [f32; 4]>(&channel.keyframe_values_u8)
            .to_vec()
            .iter()
            .copied()
            .map(Quaternion::from)
            .collect::<Vec<_>>()
    };
    let get_cubic_keyframe_values = || {
        bytemuck::cast_slice::<_, [[f32; 4]; 3]>(&channel.keyframe_values_u8)
            .to_vec()
            .iter()
            .copied()
            .map(|kf| {
                [
                    Quaternion::from(kf[0]), // in-tangent
                    Quaternion::from(kf[1]), // value
                    Quaternion::from(kf[2]), // out-tangent
                ]
            })
            .collect::<Vec<_>>()
    };

    let t = animation_time_seconds;

    match previous_keyframe {
        Some(previous_keyframe) => {
            let (next_keyframe, interpolation_factor) = match next_keyframe {
                Some(next_keyframe) => (
                    next_keyframe,
                    (t - previous_keyframe.time) / (next_keyframe.time - previous_keyframe.time),
                ),
                None => (previous_keyframe, 1.0),
            };
            match channel.interpolation_type {
                gltf::animation::Interpolation::Linear => {
                    let keyframe_values = get_basic_keyframe_values();
                    let previous_keyframe_value = keyframe_values[previous_keyframe.index];
                    let next_keyframe_value = keyframe_values[next_keyframe.index];
                    previous_keyframe_value.slerp(next_keyframe_value, interpolation_factor)
                }
                gltf::animation::Interpolation::Step => {
                    let keyframe_values = get_basic_keyframe_values();
                    keyframe_values[previous_keyframe.index]
                }
                gltf::animation::Interpolation::CubicSpline => {
                    let keyframe_values = get_cubic_keyframe_values();
                    let previous_keyframe_value = keyframe_values[previous_keyframe.index];
                    let next_keyframe_value = keyframe_values[next_keyframe.index];
                    let keyframe_length = next_keyframe.time - previous_keyframe.time;

                    do_cubic_interpolation(
                        previous_keyframe_value,
                        next_keyframe_value,
                        keyframe_length,
                        interpolation_factor,
                    )
                    .normalize()
                }
            }
        }
        None => match channel.interpolation_type {
            gltf::animation::Interpolation::Linear => get_basic_keyframe_values()[0],
            gltf::animation::Interpolation::Step => get_basic_keyframe_values()[0],
            gltf::animation::Interpolation::CubicSpline => get_cubic_keyframe_values()[0][1],
        },
    }
}

fn get_nearby_keyframes(
    keyframe_times: &[f32],
    animation_time_seconds: f32,
) -> (Option<KeyframeTime>, Option<KeyframeTime>) {
    let previous_keyframe = keyframe_times
        .iter()
        .enumerate()
        .filter(|(_, keyframe_time)| **keyframe_time <= animation_time_seconds)
        .last()
        .map(|(index, time)| KeyframeTime { index, time: *time });
    let next_keyframe = keyframe_times
        .iter()
        .enumerate()
        .find(|(_, keyframe_time)| **keyframe_time > animation_time_seconds)
        .map(|(index, time)| KeyframeTime { index, time: *time });
    (previous_keyframe, next_keyframe)
}

// see https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#appendix-c-interpolation
fn do_cubic_interpolation<T>(
    previous_keyframe_value: [T; 3],
    next_keyframe_value: [T; 3],
    keyframe_length: f32,
    interpolation_factor: f32,
) -> T
where
    T: Copy + Mul<f32, Output = T> + Add<T, Output = T>,
{
    // copy names from math formula:
    let t = interpolation_factor;
    let t_2 = t * t;
    let t_3 = t_2 * t;
    let v_k = previous_keyframe_value[1];
    let t_d = keyframe_length;
    let b_k = previous_keyframe_value[2];
    let v_k_1 = next_keyframe_value[1];
    let a_k_1 = next_keyframe_value[0];
    v_k * (2.0 * t_3 - 3.0 * t_2 + 1.0)
        + b_k * t_d * (t_3 - 2.0 * t_2 + t)
        + v_k_1 * (-2.0 * t_3 + 3.0 * t_2)
        + a_k_1 * t_d * (t_3 - t_2)
}
