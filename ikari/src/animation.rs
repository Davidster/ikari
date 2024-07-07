use std::{
    cmp::Ordering,
    ops::{Add, Mul},
};

use glam::f32::{Quat, Vec3};

use crate::scene::{GameNodeId, Scene};

#[derive(Clone, Debug, PartialEq)]
pub struct Animation {
    pub name: Option<String>,
    pub length_seconds: f32,
    pub channels: Vec<AnimationChannel>,
    pub state: AnimationState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnimationChannel {
    pub node_id: GameNodeId,
    // pub property: gltf::animation::Property,
    // pub interpolation_type: gltf::animation::Interpolation,
    pub keyframe_timings: Vec<f32>,
    // pub keyframe_values_u8: Vec<u8>,
    // pub interpolation_type: AnimationInterpolation,
    pub keyframes: AnimationKeyframes,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AnimationKeyframes {
    TranslationLinear(Vec<Vec3>),
    TranslationStep(Vec<Vec3>),
    TranslationCubic(Vec<[Vec3; 3]>),
    ScaleLinear(Vec<Vec3>),
    ScaleStep(Vec<Vec3>),
    ScaleCubic(Vec<[Vec3; 3]>),
    RotationLinear(Vec<Quat>),
    RotationStep(Vec<Quat>),
    RotationCubic(Vec<[Quat; 3]>),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AnimationState {
    pub speed: f32,
    pub current_time_seconds: f32,
    pub is_playing: bool,
    pub loop_type: LoopType,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopType {
    #[allow(dead_code)]
    Once,
    #[allow(dead_code)]
    Wrap,
    #[allow(dead_code)]
    PingPong,
}

impl Default for AnimationState {
    fn default() -> Self {
        Self {
            speed: 1.0,
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

pub(crate) fn step_animations(scene: &mut Scene, delta_time_seconds: f64) {
    enum Op {
        SetTranslation(Vec3),
        SetScale(Vec3),
        SetRotation(Quat),
    }

    let mut ops: Vec<(GameNodeId, Op)> = Vec::new();
    for animation in scene.animations.iter_mut() {
        let state = &mut animation.state;
        if !state.is_playing {
            continue;
        }
        state.current_time_seconds += delta_time_seconds as f32 * state.speed;
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
            let (previous_keyframe, next_keyframe) =
                get_nearby_keyframes(&channel.keyframe_timings, animation_time_seconds);

            let op = match previous_keyframe {
                Some(previous_keyframe) => {
                    let (next_keyframe, interpolation_factor) = match next_keyframe {
                        Some(next_keyframe) => (
                            next_keyframe,
                            (animation_time_seconds - previous_keyframe.time)
                                / (next_keyframe.time - previous_keyframe.time),
                        ),
                        // no next keyframe means we snap to the last keyframe of the animation
                        None => (previous_keyframe, 1.0),
                    };

                    match &channel.keyframes {
                        AnimationKeyframes::TranslationLinear(keyframe_values) => {
                            Op::SetTranslation(
                                keyframe_values[previous_keyframe.index].lerp(
                                    keyframe_values[next_keyframe.index],
                                    interpolation_factor,
                                ),
                            )
                        }
                        AnimationKeyframes::TranslationStep(keyframe_values) => {
                            Op::SetTranslation(keyframe_values[previous_keyframe.index])
                        }
                        AnimationKeyframes::TranslationCubic(keyframe_values) => {
                            Op::SetTranslation(do_cubic_interpolation(
                                keyframe_values[previous_keyframe.index],
                                keyframe_values[next_keyframe.index],
                                next_keyframe.time - previous_keyframe.time,
                                interpolation_factor,
                            ))
                        }
                        AnimationKeyframes::ScaleLinear(keyframe_values) => Op::SetScale(
                            keyframe_values[previous_keyframe.index]
                                .lerp(keyframe_values[next_keyframe.index], interpolation_factor),
                        ),
                        AnimationKeyframes::ScaleStep(keyframe_values) => {
                            Op::SetScale(keyframe_values[previous_keyframe.index])
                        }
                        AnimationKeyframes::ScaleCubic(keyframe_values) => {
                            Op::SetScale(do_cubic_interpolation(
                                keyframe_values[previous_keyframe.index],
                                keyframe_values[next_keyframe.index],
                                next_keyframe.time - previous_keyframe.time,
                                interpolation_factor,
                            ))
                        }
                        AnimationKeyframes::RotationLinear(keyframe_values) => Op::SetRotation(
                            keyframe_values[previous_keyframe.index]
                                .slerp(keyframe_values[next_keyframe.index], interpolation_factor),
                        ),
                        AnimationKeyframes::RotationStep(keyframe_values) => {
                            Op::SetRotation(keyframe_values[previous_keyframe.index])
                        }
                        AnimationKeyframes::RotationCubic(keyframe_values) => {
                            Op::SetRotation(do_cubic_interpolation(
                                keyframe_values[previous_keyframe.index],
                                keyframe_values[next_keyframe.index],
                                next_keyframe.time - previous_keyframe.time,
                                interpolation_factor,
                            ))
                        }
                    }
                }
                // no previous keyframe means we snap to the first keyframe of the animation
                None => match &channel.keyframes {
                    AnimationKeyframes::TranslationLinear(keyframe_values)
                    | AnimationKeyframes::TranslationStep(keyframe_values) => {
                        Op::SetTranslation(keyframe_values[0])
                    }
                    AnimationKeyframes::TranslationCubic(keyframe_values) => {
                        Op::SetTranslation(keyframe_values[0][1])
                    }
                    AnimationKeyframes::ScaleLinear(keyframe_values)
                    | AnimationKeyframes::ScaleStep(keyframe_values) => {
                        Op::SetScale(keyframe_values[0])
                    }
                    AnimationKeyframes::ScaleCubic(keyframe_values) => {
                        Op::SetScale(keyframe_values[0][1])
                    }
                    AnimationKeyframes::RotationLinear(keyframe_values)
                    | AnimationKeyframes::RotationStep(keyframe_values) => {
                        Op::SetRotation(keyframe_values[0])
                    }
                    AnimationKeyframes::RotationCubic(keyframe_values) => {
                        Op::SetRotation(keyframe_values[0][1])
                    }
                },
            };

            ops.push((channel.node_id, op));
        }
    }
    for (node_id, op) in ops {
        if let Some(node) = scene.get_node_mut(node_id) {
            let transform = &mut node.transform;
            match op {
                Op::SetTranslation(translation) => {
                    transform.set_position(translation);
                }
                Op::SetScale(scale) => {
                    transform.set_scale(scale);
                }
                Op::SetRotation(rotation) => {
                    transform.set_rotation(rotation);
                }
            }
        }
    }
}

fn get_nearby_keyframes(
    keyframe_times: &[f32],
    animation_time_seconds: f32,
) -> (Option<KeyframeTime>, Option<KeyframeTime>) {
    let (previous_keyframe_index, next_keyframe_index) =
        match keyframe_times.binary_search_by(|val| {
            val.partial_cmp(&animation_time_seconds)
                .unwrap_or(Ordering::Equal)
        }) {
            Ok(index) => {
                if index >= keyframe_times.len() - 1 {
                    (Some(keyframe_times.len() - 1), None)
                } else {
                    (Some(index), Some(index + 1))
                }
            }
            Err(index) => {
                if index == 0 {
                    (None, None)
                } else if index == keyframe_times.len() {
                    (Some(keyframe_times.len() - 1), None)
                } else {
                    (Some(index - 1), Some(index))
                }
            }
        };

    (
        previous_keyframe_index.map(|previous_keyframe_index| KeyframeTime {
            index: previous_keyframe_index,
            time: keyframe_times[previous_keyframe_index],
        }),
        next_keyframe_index.map(|next_keyframe_index| KeyframeTime {
            index: next_keyframe_index,
            time: keyframe_times[next_keyframe_index],
        }),
    )
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
