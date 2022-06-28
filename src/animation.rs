use std::{
    collections::HashMap,
    ops::{Add, Mul},
};

use anyhow::bail;
use anyhow::Result;
use cgmath::{Quaternion, Vector3};

use super::*;

#[derive(Debug)]
pub struct Animation {
    // source_animation: gltf::animation::Animation<'a>,
    length_seconds: f32,
    channels: Vec<Channel>,
}

#[derive(Debug)]
pub struct Channel {
    // source_channel: gltf::animation::Channel<'a>,
    node_index: usize,
    property: gltf::animation::Property,
    interpolation_type: gltf::animation::Interpolation,
    keyframe_timings: Vec<f32>,
    keyframe_values_u8: Vec<u8>,
}

#[derive(Copy, Clone, Debug)]
struct KeyframeTime {
    index: usize,
    time: f32,
}

pub fn update_node_transforms_at_moment(scene: &mut Scene, global_time_seconds: f32) -> Result<()> {
    for animation in scene.animations.iter() {
        for channel in animation.channels.iter() {
            let transform = &mut scene.nodes[channel.node_index].transform;
            let animation_time_seconds = global_time_seconds % animation.length_seconds;
            let (previous_key_frame, next_key_frame) =
                get_nearby_keyframes(&channel.keyframe_timings, animation_time_seconds);
            match channel.property {
                gltf::animation::Property::Translation => {
                    transform.set_position(get_vec3_at_moment(
                        channel,
                        animation_time_seconds,
                        previous_key_frame,
                        next_key_frame,
                    )?);
                }
                gltf::animation::Property::Scale => {
                    transform.set_scale(get_vec3_at_moment(
                        channel,
                        animation_time_seconds,
                        previous_key_frame,
                        next_key_frame,
                    )?);
                }
                gltf::animation::Property::Rotation => {
                    transform.set_rotation(get_quat_at_moment(
                        channel,
                        animation_time_seconds,
                        previous_key_frame,
                        next_key_frame,
                    )?);
                }
                _ => {}
            };
        }
    }
    Ok(())
}

pub fn get_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<Animation>> {
    document
        .animations()
        .map(|animation| {
            let channel_timings: Vec<_> = animation
                .channels()
                .map(|channel| get_keyframe_times(&channel.sampler(), buffers))
                .collect::<Result<Vec<_>, _>>()?;
            let length_seconds = *channel_timings
                .iter()
                .map(|keyframe_times| keyframe_times.last().unwrap())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let channels: Vec<_> = animation
                .channels()
                .enumerate()
                .map(|(channel_index, channel)| {
                    validate_channel_data_type(&channel)?;
                    let sampler = channel.sampler();
                    let accessor = sampler.output();
                    anyhow::Ok(Channel {
                        node_index: channel.target().node().index(),
                        property: channel.target().property(),
                        interpolation_type: sampler.interpolation(),
                        keyframe_timings: channel_timings[channel_index].clone(),
                        keyframe_values_u8: get_buffer_slice_from_accessor(accessor, buffers),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            anyhow::Ok(Animation {
                length_seconds,
                channels,
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

fn get_vec3_at_moment(
    channel: &Channel,
    animation_time_seconds: f32,
    previous_keyframe: Option<KeyframeTime>,
    next_keyframe: Option<KeyframeTime>,
) -> Result<Vector3<f32>> {
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

    // TODO: inline
    let t = animation_time_seconds;

    Ok(match previous_keyframe {
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
    })
}

fn get_quat_at_moment(
    channel: &Channel,
    animation_time_seconds: f32,
    previous_keyframe: Option<KeyframeTime>,
    next_keyframe: Option<KeyframeTime>,
) -> Result<Quaternion<f32>> {
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

    Ok(match previous_keyframe {
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
    })
}

fn validate_channel_data_type(channel: &gltf::animation::Channel) -> Result<()> {
    let accessor = channel.sampler().output();
    let data_type = accessor.data_type();
    let dimensions = accessor.dimensions();
    match channel.target().property() {
        gltf::animation::Property::Translation | gltf::animation::Property::Scale => {
            if dimensions != gltf::accessor::Dimensions::Vec3 {
                bail!("Expected vec3 data but found: {:?}", dimensions);
            }
            if data_type != gltf::accessor::DataType::F32 {
                bail!("Expected f32 data but found: {:?}", data_type);
            }
        }
        gltf::animation::Property::Rotation => {
            if dimensions != gltf::accessor::Dimensions::Vec4 {
                bail!("Expected vec4 data but found: {:?}", dimensions);
            }
            if data_type != gltf::accessor::DataType::F32 {
                bail!("Expected f32 data but found: {:?}", data_type);
            }
        }
        gltf::animation::Property::MorphTargetWeights => {
            bail!("MorphTargetWeights not supported")
        }
    };
    Ok(())
}

fn get_keyframe_times(
    sampler: &gltf::animation::Sampler,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<f32>> {
    let keyframe_times = {
        let accessor = sampler.input();
        let data_type = accessor.data_type();
        let dimensions = accessor.dimensions();
        if dimensions != gltf::accessor::Dimensions::Scalar {
            bail!("Expected scalar data but found: {:?}", dimensions);
        }
        if data_type != gltf::accessor::DataType::F32 {
            bail!("Expected f32 data but found: {:?}", data_type);
        }
        let result_u8 = get_buffer_slice_from_accessor(accessor, buffers);
        bytemuck::cast_slice::<_, f32>(&result_u8).to_vec()
    };

    Ok(keyframe_times)
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
    f32: Mul<f32, Output = f32> + Add<f32, Output = f32>,
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

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct ChannelPropertyStr<'a>(&'a str);

impl From<gltf::animation::Property> for ChannelPropertyStr<'_> {
    fn from(prop: gltf::animation::Property) -> Self {
        Self(match prop {
            gltf::animation::Property::Translation => "Translation",
            gltf::animation::Property::Scale => "Scale",
            gltf::animation::Property::Rotation => "Rotation",
            gltf::animation::Property::MorphTargetWeights => "MorphTargetWeights",
        })
    }
}

pub fn validate_animation_property_counts(scene: &Scene, logger: &mut Logger) {
    let property_counts: HashMap<(usize, ChannelPropertyStr), usize> = scene
        .source_asset
        .document
        .animations()
        .flat_map(|animation| animation.channels())
        .fold(HashMap::new(), |mut acc, channel| {
            let count = acc
                .entry((
                    channel.target().node().index(),
                    channel.target().property().into(),
                ))
                .or_insert(0);
            *count += 1;
            acc
        });
    for ((node_index, property), count) in property_counts {
        if count > 1 {
            logger.log(&format!(
                "Warning: expected no more than 1 animated property but found {:?} (node_index={:?}, node_name={:?}, property={:?})",
                count,
                node_index,
                scene
                    .source_asset
                    .document
                    .nodes()
                    .find(|node| node.index() == node_index)
                    .and_then(|node| node.name()),
                property.0
            ))
        }
    }
}
