use std::ops::{Add, Mul};

use cgmath::{Quaternion, Vector3, Vector4};

use super::*;

pub fn get_node_transforms_at_moment(
    scene: &mut Scene,
    moment_seconds: f32,
) -> Vec<crate::transform::Transform> {
    let buffers = &scene.source_asset.buffers;
    scene
        .node_transforms
        .iter()
        .enumerate()
        .map(|(node_index, current_transform)| {
            // TODO: check if there's more than one channel per property, this should probably throw an error

            let mut new_transform = *current_transform;

            for channel in scene
                .source_asset
                .document
                .animations()
                .flat_map(|animation| animation.channels())
                .filter(|channel| channel.target().node().index() == node_index)
            {
                match channel.target().property() {
                    gltf::animation::Property::Translation => {
                        new_transform.set_position(get_vec3_at_moment(
                            &channel.sampler(),
                            buffers,
                            moment_seconds,
                        ));
                    }
                    gltf::animation::Property::Scale => {
                        new_transform.set_scale(get_vec3_at_moment(
                            &channel.sampler(),
                            buffers,
                            moment_seconds,
                        ));
                    }
                    gltf::animation::Property::Rotation => {
                        new_transform.set_rotation(get_quat_at_moment(
                            &channel.sampler(),
                            buffers,
                            moment_seconds,
                        ));
                    }
                    _ => {}
                }
            }

            new_transform
        })
        .collect()
}

// TODO: this should not return an option
fn get_vec3_at_moment(
    vec3_sampler: &gltf::animation::Sampler,
    buffers: &[gltf::buffer::Data],
    global_time_seconds: f32,
) -> Vector3<f32> {
    let keyframe_values_u8 = {
        let accessor = vec3_sampler.output();
        let data_type = accessor.data_type();
        let dimensions = accessor.dimensions();
        if dimensions != gltf::accessor::Dimensions::Vec3 {
            // TODO: use error instead of panic
            panic!("Expected vec3 data but found: {:?}", dimensions);
        }
        if data_type != gltf::accessor::DataType::F32 {
            // TODO: use error instead of panic
            panic!("Expected f32 data but found: {:?}", data_type);
        }
        get_buffer_slice_from_accessor(accessor, buffers)
    };
    let get_basic_keyframe_values = || {
        bytemuck::cast_slice::<_, [f32; 3]>(&keyframe_values_u8)
            .to_vec()
            .iter()
            .copied()
            .map(Vector3::from)
            .collect::<Vec<_>>()
    };

    let (keyframe_times, animation_length_seconds) = get_keyframe_times(vec3_sampler, buffers);
    let animation_time_seconds = global_time_seconds % animation_length_seconds;
    let t = animation_time_seconds;

    let previous_keyframe = keyframe_times
        .iter()
        .enumerate()
        .filter(|(_, keyframe_time)| **keyframe_time <= t)
        .last()
        .unwrap();
    let next_keyframe = keyframe_times
        .iter()
        .enumerate()
        .find(|(_, keyframe_time)| **keyframe_time > t)
        .unwrap();
    let interpolation_factor = (t - previous_keyframe.1) / (next_keyframe.1 - previous_keyframe.1);

    match vec3_sampler.interpolation() {
        gltf::animation::Interpolation::Linear => {
            let keyframe_values = get_basic_keyframe_values();
            let previous_keyframe_value = keyframe_values[previous_keyframe.0];
            let next_keyframe_value = keyframe_values[next_keyframe.0];
            lerp_vec(
                previous_keyframe_value,
                next_keyframe_value,
                interpolation_factor,
            )
        }
        gltf::animation::Interpolation::Step => {
            let keyframe_values = get_basic_keyframe_values();
            keyframe_values[previous_keyframe.0]
        }
        gltf::animation::Interpolation::CubicSpline => {
            let keyframe_values = bytemuck::cast_slice::<_, [[f32; 3]; 3]>(&keyframe_values_u8)
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
                .collect::<Vec<_>>();
            let previous_keyframe_value = keyframe_values[previous_keyframe.0];
            let next_keyframe_value = keyframe_values[next_keyframe.0];
            let keyframe_length = next_keyframe.1 - previous_keyframe.1;

            do_cubic_interpolation(
                previous_keyframe_value,
                next_keyframe_value,
                keyframe_length,
                interpolation_factor,
            )
        }
    }
}

fn get_quat_at_moment(
    quat_sampler: &gltf::animation::Sampler,
    buffers: &[gltf::buffer::Data],
    global_time_seconds: f32,
) -> Quaternion<f32> {
    let keyframe_values_u8 = {
        let accessor = quat_sampler.output();
        let data_type = accessor.data_type();
        let dimensions = accessor.dimensions();
        if dimensions != gltf::accessor::Dimensions::Vec4 {
            // TODO: use error instead of panic
            panic!("Expected vec4 data but found: {:?}", dimensions);
        }
        if data_type != gltf::accessor::DataType::F32 {
            // TODO: use error instead of panic
            panic!("Expected f32 data but found: {:?}", data_type);
        }
        get_buffer_slice_from_accessor(accessor, buffers)
    };
    let get_basic_keyframe_values = || {
        bytemuck::cast_slice::<_, [f32; 4]>(&keyframe_values_u8)
            .to_vec()
            .iter()
            .copied()
            .map(Quaternion::from)
            .collect::<Vec<_>>()
    };

    let (keyframe_times, animation_length_seconds) = get_keyframe_times(quat_sampler, buffers);
    let animation_time_seconds = global_time_seconds % animation_length_seconds;
    let t = animation_time_seconds;

    let previous_keyframe = keyframe_times
        .iter()
        .enumerate()
        .filter(|(_, keyframe_time)| **keyframe_time <= t)
        .last()
        .unwrap();
    let next_keyframe = keyframe_times
        .iter()
        .enumerate()
        .find(|(_, keyframe_time)| **keyframe_time > t)
        .unwrap();
    let interpolation_factor = (t - previous_keyframe.1) / (next_keyframe.1 - previous_keyframe.1);

    match quat_sampler.interpolation() {
        gltf::animation::Interpolation::Linear => {
            let keyframe_values = get_basic_keyframe_values();
            let previous_keyframe_value = keyframe_values[previous_keyframe.0];
            let next_keyframe_value = keyframe_values[next_keyframe.0];
            previous_keyframe_value.slerp(next_keyframe_value, interpolation_factor)
        }
        gltf::animation::Interpolation::Step => {
            let keyframe_values = get_basic_keyframe_values();
            keyframe_values[previous_keyframe.0]
        }
        gltf::animation::Interpolation::CubicSpline => {
            let keyframe_values = bytemuck::cast_slice::<_, [[f32; 4]; 3]>(&keyframe_values_u8)
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
                .collect::<Vec<_>>();
            let previous_keyframe_value = keyframe_values[previous_keyframe.0];
            let next_keyframe_value = keyframe_values[next_keyframe.0];
            let keyframe_length = next_keyframe.1 - previous_keyframe.1;

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

fn get_keyframe_times(
    sampler: &gltf::animation::Sampler,
    buffers: &[gltf::buffer::Data],
) -> (Vec<f32>, f32) {
    let keyframe_times = {
        let accessor = sampler.input();
        let data_type = accessor.data_type();
        let dimensions = accessor.dimensions();
        if dimensions != gltf::accessor::Dimensions::Scalar {
            // TODO: use error instead of panic
            panic!("Expected scalar data but found: {:?}", dimensions);
        }
        if data_type != gltf::accessor::DataType::F32 {
            // TODO: use error instead of panic
            panic!("Expected f32 data but found: {:?}", data_type);
        }
        let result_u8 = get_buffer_slice_from_accessor(accessor, buffers);
        bytemuck::cast_slice::<_, f32>(&result_u8).to_vec()
    };

    let start_time = keyframe_times[0];
    let keyframe_times: Vec<_> = keyframe_times
        .iter()
        .map(|time| time - start_time)
        .collect();
    let animation_length_seconds = *keyframe_times.last().unwrap();

    (keyframe_times, animation_length_seconds)
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
