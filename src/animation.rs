use cgmath::{Quaternion, Vector3};

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

            // TODO: transform_builder probably shouldn't be an option
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
                        if let Some(position) =
                            get_vec3_at_moment(&channel.sampler(), buffers, moment_seconds)
                        {
                            new_transform.set_position(position);
                        }
                    }
                    gltf::animation::Property::Scale => {
                        if let Some(scale) =
                            get_vec3_at_moment(&channel.sampler(), buffers, moment_seconds)
                        {
                            new_transform.set_scale(scale)
                        }
                    }
                    gltf::animation::Property::Rotation => {
                        if let Some(rotation) =
                            get_quat_at_moment(&channel.sampler(), buffers, moment_seconds)
                        {
                            new_transform.set_rotation(rotation)
                        }
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
) -> Option<Vector3<f32>> {
    let keyframe_values = {
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
        let result_u8 = get_buffer_slice_from_accessor(accessor, buffers);
        bytemuck::cast_slice::<_, [f32; 3]>(&result_u8)
            .to_vec()
            .iter()
            .copied()
            .map(Vector3::from)
            .collect::<Vec<_>>()
    };

    let (keyframe_times, animation_length_seconds) = get_keyframe_times(vec3_sampler, buffers);
    let animation_time_seconds = global_time_seconds % animation_length_seconds;

    let previous_keyframe = keyframe_times
        .iter()
        .enumerate()
        .filter(|(_, t_val)| **t_val <= animation_time_seconds)
        .last()
        .unwrap();

    match vec3_sampler.interpolation() {
        gltf::animation::Interpolation::Linear => {
            let next_keyframe = keyframe_times
                .iter()
                .enumerate()
                .find(|(_, t_val)| **t_val > animation_time_seconds)
                .unwrap();
            let interpolation_factor = (animation_time_seconds - previous_keyframe.1)
                / (next_keyframe.1 - previous_keyframe.1);
            Some(lerp_vec(
                keyframe_values[previous_keyframe.0],
                keyframe_values[next_keyframe.0],
                interpolation_factor,
            ))
        }
        gltf::animation::Interpolation::Step => Some(keyframe_values[previous_keyframe.0]),
        gltf::animation::Interpolation::CubicSpline => None,
    }
}

fn get_quat_at_moment(
    quat_sampler: &gltf::animation::Sampler,
    buffers: &[gltf::buffer::Data],
    global_time_seconds: f32,
) -> Option<Quaternion<f32>> {
    let keyframe_values = {
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
        let result_u8 = get_buffer_slice_from_accessor(accessor, buffers);
        bytemuck::cast_slice::<_, [f32; 4]>(&result_u8)
            .to_vec()
            .iter()
            .copied()
            .map(Quaternion::from)
            .collect::<Vec<_>>()
    };

    let (keyframe_times, animation_length_seconds) = get_keyframe_times(quat_sampler, buffers);
    let animation_time_seconds = global_time_seconds % animation_length_seconds;

    let previous_keyframe = keyframe_times
        .iter()
        .enumerate()
        .filter(|(_, t_val)| **t_val <= animation_time_seconds)
        .last()
        .unwrap();

    match quat_sampler.interpolation() {
        gltf::animation::Interpolation::Linear => {
            let next_keyframe = keyframe_times
                .iter()
                .enumerate()
                .find(|(_, t_val)| **t_val > animation_time_seconds)
                .unwrap();
            let interpolation_factor = (animation_time_seconds - previous_keyframe.1)
                / (next_keyframe.1 - previous_keyframe.1);
            Some(
                keyframe_values[previous_keyframe.0]
                    .slerp(keyframe_values[next_keyframe.0], interpolation_factor),
            )
        }
        gltf::animation::Interpolation::Step => Some(keyframe_values[previous_keyframe.0]),
        gltf::animation::Interpolation::CubicSpline => None,
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
