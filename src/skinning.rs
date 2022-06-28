use std::collections::HashMap;

use cgmath::Matrix4;

use super::*;

pub fn get_bone_model_space_transforms(
    scene: &Scene,
    model_root_node_index: usize,
) -> Vec<Matrix4<f32>> {
    let model_root_node = &scene.nodes[model_root_node_index];
    let skin = &scene.skins[model_root_node.skin_index.unwrap()];
    let skeleton_parent_index_map: HashMap<usize, usize> = skin
        .bone_node_indices
        .iter()
        .map(|bone_node_index| {
            (
                *bone_node_index,
                *scene.parent_index_map.get(bone_node_index).unwrap(),
            )
        })
        .collect();
    // goes from world space into the model's space
    let world_space_to_model_space = model_root_node
        .transform
        .matrix()
        .inverse_transform()
        .unwrap();
    skin.bone_node_indices
        .iter()
        .enumerate()
        .map(|(bone_index, bone_node_index)| {
            // goes from the bone's space into world space given parent hierarchy
            let node_ancestry_list =
                get_node_ancestry_list(*bone_node_index, &skeleton_parent_index_map);
            let bone_space_to_world_space = node_ancestry_list
                .iter()
                .rev()
                .fold(crate::transform::Transform::new(), |acc, node_index| {
                    acc * scene.nodes[*node_index].transform
                });
            // goes from the model's space into the bone's space
            let model_space_to_bone_space = skin.bone_inverse_bind_matrices[bone_index];
            // see https://www.khronos.org/files/gltf20-reference-guide.pdf
            world_space_to_model_space
                * bone_space_to_world_space.matrix()
                * model_space_to_bone_space
        })
        .collect()
}
