use std::collections::{hash_map::Entry, HashMap};

use cgmath::Matrix4;

use super::*;

pub struct AllBoneTransforms {
    pub buffer: Vec<u8>,
    pub animated_bone_transforms: Vec<AllBoneTransformsSlice>,
    pub identity_slice: (usize, usize),
}

pub struct AllBoneTransformsSlice {
    pub binded_pbr_mesh_index: usize,
    pub start_index: usize,
    pub end_index: usize,
}

pub fn get_all_bone_data(
    game_scene: &GameScene,
    min_storage_buffer_offset_alignment: u32,
) -> AllBoneTransforms {
    let matrix_size_bytes = std::mem::size_of::<GpuMatrix4>();
    let identity_bone_count = 4;
    let identity_slice = (0, identity_bone_count * matrix_size_bytes);

    let mut buffer: Vec<u8> = bytemuck::cast_slice(
        &((0..identity_bone_count)
            .map(|_| GpuMatrix4(Matrix4::one()))
            .collect::<Vec<_>>()),
    )
    .to_vec();

    let mut animated_bone_transforms: Vec<AllBoneTransformsSlice> = Vec::new();
    let mut skin_index_to_slice_map: HashMap<usize, (usize, usize)> = HashMap::new();

    for (binded_pbr_mesh_indices, skeleton_root_node_id) in game_scene.nodes().filter_map(|node| {
        game_scene
            .get_skeleton_root(node.id())
            .and_then(|skeleton_root_node_id| match &node.mesh {
                Some(GameNodeMesh::Pbr { mesh_indices, .. }) => {
                    Some((mesh_indices, skeleton_root_node_id))
                }
                _ => None,
            })
    }) {
        for binded_pbr_mesh_index in binded_pbr_mesh_indices.iter().copied() {
            let skin_index = game_scene
                .get_node(skeleton_root_node_id)
                .unwrap()
                .skin_index
                .unwrap();
            match skin_index_to_slice_map.entry(skin_index) {
                Entry::Occupied(entry) => {
                    let (start_index, end_index) = *entry.get();
                    animated_bone_transforms.push(AllBoneTransformsSlice {
                        binded_pbr_mesh_index,
                        start_index,
                        end_index,
                    });
                }
                Entry::Vacant(entry) => {
                    let bone_transforms: Vec<_> =
                        get_bone_model_space_transforms(game_scene, skeleton_root_node_id)
                            .iter()
                            .copied()
                            .map(GpuMatrix4)
                            .collect();

                    // add padding
                    let mut padding: Vec<_> = (0..buffer.len()
                        % min_storage_buffer_offset_alignment as usize)
                        .map(|_| 0u8)
                        .collect();

                    let start_index = buffer.len();
                    let end_index = start_index + bone_transforms.len() * matrix_size_bytes;

                    buffer.append(&mut bytemuck::cast_slice(&bone_transforms).to_vec());
                    buffer.append(&mut padding);
                    animated_bone_transforms.push(AllBoneTransformsSlice {
                        binded_pbr_mesh_index,
                        start_index,
                        end_index,
                    });
                    entry.insert((start_index, end_index));
                }
            }
        }
    }

    AllBoneTransforms {
        buffer,
        animated_bone_transforms,
        identity_slice,
    }
}

fn get_bone_model_space_transforms(
    game_scene: &GameScene,
    skeleton_root_node_id: GameNodeId, // node must exist!
) -> Vec<Matrix4<f32>> {
    let skeleton_root_node = game_scene.get_node(skeleton_root_node_id).unwrap();
    let skin = &game_scene.skins[skeleton_root_node.skin_index.unwrap()];
    // goes from world space into the skeleton's space
    let world_space_to_model_space = skeleton_root_node
        .transform
        .matrix()
        .inverse_transform()
        .unwrap();
    skin.bone_node_ids
        .iter()
        .enumerate()
        .map(|(bone_index, bone_node_id)| {
            // goes from the bone's space into world space given parent hierarchy
            let node_ancestry_list =
                game_scene.get_skeleton_node_ancestry_list(*bone_node_id, skeleton_root_node_id);
            let bone_space_to_world_space = node_ancestry_list
                .iter()
                .rev()
                .fold(crate::transform::Transform::new(), |acc, node_id| {
                    acc * game_scene.get_node(*node_id).unwrap().transform
                });
            // goes from the skeletons's space into the bone's space
            let skeleton_space_to_bone_space = skin.bone_inverse_bind_matrices[bone_index];
            // see https://www.khronos.org/files/gltf20-reference-guide.pdf
            world_space_to_model_space
                * bone_space_to_world_space.matrix()
                * skeleton_space_to_bone_space
        })
        .collect()
}
