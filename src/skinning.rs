use std::collections::{hash_map::Entry, HashMap};

use cgmath::Matrix4;

use super::*;

pub struct AllBoneTransforms {
    pub buffer: Vec<u8>,
    pub animated_bone_transforms: Vec<AllBoneTransformsSlice>,
    pub identity_slice: (usize, usize),
}

pub struct AllBoneTransformsSlice {
    pub binded_mesh_index: usize,
    pub start_index: usize,
    pub end_index: usize,
}

pub fn get_all_bone_data(
    game_scene: &GameScene,
    render_scene: &RenderScene,
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

    for (binded_mesh_indices, model_root_node_index) in game_scene
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(node_index, node)| {
            game_scene
                .get_model_root_if_in_skeleton(node_index)
                .and_then(|model_root_node_index| {
                    node.binded_mesh_indices
                        .as_ref()
                        .map(|binded_mesh_indices| (binded_mesh_indices, model_root_node_index))
                })
        })
    {
        for binded_mesh_index in binded_mesh_indices.iter().copied() {
            let skin_index = game_scene.nodes[model_root_node_index]
                .renderer_skin_index
                .unwrap();
            match skin_index_to_slice_map.entry(skin_index) {
                Entry::Occupied(entry) => {
                    let (start_index, end_index) = *entry.get();
                    animated_bone_transforms.push(AllBoneTransformsSlice {
                        binded_mesh_index,
                        start_index,
                        end_index,
                    });
                }
                Entry::Vacant(entry) => {
                    let bone_transforms: Vec<_> = get_bone_model_space_transforms(
                        game_scene,
                        render_scene,
                        model_root_node_index,
                    )
                    .iter()
                    .copied()
                    .map(GpuMatrix4)
                    .collect();

                    // add padding
                    // TODO: use limit constraints at device creation time to try to lower the min_storage_buffer_offset_alignment number
                    //       cuz smaller buffer = more cache hits
                    let mut padding: Vec<_> = (0..buffer.len()
                        % min_storage_buffer_offset_alignment as usize)
                        .map(|_| 0u8)
                        .collect();

                    let start_index = buffer.len();
                    let end_index = start_index + bone_transforms.len() * matrix_size_bytes;

                    buffer.append(&mut bytemuck::cast_slice(&bone_transforms).to_vec());
                    buffer.append(&mut padding);
                    animated_bone_transforms.push(AllBoneTransformsSlice {
                        binded_mesh_index,
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

pub fn get_bone_model_space_transforms(
    game_scene: &GameScene,
    render_scene: &RenderScene,
    model_root_node_index: usize,
) -> Vec<Matrix4<f32>> {
    let model_root_node = &game_scene.nodes[model_root_node_index];
    let skin = &render_scene.skins[model_root_node.renderer_skin_index.unwrap()];
    let skeleton_parent_index_map: HashMap<usize, usize> = skin
        .bone_node_indices
        .iter()
        .filter_map(|bone_node_index| {
            game_scene
                .parent_index_map
                .get(bone_node_index)
                .map(|parent_index| (*bone_node_index, *parent_index))
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
                    acc * game_scene.nodes[*node_index].transform
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
