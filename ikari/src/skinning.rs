use crate::scene::{GameNodeId, Scene, Skin};

use std::collections::{hash_map::Entry, HashMap};

use glam::f32::Mat4;

pub struct AllBoneTransforms {
    pub buffer: Vec<u8>,
    pub animated_bone_transforms: Vec<AllBoneTransformsSlice>,
    pub identity_slice: (usize, usize),
    // pub biggest_chunk_size: usize,
}

#[derive(Debug)]
pub struct AllBoneTransformsSlice {
    pub mesh_index: usize,
    pub start_index: usize,
    pub end_index: usize,
}

pub fn get_all_bone_data(
    scene: &Scene,
    min_storage_buffer_offset_alignment: u32,
) -> AllBoneTransforms {
    let matrix_size_bytes = std::mem::size_of::<Mat4>();
    let identity_bone_count = 4;
    let identity_slice = (0, identity_bone_count * matrix_size_bytes);

    let mut buffer: Vec<u8> = bytemuck::cast_slice(
        &((0..identity_bone_count)
            .map(|_| Mat4::IDENTITY)
            .collect::<Vec<_>>()),
    )
    .to_vec();

    let mut animated_bone_transforms: Vec<AllBoneTransformsSlice> = Vec::new();
    let mut skin_index_to_slice_map: HashMap<usize, (usize, usize)> = HashMap::new();

    let mut biggest_chunk_size = 0;

    for skin in &scene.skins {
        let Some(skin_node) = scene.get_node(skin.node_id) else {
            continue;
        };
        let Some(visual) = skin_node.visual.as_ref() else {
            continue;
        };
        let Some(skin_index) = skin_node.skin_index else {
            log::error!("Skin pointed to a node with no skin index");
            continue;
        };

        match skin_index_to_slice_map.entry(skin_index) {
            Entry::Occupied(entry) => {
                let (start_index, end_index) = *entry.get();
                animated_bone_transforms.push(AllBoneTransformsSlice {
                    mesh_index: visual.mesh_index,
                    start_index,
                    end_index,
                });
            }
            Entry::Vacant(entry) => {
                let skin = &scene.skins[skin_index];
                let bone_transforms: Vec<_> = skin
                    .bone_node_ids
                    .iter()
                    .enumerate()
                    .map(|(bone_index, bone_node_id)| {
                        get_bone_skeleton_space_transform(
                            scene,
                            skin,
                            skin.node_id,
                            bone_index,
                            *bone_node_id,
                        )
                    })
                    .collect();

                let start_index = buffer.len();
                let end_index = start_index + bone_transforms.len() * matrix_size_bytes;
                buffer.append(&mut bytemuck::cast_slice(&bone_transforms).to_vec());

                // add padding
                let needed_padding = min_storage_buffer_offset_alignment as usize
                    - (buffer.len() % min_storage_buffer_offset_alignment as usize);
                let mut padding: Vec<_> = (0..needed_padding).map(|_| 0u8).collect();
                buffer.append(&mut padding);

                let chunk_size = needed_padding + end_index - start_index;
                if chunk_size > biggest_chunk_size {
                    biggest_chunk_size = chunk_size;
                }

                animated_bone_transforms.push(AllBoneTransformsSlice {
                    mesh_index: visual.mesh_index,
                    start_index,
                    end_index,
                });
                entry.insert((start_index, end_index));
            }
        }
    }

    buffer.resize(buffer.len() + biggest_chunk_size, 0);

    AllBoneTransforms {
        buffer,
        animated_bone_transforms,
        identity_slice,
        // biggest_chunk_size,
    }
}

pub fn get_bone_skeleton_space_transform(
    scene: &Scene,
    skin: &Skin,
    skeleton_skin_node_id: GameNodeId,
    bone_index: usize,
    bone_node_id: GameNodeId,
) -> Mat4 {
    // Mat4::IDENTITY
    let node_ancestry_list =
        scene.get_skeleton_node_ancestry_list(bone_node_id, skeleton_skin_node_id);

    // goes from the bone's space into skeleton space given parent hierarchy
    let bone_space_to_skeleton_space = node_ancestry_list
        .iter()
        .rev()
        .fold(crate::transform::Transform::IDENTITY, |acc, node_id| {
            acc * scene.get_node(*node_id).unwrap().transform
        });

    // goes from the skeletons's space into the bone's space
    let skeleton_space_to_bone_space = skin.bone_inverse_bind_matrices[bone_index];
    // see https://www.khronos.org/files/gltf20-reference-guide.pdf
    Mat4::from(bone_space_to_skeleton_space) * skeleton_space_to_bone_space
}
