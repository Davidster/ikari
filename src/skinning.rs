use std::collections::{hash_map::Entry, HashMap};

use cgmath::Matrix4;

use super::*;

pub struct AllBoneTransforms {
    pub buffer: Vec<u8>,
    pub animated_bone_transforms: Vec<AllBoneTransformsSlice>,
    pub identity_slice: (usize, usize),
}

pub struct AllBoneTransformsSlice {
    pub skin_index: usize,
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

    let skinned_nodes: Vec<_> = game_scene
        .nodes()
        .filter_map(|node| {
            game_scene
                .get_skeleton_skin_node_id(node.id())
                .and_then(|skeleton_skin_node_id| {
                    // dbg!(skeleton_skin_node_id, node.id(), &node.mesh);
                    match &game_scene.get_node(skeleton_skin_node_id).unwrap().mesh {
                        Some(GameNodeMesh::Pbr { mesh_indices, .. }) => {
                            Some((mesh_indices, skeleton_skin_node_id))
                        }
                        _ => None,
                    }
                })
        })
        .collect();

    for (binded_pbr_mesh_indices, skeleton_skin_node_id) in skinned_nodes {
        for binded_pbr_mesh_index in binded_pbr_mesh_indices.iter().copied() {
            let skin_index = game_scene
                .get_node(skeleton_skin_node_id)
                .unwrap()
                .skin_index
                .unwrap();
            match skin_index_to_slice_map.entry(skin_index) {
                Entry::Occupied(entry) => {
                    let (start_index, end_index) = *entry.get();
                    animated_bone_transforms.push(AllBoneTransformsSlice {
                        skin_index,
                        binded_pbr_mesh_index,
                        start_index,
                        end_index,
                    });
                }
                Entry::Vacant(entry) => {
                    let bone_transforms: Vec<_> =
                        get_bone_skeleton_space_transforms(game_scene, skeleton_skin_node_id)
                            .iter()
                            .copied()
                            .map(GpuMatrix4)
                            .collect();
                    // let skeleton_skin_node = game_scene.get_node(skeleton_skin_node_id).unwrap();
                    // let skin = &game_scene.skins[skeleton_skin_node.skin_index.unwrap()];
                    // let bone_transforms: Vec<_> = (0..skin.bone_node_ids.len())
                    //     .map(|_| GpuMatrix4(Matrix4::one()))
                    //     .collect::<Vec<_>>();

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
                        skin_index,
                        binded_pbr_mesh_index,
                        start_index,
                        end_index,
                    });
                    entry.insert((start_index, end_index));
                }
            }
        }
    }

    // println!(
    //     "animated_bone_transforms.len(): {:?}",
    //     animated_bone_transforms.len()
    // );
    // println!(
    //     "animated_bone_transforms sizes: {:?}",
    //     animated_bone_transforms
    //         .iter()
    //         .map(|trans| (trans.start_index, trans.end_index))
    //         .collect::<Vec<_>>()
    // );

    AllBoneTransforms {
        buffer,
        animated_bone_transforms,
        identity_slice,
    }
}

fn get_bone_skeleton_space_transforms(
    game_scene: &GameScene,
    skeleton_skin_node_id: GameNodeId, // node must exist!
) -> Vec<Matrix4<f32>> {
    let skeleton_skin_node = game_scene.get_node(skeleton_skin_node_id).unwrap();
    let skin = &game_scene.skins[skeleton_skin_node.skin_index.unwrap()];
    skin.bone_node_ids
        .iter()
        .enumerate()
        .map(|(bone_index, bone_node_id)| {
            let (bone_skeleton_space_transform, node_ancestry_list) =
                get_bone_skeleton_space_transform(
                    game_scene,
                    skin,
                    *bone_node_id,
                    skeleton_skin_node_id,
                    bone_index,
                );

            // add any parent joint transforms if the bone is in a recursive skeleton chain
            let mut bone_transform_chain: Vec<Matrix4<f32>> = vec![bone_skeleton_space_transform];
            let mut ancestry_list_root = *node_ancestry_list.last().unwrap();
            loop {
                let parent_bone_info = game_scene
                    .get_node_parent(ancestry_list_root)
                    .and_then(|parent_node_id| {
                        game_scene
                            .get_skeleton_skin_node_id(parent_node_id)
                            .map(|parent_skin_node_id| (parent_node_id, parent_skin_node_id))
                    })
                    .map(|(parent_node_id, parent_skin_node_id)| {
                        (
                            parent_node_id,
                            parent_skin_node_id,
                            &game_scene.skins[game_scene
                                .get_node(parent_skin_node_id)
                                .unwrap()
                                .skin_index
                                .unwrap()],
                        )
                    }); // maybe do a map here to get the skin
                if let Some((parent_node_id, parent_skin_node_id, parent_skin)) = parent_bone_info {
                    let (bone_skeleton_space_transform, node_ancestry_list) =
                        get_bone_skeleton_space_transform(
                            game_scene,
                            parent_skin,
                            parent_node_id,
                            parent_skin_node_id,
                            bone_index,
                        );
                    ancestry_list_root = *node_ancestry_list.last().unwrap();
                    bone_transform_chain.push(bone_skeleton_space_transform);
                } else {
                    break;
                }
            }
            // dbg!(bone_transform_chain.len());
            // let furthest_ancestor_node = node_ancestry_list[last or first idk];
            // let furthest_ancestor_parent = furthest_ancestor_node.parent();
            //     let parent_joint_correction_transform = if furthest_ancestor_parent.is_joint() {
            //         // let parent_joint_transform =
            //     } else {
            //         Matrix4::one()
            //     };
            // }

            // dbg!(
            //     &bone_transform_chain[0],
            //     bone_transform_chain
            //         .iter()
            //         .rev()
            //         .fold(Matrix4::one(), |acc, transform| acc * transform),
            // );

            bone_transform_chain
                .iter()
                .rev()
                .fold(Matrix4::one(), |acc, transform| acc * transform)
        })
        .collect()
}

fn get_bone_skeleton_space_transform(
    game_scene: &GameScene,
    skin: &Skin,
    bone_node_id: GameNodeId,
    skeleton_skin_node_id: GameNodeId,
    bone_index: usize,
) -> (Matrix4<f32>, Vec<GameNodeId>) {
    let node_ancestry_list =
        game_scene.get_skeleton_node_ancestry_list(bone_node_id, skeleton_skin_node_id);

    // goes from the bone's space into skeleton space given parent hierarchy
    let bone_space_to_skeleton_space = node_ancestry_list
        .iter()
        .rev()
        .fold(crate::transform::Transform::new(), |acc, node_id| {
            acc * game_scene.get_node(*node_id).unwrap().transform
        });

    // TODO: uncomment?
    // let bone_space_to_skeleton_space =
    //     game_scene.get_skeleton_transform_for_node(*bone_node_id, skeleton_skin_node_id);

    // goes from the skeletons's space into the bone's space
    let skeleton_space_to_bone_space = skin.bone_inverse_bind_matrices[bone_index];
    // see https://www.khronos.org/files/gltf20-reference-guide.pdf
    let skeleton_space_transform =
        bone_space_to_skeleton_space.matrix() * skeleton_space_to_bone_space;
    (skeleton_space_transform, node_ancestry_list)
}
