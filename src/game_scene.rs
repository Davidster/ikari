use std::collections::HashMap;

use cgmath::{Matrix4, Vector3};

use super::*;

#[derive(Debug)]
pub struct GameScene {
    nodes: Vec<GameNode>,
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
    // node index -> parent node index
    parent_index_map: HashMap<usize, usize>,
    // skeleton root node index -> parent_index_map
    skeleton_parent_index_maps: HashMap<usize, HashMap<usize, usize>>,
}

#[derive(Debug, Clone)]
pub struct GameNodeDesc {
    pub transform: crate::transform::Transform,
    pub skin_index: Option<usize>,
    pub mesh: Option<GameNodeMesh>,
}

#[derive(Debug, Clone)]
pub struct GameNode {
    pub transform: crate::transform::Transform,
    pub skin_index: Option<usize>,
    pub mesh: Option<GameNodeMesh>,
    id: GameNodeId,
}

#[derive(Debug, Copy, Clone)]
pub struct GameNodeId(usize); // index into GameScene::nodes array

#[derive(Debug, Clone)]
pub enum GameNodeMesh {
    Pbr {
        mesh_indices: Vec<usize>,
        material_override: Option<DynamicPbrParams>,
    },
    Unlit {
        mesh_indices: Vec<usize>,
        color: Vector3<f32>,
    },
}

#[derive(Debug, Clone)]
pub struct Skin {
    pub bone_node_ids: Vec<GameNodeId>,
    pub bone_inverse_bind_matrices: Vec<Matrix4<f32>>,
}

#[derive(Debug, Clone)]
pub struct IndexedSkin {
    pub bone_node_indices: Vec<usize>,
    pub bone_inverse_bind_matrices: Vec<Matrix4<f32>>,
}

#[derive(Debug)]
pub struct Animation {
    pub length_seconds: f32,
    pub channels: Vec<Channel>,
}

#[derive(Debug)]
pub struct Channel {
    pub node_id: GameNodeId,
    pub property: gltf::animation::Property,
    pub interpolation_type: gltf::animation::Interpolation,
    pub keyframe_timings: Vec<f32>,
    pub keyframe_values_u8: Vec<u8>,
}

#[derive(Debug)]
pub struct IndexedAnimation {
    pub length_seconds: f32,
    pub channels: Vec<IndexedChannel>,
}

#[derive(Debug)]
pub struct IndexedChannel {
    pub node_index: usize,
    pub property: gltf::animation::Property,
    pub interpolation_type: gltf::animation::Interpolation,
    pub keyframe_timings: Vec<f32>,
    pub keyframe_values_u8: Vec<u8>,
}

impl GameScene {
    pub fn new(
        nodes_desc: Vec<GameNodeDesc>,
        skins: Vec<IndexedSkin>,
        animations: Vec<IndexedAnimation>,
        parent_index_map: HashMap<usize, usize>,
    ) -> Self {
        let skins: Vec<_> = skins
            .iter()
            .map(|indexed_skin| Skin {
                bone_node_ids: indexed_skin
                    .bone_node_indices
                    .iter()
                    .map(|node_index| GameNodeId(*node_index))
                    .collect(),
                bone_inverse_bind_matrices: indexed_skin.bone_inverse_bind_matrices.clone(),
            })
            .collect();
        let animations: Vec<_> = animations
            .iter()
            .map(|indexed_animation| Animation {
                length_seconds: indexed_animation.length_seconds,
                channels: indexed_animation
                    .channels
                    .iter()
                    .map(|indexed_channel| Channel {
                        node_id: GameNodeId(indexed_channel.node_index),
                        property: indexed_channel.property,
                        interpolation_type: indexed_channel.interpolation_type,
                        keyframe_timings: indexed_channel.keyframe_timings.clone(),
                        keyframe_values_u8: indexed_channel.keyframe_values_u8.clone(),
                    })
                    .collect(),
            })
            .collect();
        let mut scene = GameScene {
            nodes: Vec::new(),
            skins,
            animations,
            parent_index_map,
            skeleton_parent_index_maps: HashMap::new(),
        };

        nodes_desc.iter().for_each(|node_desc| {
            scene.add_node(node_desc.clone());
        });

        scene.rebuild_skeleton_parent_index_maps();

        scene
    }

    fn rebuild_skeleton_parent_index_maps(&mut self) {
        self.skeleton_parent_index_maps = HashMap::new();
        let skinned_nodes = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(node_index, node)| {
                node.skin_index
                    .map(|skin_index| (node_index, &self.skins[skin_index]))
            });
        for (node_index, skin) in skinned_nodes {
            let skeleton_parent_index_map: HashMap<usize, usize> = skin
                .bone_node_ids
                .iter()
                .filter_map(|GameNodeId(bone_node_index)| {
                    self.parent_index_map
                        .get(bone_node_index)
                        .map(|parent_index| (*bone_node_index, *parent_index))
                })
                .collect();
            self.skeleton_parent_index_maps
                .insert(node_index, skeleton_parent_index_map);
        }
    }

    pub fn get_skeleton_root(&self, node_id: GameNodeId) -> Option<GameNodeId> {
        self.get_node_ancestry_list(node_id)
            .iter()
            .find_map(|GameNodeId(node_index)| {
                self.nodes[*node_index]
                    .skin_index
                    .map(|_| GameNodeId(*node_index))
            })
    }

    pub fn get_node_ancestry_list(&self, GameNodeId(node_index): GameNodeId) -> Vec<GameNodeId> {
        get_node_ancestry_list(node_index, &self.parent_index_map)
            .iter()
            .map(|node_index| GameNodeId(*node_index))
            .collect()
    }

    pub fn get_skeleton_node_ancestry_list(
        &self,
        GameNodeId(node_index): GameNodeId,
        GameNodeId(skeleton_root_node_index): GameNodeId,
    ) -> Vec<GameNodeId> {
        match self
            .skeleton_parent_index_maps
            .get(&skeleton_root_node_index)
        {
            Some(skeleton_parent_index_map) => {
                get_node_ancestry_list(node_index, skeleton_parent_index_map)
                    .iter()
                    .map(|node_index| GameNodeId(*node_index))
                    .collect()
            }
            None => Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: GameNodeDesc) -> GameNodeId {
        let GameNodeDesc {
            transform,
            skin_index,
            mesh,
        } = node;
        let new_node_id = GameNodeId(self.nodes.len());
        self.nodes.push(GameNode {
            transform,
            skin_index,
            mesh,
            id: new_node_id,
        });
        new_node_id
    }

    pub fn get_node(&self, GameNodeId(node_index): GameNodeId) -> &GameNode {
        &self.nodes[node_index]
    }

    pub fn get_node_mut(&mut self, GameNodeId(node_index): GameNodeId) -> &mut GameNode {
        &mut self.nodes[node_index]
    }

    pub fn get_node_mut_by_index(&mut self, index: usize) -> Option<&mut GameNode> {
        self.nodes.get_mut(index)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &GameNode> {
        self.nodes.iter()
    }
}

fn get_node_ancestry_list(
    node_index: usize,
    parent_index_map: &HashMap<usize, usize>,
) -> Vec<usize> {
    get_node_ancestry_list_impl(node_index, parent_index_map, Vec::new())
}

fn get_node_ancestry_list_impl(
    node_index: usize,
    parent_index_map: &HashMap<usize, usize>,
    acc: Vec<usize>,
) -> Vec<usize> {
    let with_self: Vec<_> = acc.iter().chain(vec![node_index].iter()).copied().collect();
    match parent_index_map.get(&node_index) {
        Some(parent_index) => {
            get_node_ancestry_list_impl(*parent_index, parent_index_map, with_self).to_vec()
        }
        None => with_self,
    }
}

impl GameNode {
    pub fn id(&self) -> GameNodeId {
        self.id
    }
}

impl Default for GameNodeDesc {
    fn default() -> Self {
        Self {
            transform: crate::transform::Transform::new(),
            skin_index: None,
            mesh: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameNodeDescBuilder {
    transform: crate::transform::Transform,
    skin_index: Option<usize>,
    mesh: Option<GameNodeMesh>,
}

impl GameNodeDescBuilder {
    pub fn new() -> Self {
        let GameNodeDesc {
            transform,
            skin_index,
            mesh,
        } = GameNodeDesc::default();
        Self {
            transform,
            skin_index,
            mesh,
        }
    }

    pub fn transform(mut self, transform: crate::transform::Transform) -> Self {
        self.transform = transform;
        self
    }

    #[allow(dead_code)]
    pub fn skin_index(mut self, skin_index: Option<usize>) -> Self {
        self.skin_index = skin_index;
        self
    }

    pub fn mesh(mut self, mesh: Option<GameNodeMesh>) -> Self {
        self.mesh = mesh;
        self
    }

    pub fn build(self) -> GameNodeDesc {
        GameNodeDesc {
            transform: self.transform,
            skin_index: self.skin_index,
            mesh: self.mesh,
        }
    }
}
