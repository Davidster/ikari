use std::collections::HashMap;

use cgmath::{Matrix4, Vector3};

use super::*;

#[derive(Debug)]
pub struct Scene {
    nodes: Vec<(Option<GameNode>, usize)>, // (node, generation number). None means the node was removed from the scene
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
    // node index -> parent node index
    parent_index_map: HashMap<usize, usize>,
    // skeleton skin node index -> parent_index_map
    skeleton_parent_index_maps: HashMap<usize, HashMap<usize, usize>>,
}

#[derive(Debug, Clone)]
pub struct GameNodeDesc {
    pub transform: crate::transform::Transform,
    pub skin_index: Option<usize>,
    pub mesh: Option<GameNodeMesh>,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GameNode {
    pub transform: crate::transform::Transform,
    pub skin_index: Option<usize>,
    pub mesh: Option<GameNodeMesh>,
    pub name: Option<String>,
    id: GameNodeId,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct GameNodeId(usize, usize); // (index into GameScene::nodes array, generation num)

#[derive(Debug, Clone)]
pub struct GameNodeMesh {
    pub mesh_type: GameNodeMeshType,
    pub mesh_indices: Vec<usize>,
    pub wireframe: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum GameNodeMeshType {
    Pbr {
        material_override: Option<DynamicPbrParams>,
    },
    Unlit {
        color: Vector3<f32>,
    },
}

#[derive(Debug, Clone)]
pub struct Skin {
    pub node_id: GameNodeId,
    pub bone_node_ids: Vec<GameNodeId>,
    pub bone_inverse_bind_matrices: Vec<Matrix4<f32>>,
    // each transform moves a 2x2x2 box centered at the origin
    // such that it surrounds the bone's vertices in bone space
    pub bone_bounding_box_transforms: Vec<crate::transform::Transform>,
}

#[derive(Debug, Clone)]
pub struct IndexedSkin {
    pub bone_node_indices: Vec<usize>,
    pub bone_inverse_bind_matrices: Vec<Matrix4<f32>>,
    pub bone_bounding_box_transforms: Vec<crate::transform::Transform>,
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

impl Scene {
    pub fn new(
        nodes_desc: Vec<GameNodeDesc>,
        indexed_skins: Vec<IndexedSkin>,
        animations: Vec<IndexedAnimation>,
        parent_index_map: HashMap<usize, usize>,
    ) -> Self {
        let animations: Vec<_> = animations
            .iter()
            .map(|indexed_animation| Animation {
                length_seconds: indexed_animation.length_seconds,
                speed: 1.0,
                channels: indexed_animation
                    .channels
                    .iter()
                    .map(|indexed_channel| Channel {
                        node_id: GameNodeId(indexed_channel.node_index, 0),
                        property: indexed_channel.property,
                        interpolation_type: indexed_channel.interpolation_type,
                        keyframe_timings: indexed_channel.keyframe_timings.clone(),
                        keyframe_values_u8: indexed_channel.keyframe_values_u8.clone(),
                    })
                    .collect(),
                state: AnimationState::default(),
            })
            .collect();
        let mut scene = Scene {
            nodes: Vec::new(),
            skins: Vec::new(),
            animations,
            parent_index_map,
            skeleton_parent_index_maps: HashMap::new(),
        };

        nodes_desc.iter().for_each(|node_desc| {
            scene.add_node(node_desc.clone());
        });

        scene.skins = (0..indexed_skins.len())
            .map(|skin_index| {
                let indexed_skin = &indexed_skins[skin_index];
                let node_id = scene
                    .nodes()
                    .find(|node| node.skin_index == Some(skin_index))
                    .unwrap()
                    .id;
                Skin {
                    node_id,
                    bone_node_ids: indexed_skin
                        .bone_node_indices
                        .iter()
                        .map(|node_index| GameNodeId(*node_index, 0))
                        .collect(),
                    bone_inverse_bind_matrices: indexed_skin.bone_inverse_bind_matrices.clone(),
                    bone_bounding_box_transforms: indexed_skin.bone_bounding_box_transforms.clone(),
                }
            })
            .collect();

        scene.rebuild_skeleton_parent_index_maps();

        scene
    }

    fn rebuild_skeleton_parent_index_maps(&mut self) {
        self.skeleton_parent_index_maps = HashMap::new();
        for skin in &self.skins {
            let skeleton_parent_index_map: HashMap<usize, usize> = skin
                .bone_node_ids
                .iter()
                .filter_map(|GameNodeId(bone_node_index, _)| {
                    self.parent_index_map
                        .get(bone_node_index)
                        .map(|parent_index| (*bone_node_index, *parent_index))
                })
                .collect();
            self.skeleton_parent_index_maps
                .insert(skin.node_id.0, skeleton_parent_index_map);
        }
    }

    pub fn merge_scene(
        &mut self,
        renderer_state: &mut RendererState,
        mut other_scene: Scene,
        mut other_render_buffers: RenderBuffers,
    ) {
        let pbr_mesh_index_offset = renderer_state.buffers.binded_pbr_meshes.len();
        let unlit_mesh_index_offset = renderer_state.buffers.binded_unlit_meshes.len();

        for binded_wireframe_mesh in &mut other_render_buffers.binded_wireframe_meshes {
            match binded_wireframe_mesh.source_mesh_type {
                MeshType::Pbr => {
                    binded_wireframe_mesh.source_mesh_index += pbr_mesh_index_offset;
                }
                MeshType::Unlit => {
                    binded_wireframe_mesh.source_mesh_index += unlit_mesh_index_offset;
                }
            }
        }

        renderer_state
            .buffers
            .binded_pbr_meshes
            .append(&mut other_render_buffers.binded_pbr_meshes);
        renderer_state
            .buffers
            .binded_unlit_meshes
            .append(&mut other_render_buffers.binded_unlit_meshes);
        renderer_state
            .buffers
            .binded_wireframe_meshes
            .append(&mut other_render_buffers.binded_wireframe_meshes);
        renderer_state
            .buffers
            .textures
            .append(&mut other_render_buffers.textures);
        let skin_index_offset = self.skins.len();
        let node_index_offset = self.nodes.len();
        for (node, _) in &mut other_scene.nodes {
            if let Some(ref mut node) = node {
                match node.mesh {
                    Some(GameNodeMesh {
                        ref mut mesh_indices,
                        mesh_type: GameNodeMeshType::Pbr { .. },
                        ..
                    }) => {
                        *mesh_indices = mesh_indices
                            .iter()
                            .map(|mesh_index| mesh_index + pbr_mesh_index_offset)
                            .collect();
                    }
                    Some(GameNodeMesh {
                        ref mut mesh_indices,
                        mesh_type: GameNodeMeshType::Unlit { .. },
                        ..
                    }) => {
                        *mesh_indices = mesh_indices
                            .iter()
                            .map(|mesh_index| mesh_index + unlit_mesh_index_offset)
                            .collect();
                    }
                    _ => {}
                }
                if let Some(ref mut skin_index) = node.skin_index {
                    *skin_index += skin_index_offset;
                }
                let GameNodeId(old_index, _) = node.id;
                let new_index = old_index + node_index_offset;
                if let Some(parent_index) = other_scene.parent_index_map.get(&old_index) {
                    self.parent_index_map
                        .insert(new_index, parent_index + node_index_offset);
                }
                node.id.0 = new_index;
            }
        }
        for mut skin in &mut other_scene.skins {
            skin.bone_node_ids = skin
                .bone_node_ids
                .iter()
                .map(|GameNodeId(node_index, _)| GameNodeId(node_index + node_index_offset, 0))
                .collect();
        }
        for animation in &mut other_scene.animations {
            for channel in &mut animation.channels {
                let GameNodeId(node_index, _) = channel.node_id;
                channel.node_id = GameNodeId(node_index + node_index_offset, 0);
            }
        }

        self.nodes.append(&mut other_scene.nodes);
        self.skins.append(&mut other_scene.skins);
        self.animations.append(&mut other_scene.animations);
        self.rebuild_skeleton_parent_index_maps();
    }

    pub fn _get_skeleton_skin_node_id(&self, node_id: GameNodeId) -> Option<GameNodeId> {
        self.nodes
            .iter()
            .flat_map(|(node, _)| node)
            .filter_map(|node| {
                node.skin_index
                    .map(|skin_index| (node.id, &self.skins[skin_index]))
            })
            .find_map(|(skin_node_id, skin)| {
                skin.bone_node_ids
                    .contains(&node_id)
                    .then_some(skin_node_id)
            })
    }

    pub fn get_node_ancestry_list(&self, node_id: GameNodeId) -> Vec<GameNodeId> {
        let GameNodeId(node_index, _) = node_id;
        get_node_ancestry_list(node_index, &self.parent_index_map)
            .iter()
            .copied()
            .map(|node_index| GameNodeId(node_index, self.nodes[node_index].1))
            .collect()
    }

    pub fn get_skeleton_node_ancestry_list(
        &self,
        node_id: GameNodeId,
        skeleton_root_node_id: GameNodeId,
    ) -> Vec<GameNodeId> {
        let GameNodeId(node_index, _) = node_id;
        let GameNodeId(skeleton_root_node_index, _) = skeleton_root_node_id;
        match self
            .skeleton_parent_index_maps
            .get(&skeleton_root_node_index)
        {
            Some(skeleton_parent_index_map) => {
                get_node_ancestry_list(node_index, skeleton_parent_index_map)
                    .iter()
                    .copied()
                    .map(|node_index| GameNodeId(node_index, self.nodes[node_index].1))
                    .collect()
            }
            None => Vec::new(),
        }
    }

    // TODO: should this return an option?
    pub fn get_global_transform_for_node(
        &self,
        node_id: GameNodeId,
    ) -> crate::transform::Transform {
        let node_ancestry_list = self.get_node_ancestry_list(node_id);
        node_ancestry_list
            .iter()
            .rev()
            .fold(crate::transform::Transform::new(), |acc, node_id| {
                acc * self.get_node(*node_id).unwrap().transform
            })
    }

    pub fn add_node(&mut self, node: GameNodeDesc) -> &GameNode {
        let GameNodeDesc {
            transform,
            skin_index,
            mesh,
            name,
        } = node;
        let empty_node = self
            .nodes
            .iter()
            .enumerate()
            .find_map(|(node_index, node)| {
                if node.0.is_none() {
                    Some((node_index, node.1))
                } else {
                    None
                }
            });
        match empty_node {
            Some((empty_node_index, empty_node_gen)) => {
                let new_gen = empty_node_gen + 1;
                let new_node = GameNode {
                    transform,
                    skin_index,
                    mesh,
                    name,
                    id: GameNodeId(empty_node_index, new_gen),
                };
                self.nodes[empty_node_index] = (Some(new_node), new_gen);
                self.nodes[empty_node_index].0.as_ref().unwrap()
            }
            None => {
                let new_node = GameNode {
                    transform,
                    skin_index,
                    mesh,
                    name,
                    id: GameNodeId(self.nodes.len(), 0),
                };
                self.nodes.push((Some(new_node), 0));
                self.nodes[self.nodes.len() - 1].0.as_ref().unwrap()
            }
        }
    }

    pub fn get_node(&self, node_id: GameNodeId) -> Option<&GameNode> {
        let GameNodeId(node_index, node_gen) = node_id;
        let (actual_node, actual_node_gen) = &self.nodes[node_index];
        if *actual_node_gen == node_gen {
            actual_node.as_ref()
        } else {
            None
        }
    }

    pub fn get_node_mut(&mut self, node_id: GameNodeId) -> Option<&mut GameNode> {
        let GameNodeId(node_index, node_gen) = node_id;
        let (actual_node, actual_node_gen) = &mut self.nodes[node_index];
        if *actual_node_gen == node_gen {
            actual_node.as_mut()
        } else {
            None
        }
    }

    pub fn _get_node_by_index(&mut self, node_index: usize) -> Option<&GameNode> {
        self.nodes[node_index].0.as_ref()
    }

    pub fn _get_node_mut_by_index(&mut self, node_index: usize) -> Option<&mut GameNode> {
        self.nodes[node_index].0.as_mut()
    }

    pub fn remove_node(&mut self, node_id: GameNodeId) {
        // make sure it still exists
        if let Some(node) = self.get_node(node_id) {
            let GameNodeId(node_index, _) = node.id;
            self.nodes[node_index].0.take();
            self.parent_index_map.remove(&node_index);
            let child_entry_option =
                self.parent_index_map
                    .iter()
                    .find_map(|(child_index, parent_index)| {
                        (*parent_index == node_index).then_some((*child_index, *parent_index))
                    });
            if let Some((child_index, _)) = child_entry_option {
                self.parent_index_map.remove(&child_index);
            }
            self.rebuild_skeleton_parent_index_maps();
        }
    }

    pub fn _get_node_parent(&self, node_id: GameNodeId) -> Option<GameNodeId> {
        let GameNodeId(node_index, _) = node_id;
        self.parent_index_map
            .get(&node_index)
            .and_then(|parent_node_index| self.nodes[*parent_node_index].0.as_ref())
            .map(|node| node.id)
    }

    pub fn set_node_parent(&mut self, node_id: GameNodeId, parent_id: GameNodeId) {
        if let (Some(node), Some(parent_node)) = (self.get_node(node_id), self.get_node(parent_id))
        {
            let GameNodeId(node_index, _) = node.id();
            let GameNodeId(parent_node_index, _) = parent_node.id();
            self.parent_index_map.insert(node_index, parent_node_index);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|(node, _)| node.is_some()).count()
    }

    pub fn nodes(&self) -> impl Iterator<Item = &GameNode> {
        self.nodes.iter().flat_map(|(node, _)| node)
    }

    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut GameNode> {
        self.nodes.iter_mut().flat_map(|(node, _)| node)
    }
}

pub fn get_node_ancestry_list(
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

impl GameNodeId {
    pub fn _raw(&self) -> (usize, usize) {
        (self.0, self.1)
    }
}

impl Default for GameNodeDesc {
    fn default() -> Self {
        Self {
            transform: crate::transform::Transform::new(),
            skin_index: None,
            mesh: None,
            name: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameNodeDescBuilder {
    transform: crate::transform::Transform,
    skin_index: Option<usize>,
    mesh: Option<GameNodeMesh>,
    name: Option<String>,
}

impl GameNodeDescBuilder {
    pub fn new() -> Self {
        let GameNodeDesc {
            transform,
            skin_index,
            mesh,
            name,
        } = GameNodeDesc::default();
        Self {
            transform,
            skin_index,
            mesh,
            name,
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

    pub fn name(mut self, name: Option<String>) -> Self {
        self.name = name;
        self
    }

    pub fn build(self) -> GameNodeDesc {
        GameNodeDesc {
            transform: self.transform,
            skin_index: self.skin_index,
            mesh: self.mesh,
            name: self.name,
        }
    }
}

impl GameNodeMesh {
    pub fn from_pbr_mesh_index(pbr_mesh_index: usize) -> Self {
        Self {
            mesh_indices: vec![pbr_mesh_index],
            mesh_type: GameNodeMeshType::Pbr {
                material_override: None,
            },
            wireframe: false,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn removing_nodes_invalidates_ids() {
        let mut scene = Scene::new(vec![], vec![], vec![], HashMap::new());

        let node_1 = scene.add_node(GameNodeDesc::default());
        let node_1_id = node_1.id();

        let node_2 = scene.add_node(GameNodeDesc::default());
        let node_2_id = node_2.id();

        assert_node_exists(&scene, node_1_id);
        assert_node_exists(&scene, node_2_id);

        scene.remove_node(node_1_id);

        assert_node_doesnt_exist(&scene, node_1_id);
        assert_node_exists(&scene, node_2_id);

        let node_3 = scene.add_node(GameNodeDesc::default());
        let node_3_id = node_3.id();

        assert_node_doesnt_exist(&scene, node_1_id);
        assert_node_exists(&scene, node_2_id);
        assert_node_exists(&scene, node_3_id);
        assert_eq!(scene.nodes.len(), 2);

        scene.remove_node(node_2_id);

        assert_node_doesnt_exist(&scene, node_1_id);
        assert_node_doesnt_exist(&scene, node_2_id);
        assert_node_exists(&scene, node_3_id);
    }

    fn assert_node_exists(scene: &Scene, node_id: GameNodeId) {
        assert_eq!(scene.get_node(node_id).map(|node| node.id), Some(node_id));
    }

    fn assert_node_doesnt_exist(scene: &Scene, node_id: GameNodeId) {
        assert_eq!(scene.get_node(node_id).map(|node| node.id), None);
    }
}
