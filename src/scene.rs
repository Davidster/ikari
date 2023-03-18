use crate::animation::*;
use crate::collisions::*;
use crate::mesh::*;
use crate::renderer::*;

use std::{collections::HashMap, hash::BuildHasherDefault};

use glam::f32::{Mat4, Vec3, Vec4};
use twox_hash::XxHash64;

#[derive(Debug, Default)]
pub struct Scene {
    nodes: Vec<(Option<GameNode>, usize)>, // (node, generation number). None means the node was removed from the scene
    // node_transforms: Vec<Mat4>,
    global_node_transforms: Vec<crate::transform::Transform>,
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
    // skeleton skin node index -> parent_index_map
    skeleton_parent_index_maps:
        HashMap<u32, HashMap<u32, u32, BuildHasherDefault<XxHash64>>, BuildHasherDefault<XxHash64>>,
}

#[derive(Debug, Clone)]
pub struct GameNodeDesc {
    pub transform: crate::transform::Transform,
    pub skin_index: Option<usize>,
    pub mesh: Option<GameNodeMesh>,
    pub name: Option<String>,
    pub parent_id: Option<GameNodeId>,
}

#[derive(Debug, Clone)]
pub struct GameNode {
    pub transform: crate::transform::Transform,
    pub skin_index: Option<usize>,
    pub mesh: Option<GameNodeMesh>,
    pub name: Option<String>,
    pub parent_id: Option<GameNodeId>,
    id: GameNodeId,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct GameNodeId(u32, usize); // (index into GameScene::nodes array, generation num)

#[derive(Debug, Clone)]
pub struct GameNodeMesh {
    pub mesh_type: GameNodeMeshType,
    pub mesh_indices: Vec<usize>,
    pub wireframe: bool,
    pub cullable: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum GameNodeMeshType {
    Pbr {
        material_override: Option<DynamicPbrParams>,
    },
    Unlit {
        color: Vec3,
    },
}

#[derive(Debug, Clone)]
pub struct Skin {
    pub node_id: GameNodeId,
    pub bone_node_ids: Vec<GameNodeId>,
    pub bone_inverse_bind_matrices: Vec<Mat4>,
    // each transform moves a 2x2x2 box centered at the origin
    // such that it surrounds the bone's vertices in bone space
    pub bone_bounding_box_transforms: Vec<crate::transform::Transform>,
}

#[derive(Debug, Clone)]
pub struct IndexedGameNodeDesc {
    pub transform: crate::transform::Transform,
    pub skin_index: Option<usize>,
    pub mesh: Option<GameNodeMesh>,
    pub name: Option<String>,
    pub parent_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct IndexedSkin {
    pub bone_node_indices: Vec<usize>,
    pub bone_inverse_bind_matrices: Vec<Mat4>,
    pub bone_bounding_box_transforms: Vec<crate::transform::Transform>,
}

#[derive(Debug)]
pub struct IndexedAnimation {
    pub name: Option<String>,
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

const MAX_NODE_HIERARCHY_LEVELS: usize = 32;

impl Scene {
    pub fn new(
        nodes_desc: Vec<IndexedGameNodeDesc>,
        indexed_skins: Vec<IndexedSkin>,
        animations: Vec<IndexedAnimation>,
    ) -> Self {
        let animations: Vec<_> = animations
            .iter()
            .map(|indexed_animation| Animation {
                name: indexed_animation.name.clone(),
                length_seconds: indexed_animation.length_seconds,
                speed: 1.0,
                channels: indexed_animation
                    .channels
                    .iter()
                    .map(|indexed_channel| Channel {
                        node_id: GameNodeId(indexed_channel.node_index.try_into().unwrap(), 0),
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
            global_node_transforms: Vec::new(),
            skins: Vec::new(),
            animations,
            skeleton_parent_index_maps: Default::default(),
        };

        nodes_desc.iter().for_each(|node_desc| {
            let IndexedGameNodeDesc {
                transform,
                skin_index,
                mesh,
                name,
                parent_index,
            } = node_desc.clone();
            scene.add_node(GameNodeDesc {
                transform,
                skin_index,
                mesh,
                name,
                parent_id: parent_index
                    .map(|parent_index| GameNodeId((parent_index).try_into().unwrap(), 0)),
            });
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
                        .map(|node_index| GameNodeId((*node_index).try_into().unwrap(), 0))
                        .collect(),
                    bone_inverse_bind_matrices: indexed_skin.bone_inverse_bind_matrices.clone(),
                    bone_bounding_box_transforms: indexed_skin.bone_bounding_box_transforms.clone(),
                }
            })
            .collect();

        scene.rebuild_skeleton_parent_index_maps();

        scene.recompute_global_node_transforms();

        scene
    }

    fn rebuild_skeleton_parent_index_maps(&mut self) {
        self.skeleton_parent_index_maps = Default::default();
        for skin in &self.skins {
            let skeleton_parent_index_map: HashMap<u32, u32, BuildHasherDefault<XxHash64>> = skin
                .bone_node_ids
                .iter()
                .filter_map(|GameNodeId(bone_node_index, _)| {
                    let (node, _) = &self.nodes[*bone_node_index as usize];
                    node.as_ref().and_then(|node| node.parent_id).map(
                        |GameNodeId(parent_node_index, _)| (*bone_node_index, parent_node_index),
                    )
                })
                .collect();
            self.skeleton_parent_index_maps
                .insert(skin.node_id.0, skeleton_parent_index_map);
        }
    }

    #[profiling::function]
    pub fn recompute_global_node_transforms(&mut self) {
        if self.nodes.len() <= self.global_node_transforms.len() {
            self.global_node_transforms.truncate(self.nodes.len());
        } else {
            // eliminate potential allocations in the subsequent push() calls
            self.global_node_transforms
                .reserve_exact(self.nodes.len() - self.global_node_transforms.len());
        }
        for (node_index, (node, _)) in self.nodes.iter().enumerate() {
            let transform = node
                .as_ref()
                .map(|node| self.get_global_transform_for_node_internal(node.id()))
                .unwrap_or(crate::transform::Transform::IDENTITY);
            if node_index < self.global_node_transforms.len() {
                self.global_node_transforms[node_index] = transform;
            } else {
                self.global_node_transforms.push(transform);
            }
        }
    }

    pub fn merge_scene(
        &mut self,
        renderer_data: &mut RendererStatePublicData,
        mut other_scene: Scene,
        mut other_render_buffers: RenderBuffers,
    ) {
        let pbr_mesh_index_offset = renderer_data.binded_pbr_meshes.len();
        let unlit_mesh_index_offset = renderer_data.binded_unlit_meshes.len();

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

        renderer_data
            .binded_pbr_meshes
            .append(&mut other_render_buffers.binded_pbr_meshes);
        renderer_data
            .binded_unlit_meshes
            .append(&mut other_render_buffers.binded_unlit_meshes);
        renderer_data
            .binded_wireframe_meshes
            .append(&mut other_render_buffers.binded_wireframe_meshes);
        renderer_data
            .textures
            .append(&mut other_render_buffers.textures);
        let skin_index_offset = self.skins.len();
        let node_index_offset = self.nodes.len();
        let convert_node_id = |old_node_id| {
            let GameNodeId(old_index, _) = old_node_id;
            let new_index = old_index + node_index_offset as u32;
            GameNodeId(new_index, 0)
        };
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
                if let Some(ref mut parent_id) = node.parent_id {
                    *parent_id = convert_node_id(*parent_id);
                }
                node.id = convert_node_id(node.id);
            }
        }
        for mut skin in &mut other_scene.skins {
            skin.bone_node_ids = skin
                .bone_node_ids
                .iter()
                .copied()
                .map(convert_node_id)
                .collect();
            skin.node_id = convert_node_id(skin.node_id);
        }
        for animation in &mut other_scene.animations {
            for channel in &mut animation.channels {
                channel.node_id = convert_node_id(channel.node_id);
            }
        }

        self.nodes.append(&mut other_scene.nodes);
        self.skins.append(&mut other_scene.skins);
        self.animations.append(&mut other_scene.animations);
        self.rebuild_skeleton_parent_index_maps();
    }

    pub fn get_node_bounding_sphere(
        &self,
        node_id: GameNodeId,
        renderer_state_data: &RendererStatePublicData,
    ) -> Option<Sphere> {
        self.get_node(node_id)
            .and_then(|node| node.mesh.as_ref())
            .map(|mesh| {
                build_node_bounding_sphere(
                    mesh,
                    &self.get_global_transform_for_node(node_id),
                    renderer_state_data,
                )
            })
    }

    pub fn get_node_bounding_sphere_opt(
        &self,
        node_id: GameNodeId,
        renderer_data: &RendererStatePublicData,
    ) -> Option<Sphere> {
        self.get_node(node_id)
            .and_then(|node| node.mesh.as_ref())
            .map(|mesh| {
                build_node_bounding_sphere(
                    mesh,
                    &self.get_global_transform_for_node_opt(node_id),
                    renderer_data,
                )
            })
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

    fn get_node_ancestry_list(&self, node_id: GameNodeId) -> impl Iterator<Item = GameNodeId> + '_ {
        std::iter::successors(Some(node_id), |node_id| {
            self.get_node(*node_id).and_then(|node| node.parent_id)
        })
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
                std::iter::successors(Some(node_index), |node_index| {
                    skeleton_parent_index_map.get(node_index).copied()
                })
                .map(|node_index| GameNodeId(node_index, self.nodes[node_index as usize].1))
                .collect()
            }
            None => Vec::new(),
        }
    }

    // TODO: should this return an option?
    #[profiling::function]
    pub fn get_global_transform_for_node(
        &self,
        node_id: GameNodeId,
    ) -> crate::transform::Transform {
        let node_ancestry_list: Vec<_> = self.get_node_ancestry_list(node_id).collect();
        node_ancestry_list.iter().rev().fold(
            crate::transform::Transform::IDENTITY,
            |acc, node_id| {
                let GameNodeId(node_index, _) = node_id;
                let (node, _) = &self.nodes[*node_index as usize];
                acc * node.as_ref().unwrap().transform
            },
        )
    }

    // #[profiling::function]
    fn get_global_transform_for_node_internal(
        &self,
        node_id: GameNodeId,
    ) -> crate::transform::Transform {
        let mut node_ancestry_list: [u32; MAX_NODE_HIERARCHY_LEVELS] =
            [0; MAX_NODE_HIERARCHY_LEVELS];
        let mut ancestry_length = 0;
        for (i, node_id) in self.get_node_ancestry_list(node_id).enumerate() {
            let GameNodeId(node_index, _) = node_id;
            node_ancestry_list[i] = node_index;
            ancestry_length += 1;
        }
        let mut ancestry_transforms = (0..ancestry_length).rev().map(|ancestry_list_index| {
            let node_index = node_ancestry_list[ancestry_list_index];
            let (node, _) = &self.nodes[node_index as usize];
            node.as_ref().unwrap().transform
        });
        let mut acc: crate::transform::Transform = ancestry_transforms.next().unwrap();
        for ancestry_transform in ancestry_transforms {
            acc = acc * ancestry_transform
        }
        acc
    }

    pub fn get_global_transform_for_node_opt(
        &self,
        node_id: GameNodeId,
    ) -> crate::transform::Transform {
        let GameNodeId(node_index, _) = node_id;
        self.global_node_transforms[node_index as usize]
    }

    pub fn add_node(&mut self, node: GameNodeDesc) -> &GameNode {
        let GameNodeDesc {
            transform,
            skin_index,
            mesh,
            name,
            parent_id,
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
                    id: GameNodeId(empty_node_index.try_into().unwrap(), new_gen),
                    parent_id,
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
                    id: GameNodeId(self.nodes.len().try_into().unwrap(), 0),
                    parent_id,
                };
                self.nodes.push((Some(new_node), 0));
                self.nodes[self.nodes.len() - 1].0.as_ref().unwrap()
            }
        }
    }

    pub fn get_node(&self, node_id: GameNodeId) -> Option<&GameNode> {
        let GameNodeId(node_index, node_gen) = node_id;
        let (actual_node, actual_node_gen) = &self.nodes[node_index as usize];
        if *actual_node_gen == node_gen {
            actual_node.as_ref()
        } else {
            None
        }
    }

    pub fn get_node_unchecked(&self, node_id: GameNodeId) -> &GameNode {
        let GameNodeId(node_index, _) = node_id;
        let (actual_node, _) = &self.nodes[node_index as usize];
        actual_node.as_ref().unwrap()
    }

    pub fn get_node_mut(&mut self, node_id: GameNodeId) -> Option<&mut GameNode> {
        let GameNodeId(node_index, node_gen) = node_id;
        let (actual_node, actual_node_gen) = &mut self.nodes[node_index as usize];
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
            self.nodes[node_index as usize].0.take();
            self.rebuild_skeleton_parent_index_maps();
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

fn build_node_bounding_sphere(
    mesh: &GameNodeMesh,
    global_transform: &crate::transform::Transform,
    renderer_data: &RendererStatePublicData,
) -> Sphere {
    let global_node_scale = global_transform.scale();
    let largest_axis_scale = global_node_scale
        .x
        .max(global_node_scale.y)
        .max(global_node_scale.z);

    let mut mesh_aabbs = mesh
        .mesh_indices
        .iter()
        .copied()
        .map(|mesh_index| match mesh.mesh_type {
            GameNodeMeshType::Pbr { .. } => {
                renderer_data.binded_pbr_meshes[mesh_index]
                    .geometry_buffers
                    .bounding_box
            }
            GameNodeMeshType::Unlit { .. } => {
                renderer_data.binded_unlit_meshes[mesh_index].bounding_box
            }
        });

    let mut merged_aabb = mesh_aabbs.next().unwrap();
    for aabb in mesh_aabbs {
        for i in 0..3 {
            merged_aabb.min[i] = aabb.min.x.min(merged_aabb.min[i]);
            merged_aabb.max[i] = aabb.max.x.max(merged_aabb.max[i]);
        }
    }

    let global_mat4 = Mat4::from(*global_transform);

    let transform_point = |point: Vec3| {
        let transformed = global_mat4 * Vec4::new(point.x, point.y, point.z, 1.0);
        Vec3::new(transformed.x, transformed.y, transformed.z)
    };

    let origin = transform_point((merged_aabb.max + merged_aabb.min) / 2.0);

    let half_length = (merged_aabb.max - merged_aabb.min) / 2.0;
    let radius = largest_axis_scale * half_length.length();

    Sphere { origin, radius }
}

impl GameNode {
    pub fn id(&self) -> GameNodeId {
        self.id
    }
}

impl GameNodeId {
    pub fn _raw(&self) -> (u32, usize) {
        (self.0, self.1)
    }
}

impl Default for GameNodeDesc {
    fn default() -> Self {
        Self {
            transform: crate::transform::Transform::IDENTITY,
            skin_index: None,
            mesh: None,
            name: None,
            parent_id: None,
        }
    }
}

impl Default for GameNodeMesh {
    fn default() -> Self {
        Self {
            mesh_type: GameNodeMeshType::Unlit {
                color: Vec3::default(),
            },
            mesh_indices: vec![],
            wireframe: false,
            cullable: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameNodeDescBuilder {
    transform: crate::transform::Transform,
    skin_index: Option<usize>,
    mesh: Option<GameNodeMesh>,
    name: Option<String>,
    parent_id: Option<GameNodeId>,
}

impl GameNodeDescBuilder {
    pub fn new() -> Self {
        let GameNodeDesc {
            transform,
            skin_index,
            mesh,
            name,
            parent_id,
        } = GameNodeDesc::default();
        Self {
            transform,
            skin_index,
            mesh,
            name,
            parent_id,
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

    pub fn parent_id(mut self, parent_id: Option<GameNodeId>) -> Self {
        self.parent_id = parent_id;
        self
    }

    pub fn build(self) -> GameNodeDesc {
        GameNodeDesc {
            transform: self.transform,
            skin_index: self.skin_index,
            mesh: self.mesh,
            name: self.name,
            parent_id: self.parent_id,
        }
    }
}

impl Default for GameNodeDescBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: cache the merged aabb, make mesh_indices no longer public, update the aabb whenever mesh_indices changes
impl GameNodeMesh {
    pub fn from_pbr_mesh_index(pbr_mesh_index: usize) -> Self {
        Self {
            mesh_indices: vec![pbr_mesh_index],
            mesh_type: GameNodeMeshType::Pbr {
                material_override: None,
            },
            wireframe: false,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn removing_nodes_invalidates_ids() {
        let mut scene = Scene::new(vec![], vec![], vec![]);

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
