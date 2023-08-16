use crate::game::*;
use crate::mesh::*;
use crate::physics::*;
use crate::renderer::*;
use crate::scene::*;

use glam::Vec4;

pub struct Character {
    root_node_id: GameNodeId,
    skin_index: usize,
    collision_box_nodes: Vec<GameNodeId>,
    collision_box_colliders: Vec<ColliderHandle>,
    collision_debug_mesh_index: usize,
    is_displaying_collision_boxes: bool,
}

impl Character {
    pub fn new(
        scene: &mut Scene,
        physics_state: &mut PhysicsState,
        renderer_base: &BaseRenderer,
        renderer_data: &mut RendererData,
        root_node_id: GameNodeId,
        skin_index: usize,
        cube_mesh: &BasicMesh,
    ) -> Self {
        let collision_debug_mesh_index =
            Renderer::bind_basic_transparent_mesh(renderer_base, renderer_data, cube_mesh);
        let mut result = Self {
            root_node_id,
            skin_index,
            collision_box_nodes: vec![],
            collision_box_colliders: vec![],
            collision_debug_mesh_index,
            is_displaying_collision_boxes: false,
        };
        result.update(scene, physics_state);
        result
    }

    pub fn update(&mut self, scene: &mut Scene, physics_state: &mut PhysicsState) {
        let root_node_global_transform: crate::transform::Transform =
            scene.get_global_transform_for_node(self.root_node_id);
        let should_fill_collision_boxes = self.collision_box_colliders.is_empty();
        if let Some((skin_node_id, first_skin_bounding_box_transforms)) =
            scene.skins.get(self.skin_index).and_then(|skin| {
                scene
                    .nodes()
                    .find(|node| node.skin_index == Some(self.skin_index))
                    .map(|node| (node.id(), skin.bone_bounding_box_transforms.clone()))
            })
        {
            for (bone_index, bone_bounding_box_transform) in first_skin_bounding_box_transforms
                .iter()
                .copied()
                .enumerate()
            {
                let transform = {
                    let skin = &scene.skins[self.skin_index];
                    let skeleton_space_transform = {
                        let node_ancestry_list = scene.get_skeleton_node_ancestry_list(
                            skin.bone_node_ids[bone_index],
                            skin_node_id,
                        );

                        // goes from the bone's space into skeleton space given parent hierarchy
                        let bone_space_to_skeleton_space = node_ancestry_list
                            .iter()
                            .rev()
                            .fold(crate::transform::Transform::IDENTITY, |acc, node_id| {
                                acc * scene.get_node(*node_id).unwrap().transform
                            });
                        bone_space_to_skeleton_space
                    };
                    root_node_global_transform
                        * skeleton_space_transform
                        * bone_bounding_box_transform
                };
                let transform_decomposed = transform.decompose();

                if should_fill_collision_boxes {
                    self.collision_box_nodes
                        .push(scene.add_node(Default::default()).id());
                    let box_scale = transform_decomposed.scale;
                    let the_box = ColliderBuilder::cuboid(box_scale.x, box_scale.y, box_scale.z)
                        .collision_groups(
                            InteractionGroups::all()
                                .with_memberships(!COLLISION_GROUP_PLAYER_UNSHOOTABLE),
                        )
                        .build();
                    self.collision_box_colliders
                        .push(physics_state.collider_set.insert(the_box));
                }

                if let Some(node) = scene.get_node_mut(self.collision_box_nodes[bone_index]) {
                    node.transform = transform;
                }
                if let Some(collider) = physics_state
                    .collider_set
                    .get_mut(self.collision_box_colliders[bone_index])
                {
                    collider.set_position(Isometry::from_parts(
                        nalgebra::Translation3::new(
                            transform_decomposed.position.x,
                            transform_decomposed.position.y,
                            transform_decomposed.position.z,
                        ),
                        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            transform_decomposed.rotation.w,
                            transform_decomposed.rotation.x,
                            transform_decomposed.rotation.y,
                            transform_decomposed.rotation.z,
                        )),
                    ))
                }
            }
        }
    }

    pub fn handle_hit(&self, scene: &mut Scene, collider_handle: ColliderHandle) {
        if let Some(bone_index) = self.collision_box_colliders.iter().enumerate().find_map(
            |(bone_index, bone_collider_handle)| {
                (*bone_collider_handle == collider_handle).then_some(bone_index)
            },
        ) {
            if let Some(node) = scene.get_node_mut(self.collision_box_nodes[bone_index]) {
                node.mesh = Some(GameNodeMesh {
                    mesh_indices: node
                        .mesh
                        .as_mut()
                        .map(|mesh| {
                            if mesh.mesh_indices.contains(&self.collision_debug_mesh_index) {
                                mesh.mesh_indices.clone()
                            } else {
                                let mut res = mesh.mesh_indices.clone();
                                res.push(self.collision_debug_mesh_index);
                                res
                            }
                        })
                        .unwrap_or_else(|| vec![self.collision_debug_mesh_index]),
                    mesh_type: GameNodeMeshType::Transparent {
                        color: Vec4::new(1.0, 0.0, 0.0, 0.3),
                        premultiplied_alpha: false,
                    },
                    ..Default::default()
                })
            }
        }
    }

    fn enable_collision_box_display(&mut self, scene: &mut Scene) {
        for node_id in self.collision_box_nodes.iter().cloned() {
            if let Some(node) = scene.get_node_mut(node_id) {
                node.mesh = Some(GameNodeMesh {
                    mesh_indices: node
                        .mesh
                        .as_mut()
                        .map(|mesh| {
                            if mesh.mesh_indices.contains(&self.collision_debug_mesh_index) {
                                mesh.mesh_indices.clone()
                            } else {
                                let mut res = mesh.mesh_indices.clone();
                                res.push(self.collision_debug_mesh_index);
                                res
                            }
                        })
                        .unwrap_or_else(|| vec![self.collision_debug_mesh_index]),
                    mesh_type: GameNodeMeshType::Transparent {
                        color: Vec4::new(rand::random(), rand::random(), rand::random(), 0.3),
                        premultiplied_alpha: false,
                    },
                    ..Default::default()
                })
            }
        }
        self.is_displaying_collision_boxes = true;
    }

    fn disable_collision_box_display(&mut self, scene: &mut Scene) {
        for node_id in self.collision_box_nodes.iter().cloned() {
            if let Some(node) = scene.get_node_mut(node_id) {
                node.mesh = None;
            }
        }
        self.is_displaying_collision_boxes = false;
    }

    pub fn toggle_collision_box_display(&mut self, scene: &mut Scene) {
        if self.is_displaying_collision_boxes {
            self.disable_collision_box_display(scene);
        } else {
            self.enable_collision_box_display(scene);
        }
    }
}
