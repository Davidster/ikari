use glam::Vec4;
use ikari::mesh::BasicMesh;
use ikari::physics::PhysicsState;
use ikari::renderer::{BaseRenderer, Renderer, RendererConstantData, RendererData};
use ikari::scene::{GameNodeId, GameNodeVisual, Material, Scene};

use ikari::physics::rapier3d_f64::prelude::*;
use ikari::transform::Transform;

use crate::game::COLLISION_GROUP_PLAYER_UNSHOOTABLE;

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
        renderer_constant_data: &RendererConstantData,
        renderer_data: &mut RendererData,
        root_node_id: GameNodeId,
        skin_index: usize,
    ) -> Self {
        let mut res = Self {
            root_node_id,
            skin_index,
            collision_box_nodes: vec![],
            collision_box_colliders: vec![],
            collision_debug_mesh_index: renderer_constant_data.cube_mesh_index,
            is_displaying_collision_boxes: false,
        };
        res.update(scene, physics_state);
        res
    }

    pub fn update(&mut self, scene: &mut Scene, physics_state: &mut PhysicsState) {
        let root_node_global_transform: Transform =
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
                            .fold(Transform::IDENTITY, |acc, node_id| {
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
                    let the_box = ColliderBuilder::cuboid(
                        box_scale.x as f64,
                        box_scale.y as f64,
                        box_scale.z as f64,
                    )
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
                            transform_decomposed.position.x as f64,
                            transform_decomposed.position.y as f64,
                            transform_decomposed.position.z as f64,
                        ),
                        nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                            transform_decomposed.rotation.w as f64,
                            transform_decomposed.rotation.x as f64,
                            transform_decomposed.rotation.y as f64,
                            transform_decomposed.rotation.z as f64,
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
                node.visual = Some(GameNodeVisual {
                    mesh_index: self.collision_debug_mesh_index,
                    material: Material::Transparent {
                        color: Vec4::new(1.0, 0.0, 0.0, 0.3),
                        premultiplied_alpha: false,
                    },
                    wireframe: false,
                    cullable: true,
                })
            }
        }
    }

    fn enable_collision_box_display(&mut self, scene: &mut Scene) {
        for node_id in self.collision_box_nodes.iter().cloned() {
            if let Some(node) = scene.get_node_mut(node_id) {
                node.visual = Some(GameNodeVisual {
                    mesh_index: self.collision_debug_mesh_index,
                    material: Material::Transparent {
                        color: Vec4::new(rand::random(), rand::random(), rand::random(), 0.3),
                        premultiplied_alpha: false,
                    },
                    wireframe: false,
                    cullable: true,
                })
            }
        }
        self.is_displaying_collision_boxes = true;
    }

    fn disable_collision_box_display(&mut self, scene: &mut Scene) {
        for node_id in self.collision_box_nodes.iter().cloned() {
            if let Some(node) = scene.get_node_mut(node_id) {
                node.visual = None;
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
