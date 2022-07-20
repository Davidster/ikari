use cgmath::Vector3;

use super::*;

pub struct Character {
    root_node_id: GameNodeId,
    skin_index: usize,
    collision_box_nodes: Vec<GameNodeId>,
    collision_debug_mesh_index: usize,
    is_displaying_collision_boxes: bool,
}

impl Character {
    pub fn new(
        scene: &mut Scene,
        renderer_state: &mut RendererState,
        root_node_id: GameNodeId,
        skin_index: usize,
        cube_mesh: &BasicMesh,
    ) -> Self {
        let collision_debug_mesh_index = renderer_state.bind_basic_unlit_mesh(cube_mesh);

        let bone_count = scene.skins[skin_index].bone_node_ids.len();
        let mut collision_box_nodes: Vec<GameNodeId> = Vec::new();
        (0..bone_count).for_each(|_| {
            collision_box_nodes.push(scene.add_node(Default::default()).id());
        });

        let result = Self {
            root_node_id,
            skin_index,
            collision_box_nodes,
            collision_debug_mesh_index,
            is_displaying_collision_boxes: false,
        };
        result.update(scene);
        result
    }

    pub fn update(&self, scene: &mut Scene) {
        let root_node_global_transform = scene.get_global_transform_for_node(self.root_node_id);
        if let Some((skin_node_id, first_skin_bounding_box_transforms)) =
            scene.skins.get(self.skin_index).map(|skin| {
                (
                    scene
                        .nodes()
                        .find(|node| node.skin_index == Some(self.skin_index))
                        .unwrap() // TODO: dont unwrap here?
                        .id(),
                    skin.bone_bounding_box_transforms.clone(),
                )
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
                            .fold(crate::transform::Transform::new(), |acc, node_id| {
                                acc * scene.get_node(*node_id).unwrap().transform
                            });
                        bone_space_to_skeleton_space
                    };
                    root_node_global_transform
                        * skeleton_space_transform
                        * bone_bounding_box_transform
                };

                if let Some(node) = scene.get_node_mut(self.collision_box_nodes[bone_index]) {
                    node.transform = transform;
                }
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
                    mesh_type: GameNodeMeshType::Unlit {
                        color: Vector3::new(rand::random(), rand::random(), rand::random()),
                    },
                    wireframe: true,
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
