use crate::collisions::*;
use crate::renderer::*;
use crate::scene::*;

use glam::f32::Vec3;
use smallvec::{smallvec, SmallVec};

/*
    Currently not used.
    Octree implementation that can be used to accelerate intersection tests between
    objects in the world.
*/

type Aabb = crate::collisions::Aabb;

const ROOT_AABB_SIZE: f32 = 4000.0;
const MAX_DEPTH: u8 = 5;
const K: f32 = 2.0; // looseness factor, set to 1.0 to disable loosening
const K_FACTOR: f32 = (K - 1.0) / 2.0;

#[derive(Debug, Clone)]
pub struct SceneTree {
    node_list: Vec<SceneTreeNode>,
}

#[derive(Debug, Clone)]
pub struct SceneTreeNode {
    base_aabb: Aabb,
    game_nodes: SmallVec<[(GameNodeId, Sphere); 1]>,
    children: Option<u32>,
}

impl Default for SceneTreeNode {
    fn default() -> Self {
        Self {
            base_aabb: Aabb {
                min: Vec3::default(),
                max: Vec3::default(),
            },
            game_nodes: smallvec![],
            children: None,
        }
    }
}

#[allow(dead_code)]
#[profiling::function]
pub fn build_scene_tree(
    scene: &Scene,
    renderer: &Renderer,
    old_scene_tree: Option<SceneTree>,
) -> SceneTree {
    let mut scene_tree = SceneTree::new(old_scene_tree);

    for node in scene.nodes() {
        if node.visual.is_none() || !node.visual.as_ref().unwrap().cullable {
            continue;
        }
        if let Some(node_bounding_sphere) =
            scene.get_node_bounding_sphere_opt(node.id(), &renderer.data.lock().unwrap())
        {
            scene_tree.insert(node.id(), node_bounding_sphere);
        }
    }

    scene_tree
}

impl SceneTree {
    fn new(old_scene_tree: Option<SceneTree>) -> Self {
        let mut tree = old_scene_tree.unwrap_or(SceneTree { node_list: vec![] });
        tree.node_list.clear();

        // logger_log(&format!(
        //     "node_list capacity: {:?}",
        //     tree.node_list.capacity(),
        // ));

        let _offset = ROOT_AABB_SIZE * 0.02 * std::f32::consts::PI * Vec3::new(1.0, 1.0, 1.0);
        let root_aabb_max = ROOT_AABB_SIZE * Vec3::new(1.0, 1.0, 1.0);
        let root = SceneTreeNode {
            base_aabb: Aabb {
                min: -root_aabb_max, /*  + _offset */
                max: root_aabb_max,  /*  + _offset */
            },
            ..Default::default()
        };

        tree.node_list.push(root);

        tree
    }

    pub fn root(&self) -> &SceneTreeNode {
        &self.node_list[0]
    }

    pub fn root_mut(&mut self) -> &mut SceneTreeNode {
        &mut self.node_list[0]
    }

    fn insert(&mut self, node_id: GameNodeId, node_bounding_sphere: Sphere) {
        let root = self.root_mut();
        if !root.aabb().fully_contains_sphere(node_bounding_sphere) {
            log::warn!("WARNING Tried to insert a node that's not fully contained by the scene tree. Consider increasing size of the base scene tree. Sphere: {:?}, Root aabb: {:?}", node_bounding_sphere, root.base_aabb);
        }
        self.insert_internal(0, node_id, node_bounding_sphere, 0);
    }

    fn insert_internal(
        &mut self,
        node_index: usize,
        node_id: GameNodeId,
        node_bounding_sphere: Sphere,
        depth: u8,
    ) {
        let entry = (node_id, node_bounding_sphere);

        let first_new_child_index = self.node_list.len();
        let node = &mut self.node_list[node_index];

        if depth >= MAX_DEPTH {
            node.game_nodes.push(entry);
            return;
        }

        let mut fully_contained_index: Option<(usize, f32)> = None;
        let children_base_aabbs = node.base_aabb.subdivide();
        for (i, child_base_aabb) in children_base_aabbs.iter().enumerate() {
            let child_aabb = node.loosen_aabb(child_base_aabb);
            if node_bounding_sphere.radius < child_base_aabb.size().x / 2.0
                || child_aabb.fully_contains_sphere(node_bounding_sphere)
            {
                let distance2 =
                    (child_aabb.center() - node_bounding_sphere.center).length_squared();
                // pick the aabb whose center is closest to the object
                fully_contained_index = match fully_contained_index {
                    Some((other_i, other_dist2)) => Some(if distance2 < other_dist2 {
                        (i, distance2)
                    } else {
                        (other_i, other_dist2)
                    }),
                    None => Some((i, distance2)),
                };
            }
        }

        match (node.children, fully_contained_index) {
            (None, Some((fully_contained_index, _))) => {
                node.children = Some(first_new_child_index.try_into().unwrap());
                for base_aabb in children_base_aabbs {
                    self.node_list.push(SceneTreeNode {
                        base_aabb,
                        ..Default::default()
                    });
                }
                self.insert_internal(
                    first_new_child_index + fully_contained_index,
                    node_id,
                    node_bounding_sphere,
                    depth + 1,
                );
            }
            (Some(first_child_index), Some((fully_contained_index, _))) => {
                self.insert_internal(
                    first_child_index as usize + fully_contained_index,
                    node_id,
                    node_bounding_sphere,
                    depth + 1,
                );
            }
            _ => {
                node.game_nodes.push(entry);
            }
        }
    }

    #[allow(dead_code)]
    #[profiling::function]
    pub fn get_intersecting_nodes(&self, frustum: Frustum) -> Vec<GameNodeId> {
        let mut result: Vec<GameNodeId> = Vec::new();
        self.get_intersecting_nodes_impl(self.root(), frustum, &mut result);
        result
    }

    fn iter_children(&self, first_child_index: usize) -> impl Iterator<Item = &SceneTreeNode> {
        (0..8).map(move |child_index| &self.node_list[child_index + first_child_index])
    }

    fn get_intersecting_nodes_impl(
        &self,
        node: &SceneTreeNode,
        frustum: Frustum,
        acc: &mut Vec<GameNodeId>,
    ) {
        use IntersectionResult::*;

        match frustum.aabb_intersection_test(node.aabb()) {
            NotIntersecting => {}
            FullyContained => {
                self.get_all_nodes_impl(node, acc);
            }
            PartiallyIntersecting => {
                for (node_id, bounding_shere) in &node.game_nodes {
                    if frustum.aabb_intersection_test(bounding_shere.aabb()) != NotIntersecting {
                        acc.push(*node_id);
                    }
                }
                // acc.extend(node.game_nodes.iter().map(|(node_id, _)| node_id));
                if let Some(first_child_index) = node.children.as_ref() {
                    for child in self.iter_children(*first_child_index as usize) {
                        self.get_intersecting_nodes_impl(child, frustum, acc);
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn get_non_empty_node_count(&self, node: &SceneTreeNode) -> u32 {
        let mut total = 0_u32;
        if !node.game_nodes.is_empty() {
            total += 1;
        }
        if let Some(first_child_index) = node.children.as_ref() {
            for child in self.iter_children(*first_child_index as usize) {
                total += self.get_non_empty_node_count(child);
            }
        }
        total
    }

    fn _get_node_count(&self, node: &SceneTreeNode) -> u32 {
        let mut total = 0_u32;
        total += 1;
        if let Some(first_child_index) = node.children.as_ref() {
            for child in self.iter_children(*first_child_index as usize) {
                total += self._get_node_count(child);
            }
        }
        total
    }

    fn _get_game_node_count(&self, node: &SceneTreeNode) -> u32 {
        let mut total = 0_u32;
        total += node.game_nodes.len() as u32;
        if let Some(first_child_index) = node.children.as_ref() {
            for child in self.iter_children(*first_child_index as usize) {
                total += self._get_game_node_count(child);
            }
        }
        total
    }

    #[allow(dead_code)]
    #[profiling::function]
    pub fn to_aabb_list(&self) -> Vec<Aabb> {
        let mut list: Vec<Aabb> = Vec::new();
        self.to_aabb_list_impl(self.root(), &mut list);
        list
    }

    // includes all branch nodes and nodes that contain game nodes
    fn to_aabb_list_impl(&self, node: &SceneTreeNode, acc: &mut Vec<Aabb>) {
        if !node.game_nodes.is_empty() || node.children.as_ref().is_some() {
            acc.push(node.aabb());
        }
        if let Some(first_child_index) = node.children.as_ref() {
            for child in self.iter_children(*first_child_index as usize) {
                self.to_aabb_list_impl(child, acc);
            }
        }
    }

    #[profiling::function]
    pub fn _get_all_nodes(&self) -> Vec<GameNodeId> {
        let mut result: Vec<GameNodeId> = Vec::new();
        self.get_all_nodes_impl(self.root(), &mut result);
        result
    }

    fn get_all_nodes_impl(&self, node: &SceneTreeNode, acc: &mut Vec<GameNodeId>) {
        acc.extend(node.game_nodes.iter().map(|(node_id, _)| node_id));
        if let Some(first_child_index) = node.children.as_ref() {
            for child in self.iter_children(*first_child_index as usize) {
                self.get_all_nodes_impl(child, acc);
            }
        }
    }
}

impl SceneTreeNode {
    // it's a loose octree, so the base_aabb shouldn't be used for intersection tests, only subdivision
    // see Real-Time Collision Detection, section 7.3.6
    fn aabb(&self) -> Aabb {
        self.loosen_aabb(&self.base_aabb)
    }

    fn loosen_aabb(&self, aabb: &Aabb) -> Aabb {
        // aabb must be a cube!
        let extension_factor = (aabb.max.x - aabb.min.x) * K_FACTOR;
        let extension_factor_vector =
            Vec3::new(extension_factor, extension_factor, extension_factor);
        Aabb {
            min: aabb.min - extension_factor_vector,
            max: aabb.max + extension_factor_vector,
        }
    }
}
