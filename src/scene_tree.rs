use anyhow::{bail, Result};
use cgmath::Vector3;

use super::*;

#[derive(Debug, Clone)]
pub struct SceneTree {
    pub aabb: Aabb,
    pub children: Option<SceneTreeChildren>,
}

#[derive(Debug, Clone)]
pub enum SceneTreeChildren {
    Nodes(Vec<(GameNodeId, Sphere)>),
    SubTrees(Box<[SceneTree; 8]>),
}

use SceneTreeChildren::*;

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub origin: Vector3<f32>,
    pub radius: f32,
}

const BASE_AABB_SIZE: f32 = 1000.0;
const MAX_DEPTH: u8 = 10;
// const MIN_CELL_SIZE: f32 = BASE_AABB_SIZE / 2_u32.pow(MAX_DEPTH) as f32;

impl Default for SceneTree {
    fn default() -> Self {
        Self {
            aabb: Aabb {
                min: Vector3::zero(),
                max: Vector3::zero(),
            },
            children: None,
        }
    }
}

impl Aabb {
    fn volume(&self) -> f32 {
        let size = self.max - self.min;
        size.x * size.y * size.z
    }

    // taken from https://gamedev.stackexchange.com/questions/156870/how-do-i-implement-a-aabb-sphere-collision
    // ClosestPtPointAABB
    fn find_closest_surface_point(&self, p: Vector3<f32>) -> Vector3<f32> {
        let mut q: Vector3<f32> = Vector3::zero();
        for i in 0..3 {
            let mut v = p[i];
            if v < self.min[i] {
                v = self.min[i];
            }
            if v > self.max[i] {
                v = self.max[i];
            }
            q[i] = v;
        }
        q
    }

    fn contains_point(&self, point: Vector3<f32>) -> bool {
        self.min.x < point.x
            && self.max.x > point.x
            && self.min.y < point.y
            && self.max.y > point.y
            && self.min.z < point.z
            && self.max.z > point.z
    }

    // true if fully contains or partially contains
    fn contains_sphere(&self, sphere: Sphere) -> bool {
        if self.contains_point(sphere.origin) {
            return true;
        }

        let closest_surface_point = self.find_closest_surface_point(sphere.origin);
        let delta = closest_surface_point - sphere.origin;
        let distance = delta.magnitude();
        // dbg!(distance, sphere.radius);
        distance < sphere.radius
    }
}

pub fn build_scene_tree(scene: &Scene, renderer_state: &RendererState) -> Result<SceneTree> {
    let scene_bounds = ARENA_SIDE_LENGTH * 2.0;
    let base_aabb_max = Vector3::new(scene_bounds, scene_bounds, scene_bounds);
    let mut tree = SceneTree {
        aabb: Aabb {
            min: -base_aabb_max,
            max: base_aabb_max,
        },
        children: None,
    };

    let mut inserted_node_count = 0;

    for node in scene.nodes() {
        if let Some(node_bounding_sphere) =
            scene.get_node_bounding_sphere(node.id(), renderer_state)
        {
            tree.insert(node.id(), node_bounding_sphere)?;
            inserted_node_count += 1;
        }
    }

    logger_log(&format!("inserted_node_count={:?}", inserted_node_count));

    Ok(tree)
}

impl SceneTree {
    pub fn insert(&mut self, node_id: GameNodeId, node_bounding_sphere: Sphere) -> Result<()> {
        if !self.aabb.contains_sphere(node_bounding_sphere) {
            bail!("Tried to insert a node that's outside of the scene tree. Consider increasing size of the base aabb");
        }
        self.insert_internal(node_id, node_bounding_sphere, 0);
        Ok(())
    }

    pub fn get_non_empty_node_count(&self) -> u32 {
        match &self.children {
            None => 0,
            Some(SceneTreeChildren::Nodes(_)) => 1,
            Some(SceneTreeChildren::SubTrees(sub_trees)) => {
                let mut total = 0_u32;
                for sub_tree in sub_trees.iter() {
                    total += Self::get_non_empty_node_count(sub_tree);
                }
                total
            }
        }
    }

    fn subdivide(&mut self, new_depth: u8) {
        if let Some(Nodes(nodes)) = self.children.as_mut() {
            let mut new_subtrees: [SceneTree; 8] = Default::default();
            let min = self.aabb.min;
            let new_size = (self.aabb.max - min) / 2.0;
            // dbg!(new_size);
            let mut count = 0;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        let offset = Vector3::new(
                            min.x + i as f32 * new_size.x,
                            min.y + j as f32 * new_size.y,
                            min.z + k as f32 * new_size.z,
                        );
                        let subtree = &mut new_subtrees[count];
                        subtree.aabb = Aabb {
                            min: offset,
                            max: offset + new_size,
                        };
                        // dbg!(count, subtree.aabb);

                        for entry in nodes.iter().cloned() {
                            let (node_id, node_bounding_sphere) = entry;
                            if subtree.aabb.contains_sphere(node_bounding_sphere) {
                                subtree.insert_internal(node_id, node_bounding_sphere, new_depth);
                            }
                        }

                        count += 1;
                    }
                }
            }
            self.children = Some(SubTrees(Box::new(new_subtrees)));
        }
    }

    fn insert_internal(&mut self, node_id: GameNodeId, node_bounding_sphere: Sphere, depth: u8) {
        // dbg!(node_id, node_bounding_sphere, depth);

        let entry = (node_id, node_bounding_sphere);

        match self.children.as_mut() {
            None => {
                self.children = Some(Nodes(vec![entry]));
            }
            Some(Nodes(nodes)) => {
                if depth >= MAX_DEPTH {
                    nodes.push(entry);
                } else {
                    // subdivide and add the node
                    self.subdivide(depth + 1);
                    self.insert_internal(node_id, node_bounding_sphere, depth);
                }
            }
            Some(SubTrees(sub_trees)) => {
                for subtree in sub_trees.iter_mut() {
                    if subtree.aabb.contains_sphere(node_bounding_sphere) {
                        subtree.insert_internal(node_id, node_bounding_sphere, depth + 1);
                    }
                }
            }
        }
    }
}
