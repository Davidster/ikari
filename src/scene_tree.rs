use anyhow::{bail, Result};
use cgmath::Vector3;

use super::*;

#[derive(Debug, Clone)]
pub struct SceneTree {
    pub aabb: Aabb,
    pub nodes: Vec<(GameNodeId, Sphere)>,
    pub children: Option<Box<[SceneTree; 8]>>,
}

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
const MAX_DEPTH: u8 = 12;
// const MIN_CELL_SIZE: f32 = BASE_AABB_SIZE / 2_u32.pow(MAX_DEPTH) as f32;

impl Default for SceneTree {
    fn default() -> Self {
        Self {
            aabb: Aabb {
                min: Vector3::zero(),
                max: Vector3::zero(),
            },
            nodes: vec![],
            children: None,
        }
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self {
            min: Vector3::new(-1.0, -1.0, -1.0),
            max: Vector3::new(1.0, 1.0, 1.0),
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
    fn partially_contains_sphere(&self, sphere: Sphere) -> bool {
        if self.contains_point(sphere.origin) {
            return true;
        }

        let closest_surface_point = self.find_closest_surface_point(sphere.origin);
        let delta = closest_surface_point - sphere.origin;
        let distance = delta.magnitude();
        distance < sphere.radius
    }

    fn fully_contains_sphere(&self, sphere: Sphere) -> bool {
        let sphere_bb_half_size = Vector3::new(sphere.radius, sphere.radius, sphere.radius);
        let sphere_min = sphere.origin - sphere_bb_half_size;
        let sphere_max = sphere.origin + sphere_bb_half_size;

        self.contains_point(sphere_min) && self.contains_point(sphere_max)
    }

    pub fn subdivide(&self) -> [Self; 8] {
        let mut new_aabbs: [Self; 8] = Default::default();
        let new_size = (self.max - self.min) / 2.0;
        let mut count = 0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let offset = Vector3::new(
                        self.min.x + i as f32 * new_size.x,
                        self.min.y + j as f32 * new_size.y,
                        self.min.z + k as f32 * new_size.z,
                    );
                    new_aabbs[count] = Aabb {
                        min: offset,
                        max: offset + new_size,
                    };

                    count += 1;
                }
            }
        }
        new_aabbs
    }
}

pub fn build_scene_tree(scene: &Scene, renderer_state: &RendererState) -> Result<SceneTree> {
    let base_aabb_max = Vector3::new(BASE_AABB_SIZE, BASE_AABB_SIZE, BASE_AABB_SIZE);
    let mut tree = SceneTree {
        aabb: Aabb {
            min: -base_aabb_max,
            max: base_aabb_max,
        },
        ..Default::default()
    };

    let mut inserted_node_count = 0;

    for node in scene.nodes() {
        if node.mesh.is_none() || !node.mesh.as_ref().unwrap().cullable {
            continue;
        }
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
        if !self.aabb.fully_contains_sphere(node_bounding_sphere) {
            bail!("Tried to insert a node that's not fully contained by the scene tree. Consider increasing size of the base scene tree. Sphere: {:?}, Base aabb: {:?}", node_bounding_sphere, self.aabb);
        }
        self.insert_internal(node_id, node_bounding_sphere, 0);
        Ok(())
    }

    pub fn get_non_empty_node_count(&self) -> u32 {
        let mut total = 0_u32;
        if !self.nodes.is_empty() {
            total += 1;
        }
        if let Some(sub_trees) = self.children.as_ref() {
            for sub_tree in sub_trees.iter() {
                total += Self::get_non_empty_node_count(sub_tree);
            }
        }
        total
    }

    pub fn get_node_count(&self) -> u32 {
        let mut total = 0_u32;
        total += 1;
        if let Some(sub_trees) = self.children.as_ref() {
            for sub_tree in sub_trees.iter() {
                total += 1 + Self::get_non_empty_node_count(sub_tree);
            }
        }
        total
    }

    pub fn to_aabb_list(&self) -> Vec<Aabb> {
        let mut list: Vec<Aabb> = Vec::new();
        self.to_aabb_list_impl(&mut list);
        list
    }

    // only includes branch nodes and leaf nodes with game nodes inside them
    fn to_aabb_list_impl(&self, acc: &mut Vec<Aabb>) {
        if let Some(children) = self.children.as_ref() {
            acc.push(self.aabb);
            for child in children.iter() {
                child.to_aabb_list_impl(acc);
            }
        } else if !self.nodes.is_empty() {
            acc.push(self.aabb);
        }
    }

    // fn subdivide(&mut self, new_depth: u8) {
    //     if let Some(Nodes(nodes)) = self.children.as_mut() {
    //         let mut new_subtrees: [SceneTree; 8] = Default::default();
    //         let min = self.aabb.min;
    //         let new_size = (self.aabb.max - min) / 2.0;
    //         // dbg!(new_size);
    //         let mut count = 0;
    //         for i in 0..2 {
    //             for j in 0..2 {
    //                 for k in 0..2 {
    //                     let offset = Vector3::new(
    //                         min.x + i as f32 * new_size.x,
    //                         min.y + j as f32 * new_size.y,
    //                         min.z + k as f32 * new_size.z,
    //                     );
    //                     let subtree = &mut new_subtrees[count];
    //                     subtree.aabb = Aabb {
    //                         min: offset,
    //                         max: offset + new_size,
    //                     };
    //                     // dbg!(count, subtree.aabb);

    //                     for entry in nodes.iter().cloned() {
    //                         let (node_id, node_bounding_sphere) = entry;
    //                         if subtree.aabb.contains_sphere(node_bounding_sphere) {
    //                             subtree.insert_internal(node_id, node_bounding_sphere, new_depth);
    //                         }
    //                     }

    //                     count += 1;
    //                 }
    //             }
    //         }
    //         self.children = Some(SubTrees(Box::new(new_subtrees)));
    //     }
    // }

    fn insert_internal(&mut self, node_id: GameNodeId, node_bounding_sphere: Sphere, depth: u8) {
        // logger_log("1");
        // dbg!(node_id, node_bounding_sphere, depth);

        let entry = (node_id, node_bounding_sphere);

        if depth >= MAX_DEPTH {
            self.nodes.push(entry);
            return;
        }

        // logger_log("2");

        // let child_aabbs = self
        //     .children
        //     .as_ref()
        //     .map(|children| {
        //         let mut aabbs: [Aabb; 8] = Default::default();
        //         for (i, child) in children.iter().enumerate() {
        //             aabbs[i] = child.aabb;
        //         }
        //         aabbs
        //     })
        //     .unwrap_or_else(|| self.aabb.subdivide());

        // let fully_contained_index = child_aabbs
        //     .iter()
        //     .enumerate()
        //     .find_map(|(i, aabb)| aabb.fully_contains_sphere(node_bounding_sphere).then(|| i));

        // match (fully_contained_index, self.children.as_mut()) {
        //     (Some(fully_contained_index)) => {
        //         let mut new_children: [SceneTree; 8] = Default::default();
        //         for (i, aabb) in child_aabbs.iter().copied().enumerate() {
        //             new_children[i].aabb = aabb;
        //         }
        //         let child_iterator = match self.children.as_ref() {
        //             Some(children) => children.iter(),
        //             None => new_children.iter(),
        //         };

        //         new_children[fully_contained_index].insert_internal(
        //             node_id,
        //             node_bounding_sphere,
        //             depth + 1,
        //         )
        //     }
        //     (None, _) => {
        //         self.nodes.push(entry);
        //     }
        // }

        match self.children.as_mut() {
            None => {
                let subdivided = self.aabb.subdivide();
                let fully_contained_index = subdivided.iter().enumerate().find_map(|(i, aabb)| {
                    aabb.fully_contains_sphere(node_bounding_sphere)
                        .then_some(i)
                });
                match fully_contained_index {
                    Some(fully_contained_index) => {
                        // logger_log("3");
                        let mut new_children: [SceneTree; 8] = Default::default();
                        for (i, aabb) in subdivided.iter().copied().enumerate() {
                            new_children[i].aabb = aabb;
                        }
                        new_children[fully_contained_index].insert_internal(
                            node_id,
                            node_bounding_sphere,
                            depth + 1,
                        );
                        self.children = Some(Box::new(new_children));
                    }
                    None => {
                        // logger_log("4");
                        self.nodes.push(entry);
                    }
                }
            }
            Some(children) => {
                let fully_contained_index = children.iter().enumerate().find_map(|(i, tree)| {
                    tree.aabb
                        .fully_contains_sphere(node_bounding_sphere)
                        .then_some(i)
                });
                match fully_contained_index {
                    Some(fully_contained_index) => {
                        // logger_log("5");
                        children[fully_contained_index].insert_internal(
                            node_id,
                            node_bounding_sphere,
                            depth + 1,
                        );
                    }
                    None => {
                        // logger_log("6");
                        self.nodes.push(entry);
                    }
                }
            }
        }
    }
}
