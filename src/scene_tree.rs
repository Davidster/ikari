use anyhow::{bail, Result};
use cgmath::{Matrix4, Rad, Vector3, Vector4};

use super::*;

#[derive(Debug, Clone)]
pub struct SceneTree {
    base_aabb: Aabb,
    pub nodes: Vec<(GameNodeId, Sphere)>,
    pub children: Option<Box<[SceneTree; 8]>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vector3<f32>,
    pub d: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub left: Plane,
    pub right: Plane,
    pub top: Plane,
    pub bottom: Plane,
    pub near: Plane,
    pub far: Plane,
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub origin: Vector3<f32>,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionResult {
    FullyContained,
    PartiallyIntersecting,
    NotIntersecting,
}

const ROOT_AABB_SIZE: f32 = 4000.0;
const MAX_DEPTH: u8 = 20;
const K: f32 = 2.0; // looseness factor, set to 1.0 to disable loosening

impl Default for SceneTree {
    fn default() -> Self {
        Self {
            base_aabb: Aabb {
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
    fn _volume(&self) -> f32 {
        let size = self.size();
        size.x * size.y * size.z
    }

    fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }

    fn origin(&self) -> Vector3<f32> {
        self.max - self.size() / 2.0
    }

    pub fn vertices(&self) -> [Vector3<f32>; 8] {
        let size = self.size();
        let mut vertices: [Vector3<f32>; 8] = [Vector3::zero(); 8];
        let mut counter = 0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    vertices[counter] = Vector3::new(
                        self.min.x + size.x * i as f32,
                        self.min.y + size.y * j as f32,
                        self.min.z + size.z * k as f32,
                    );
                    counter += 1;
                }
            }
        }
        vertices
    }

    // taken from https://gamedev.stackexchange.com/questions/156870/how-do-i-implement-a-aabb-sphere-collision
    // ClosestPtPointAABB
    fn _find_closest_surface_point(&self, p: Vector3<f32>) -> Vector3<f32> {
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
    fn _partially_contains_sphere(&self, sphere: Sphere) -> bool {
        if self.contains_point(sphere.origin) {
            return true;
        }

        let closest_surface_point = self._find_closest_surface_point(sphere.origin);
        let delta = closest_surface_point - sphere.origin;
        let distance = delta.magnitude();
        distance < sphere.radius
    }

    // TODO: convert to an aabb test since that's what it is anyways.
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

impl Plane {
    pub fn from_normal_and_point(normal: Vector3<f32>, point: Vector3<f32>) -> Self {
        Self {
            normal: normal.normalize(),
            d: -normal.normalize().dot(point),
        }
    }
}

impl Frustum {
    pub fn from_camera_params(
        transform: Matrix4<f32>,
        aspect_ratio: f32,
        near_plane_distance: f32,
        far_plane_distance: f32,
        fov_y: Rad<f32>,
    ) -> Self {
        let position = get_translation_from_matrix(transform);
        // see https://learnopengl.com/Guest-Articles/2021/Scene/Frustum-Culling
        let up = transform.y.truncate();
        let forward = transform.z.truncate();
        let right = forward.cross(up);
        let half_v_side = far_plane_distance * (fov_y.0 * 0.5).tan();
        let half_h_side = half_v_side * aspect_ratio;
        let front_mult_far = far_plane_distance * forward;

        Self {
            left: Plane::from_normal_and_point(
                (front_mult_far - right * half_h_side).cross(up),
                position,
            ),
            right: Plane::from_normal_and_point(
                up.cross(front_mult_far + right * half_h_side),
                position,
            ),
            bottom: Plane::from_normal_and_point(
                (front_mult_far + up * half_v_side).cross(right),
                position,
            ),
            top: Plane::from_normal_and_point(
                right.cross(front_mult_far - up * half_v_side),
                position,
            ),
            near: Plane::from_normal_and_point(forward, position + near_plane_distance * forward),
            far: Plane::from_normal_and_point(-forward, position + front_mult_far),
        }
    }

    pub fn from_camera_params_2(
        position: Vector3<f32>,
        forward: Vector3<f32>,
        right: Vector3<f32>,
        aspect_ratio: f32,
        near_plane_distance: f32,
        far_plane_distance: f32,
        fov_y: Rad<f32>,
    ) -> Self {
        // see https://learnopengl.com/Guest-Articles/2021/Scene/Frustum-Culling
        let up = right.cross(forward).normalize();
        let half_v_side = far_plane_distance * (fov_y.0 * 0.5).tan();
        let half_h_side = half_v_side * aspect_ratio;
        let front_mult_far = far_plane_distance * forward;

        Self {
            left: Plane::from_normal_and_point(
                (front_mult_far - right * half_h_side).cross(up),
                position,
            ),
            right: Plane::from_normal_and_point(
                up.cross(front_mult_far + right * half_h_side),
                position,
            ),
            bottom: Plane::from_normal_and_point(
                (front_mult_far + up * half_v_side).cross(right),
                position,
            ),
            top: Plane::from_normal_and_point(
                right.cross(front_mult_far - up * half_v_side),
                position,
            ),
            near: Plane::from_normal_and_point(forward, position + near_plane_distance * forward),
            far: Plane::from_normal_and_point(-forward, position + front_mult_far),
        }
    }

    pub fn planes(&self) -> [Plane; 6] {
        [
            self.left,
            self.right,
            self.top,
            self.bottom,
            self.near,
            self.far,
        ]
    }

    // see https://gdbooks.gitbooks.io/legacyopengl/content/Chapter8/halfspace.html
    // and https://gdbooks.gitbooks.io/legacyopengl/content/Chapter8/frustum.html
    pub fn contains_point(&self, point: Vector3<f32>) -> bool {
        for plane in self.planes() {
            if plane.normal.dot(point) + plane.d < 0.0 {
                return false;
            }
        }
        true
    }

    // see https://www.flipcode.com/archives/Frustum_Culling.shtml Frustrum::ContainsAaBox
    // check if the aabb is fully contained in the frustum
    pub fn aabb_intersection_test(&self, aabb: Aabb) -> IntersectionResult {
        let mut total_in = 0;

        let vertices = aabb.vertices();

        for plane in self.planes() {
            let mut in_count = 0;

            for vertex in vertices {
                if plane.normal.dot(vertex) + plane.d > 0.0 {
                    in_count += 1;
                }
            }

            if in_count == 0 {
                return IntersectionResult::NotIntersecting;
            }

            if in_count == 8 {
                total_in += 1;
            }
        }

        if total_in == 6 {
            IntersectionResult::FullyContained
        } else {
            IntersectionResult::PartiallyIntersecting
        }
    }
}

#[profiling::function]
pub fn build_scene_tree(scene: &Scene, renderer_state: &RendererState) -> SceneTree {
    let _offset = ROOT_AABB_SIZE * 0.02 * std::f32::consts::PI * Vector3::new(1.0, 1.0, 1.0);
    let root_aabb_max = ROOT_AABB_SIZE * Vector3::new(1.0, 1.0, 1.0);
    let mut tree = SceneTree {
        base_aabb: Aabb {
            min: -root_aabb_max, /*  + offset */
            max: root_aabb_max,  /*  + offset */
        },
        ..Default::default()
    };

    for node in scene.nodes() {
        if node.mesh.is_none() || !node.mesh.as_ref().unwrap().cullable {
            continue;
        }
        if let Some(node_bounding_sphere) =
            scene.get_node_bounding_sphere(node.id(), renderer_state)
        {
            tree.insert(node.id(), node_bounding_sphere);
        }
    }

    tree
}

impl SceneTree {
    pub fn insert(&mut self, node_id: GameNodeId, node_bounding_sphere: Sphere) {
        if !self.aabb().fully_contains_sphere(node_bounding_sphere) {
            logger_log(&format!("WARNING Tried to insert a node that's not fully contained by the scene tree. Consider increasing size of the base scene tree. Sphere: {:?}, Base aabb: {:?}", node_bounding_sphere, self.base_aabb));
            // bail!("Tried to insert a node that's not fully contained by the scene tree. Consider increasing size of the base scene tree. Sphere: {:?}, Base aabb: {:?}", node_bounding_sphere, self.aabb);
        }
        self.insert_internal(node_id, node_bounding_sphere, 0);
    }

    pub fn _get_non_empty_node_count(&self) -> u32 {
        let mut total = 0_u32;
        if !self.nodes.is_empty() {
            total += 1;
        }
        if let Some(sub_trees) = self.children.as_ref() {
            for sub_tree in sub_trees.iter() {
                total += Self::_get_non_empty_node_count(sub_tree);
            }
        }
        total
    }

    pub fn _get_node_count(&self) -> u32 {
        let mut total = 0_u32;
        total += 1;
        if let Some(sub_trees) = self.children.as_ref() {
            for sub_tree in sub_trees.iter() {
                total += Self::_get_node_count(sub_tree);
            }
        }
        total
    }

    pub fn _get_game_node_count(&self) -> u32 {
        let mut total = 0_u32;
        total += self.nodes.len() as u32;
        if let Some(sub_trees) = self.children.as_ref() {
            for sub_tree in sub_trees.iter() {
                total += Self::_get_game_node_count(sub_tree);
            }
        }
        total
    }

    #[profiling::function]
    pub fn to_aabb_list(&self) -> Vec<Aabb> {
        let mut list: Vec<Aabb> = Vec::new();
        self.to_aabb_list_impl(&mut list);
        list
    }

    // includes all branch nodes and nodes that contain game nodes
    fn to_aabb_list_impl(&self, acc: &mut Vec<Aabb>) {
        if !self.nodes.is_empty() || self.children.as_ref().is_some() {
            acc.push(self.aabb());
        }
        if let Some(children) = self.children.as_ref() {
            for child in children.iter() {
                child.to_aabb_list_impl(acc);
            }
        }
    }

    #[profiling::function]
    pub fn _get_all_nodes(&self) -> Vec<GameNodeId> {
        let mut result: Vec<GameNodeId> = Vec::new();
        self.get_all_nodes_impl(&mut result);
        result
    }

    fn get_all_nodes_impl(&self, acc: &mut Vec<GameNodeId>) {
        acc.extend(self.nodes.iter().map(|(node_id, _)| node_id));
        if let Some(children) = self.children.as_ref() {
            for child in children.iter() {
                child.get_all_nodes_impl(acc);
            }
        }
    }

    #[profiling::function]
    pub fn get_intersecting_nodes(&self, frustum: Frustum) -> Vec<GameNodeId> {
        let mut result: Vec<GameNodeId> = Vec::new();
        self.get_intersecting_nodes_impl(frustum, &mut result);
        result
    }

    // it's a loose octree, so the base_aabb shouldn't be used for intersection tests, only subdivision
    // see Real-Time Collision Detection, section 7.3.6
    fn aabb(&self) -> Aabb {
        self.loosen_aabb(&self.base_aabb)
    }

    fn loosen_aabb(&self, aabb: &Aabb) -> Aabb {
        let extension_factor = aabb.size() * ((K - 1.0) / 2.0);
        Aabb {
            min: aabb.min - extension_factor,
            max: aabb.max + extension_factor,
        }
    }

    fn get_intersecting_nodes_impl(&self, frustum: Frustum, acc: &mut Vec<GameNodeId>) {
        use IntersectionResult::*;

        match frustum.aabb_intersection_test(self.aabb()) {
            NotIntersecting => {}
            FullyContained => {
                self.get_all_nodes_impl(acc);
            }
            PartiallyIntersecting => {
                acc.extend(self.nodes.iter().map(|(node_id, _)| node_id));
                if let Some(children) = self.children.as_ref() {
                    for child in children.iter() {
                        child.get_intersecting_nodes_impl(frustum, acc);
                    }
                }
            }
        }
    }

    fn insert_internal(&mut self, node_id: GameNodeId, node_bounding_sphere: Sphere, depth: u8) {
        let entry = (node_id, node_bounding_sphere);

        if depth >= MAX_DEPTH {
            self.nodes.push(entry);
            return;
        }

        let mut fully_contained_index: Option<(usize, f32)> = None;
        let children_base_aabbs = self.base_aabb.subdivide();
        let mut children_aabbs: [Aabb; 8] = Default::default();
        for (i, child_base_aabb) in children_base_aabbs.iter().enumerate() {
            children_aabbs[i] = self.loosen_aabb(child_base_aabb);
        }
        for (i, aabb) in children_aabbs.iter().enumerate() {
            if aabb.fully_contains_sphere(node_bounding_sphere) {
                let distance2 = (aabb.origin() - node_bounding_sphere.origin).magnitude2();
                // pick the aabb whose origin is closest to the object
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

        match (self.children.as_mut(), fully_contained_index) {
            (None, Some((fully_contained_index, _))) => {
                let mut new_children: [SceneTree; 8] = Default::default();
                for (i, aabb) in children_base_aabbs.iter().copied().enumerate() {
                    new_children[i].base_aabb = aabb;
                }
                new_children[fully_contained_index].insert_internal(
                    node_id,
                    node_bounding_sphere,
                    depth + 1,
                );
                self.children = Some(Box::new(new_children));
            }
            (Some(children), Some((fully_contained_index, _))) => {
                children[fully_contained_index].insert_internal(
                    node_id,
                    node_bounding_sphere,
                    depth + 1,
                );
            }
            _ => {
                self.nodes.push(entry);
            }
        }
    }
}
