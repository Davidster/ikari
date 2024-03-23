use std::cmp::Ordering;

use crate::mesh::BasicMesh;
use crate::mesh::ShaderVertex;
use crate::physics::rapier3d_f64::prelude::*;
use glam::f32::Vec3;
use rand::Rng;
use rand::SeedableRng;

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
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
pub struct CameraFrustumDescriptor {
    pub focal_point: Vec3,
    pub forward_vector: Vec3,
    pub aspect_ratio: f32,
    pub near_plane_distance: f32,
    pub far_plane_distance: f32,
    pub fov_y_rad: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            center: Default::default(),
            radius: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionResult {
    FullyContained,
    PartiallyIntersecting,
    NotIntersecting,
}

impl Default for Aabb {
    fn default() -> Self {
        Self {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

impl Aabb {
    // None if the iterator had less than two points
    pub fn make_from_points(mut points: impl Iterator<Item = Vec3>) -> Option<Self> {
        let first_point = points.next();

        first_point?;

        let first_point = first_point.unwrap();
        let mut min = first_point;
        let mut max = first_point;

        let mut got_second_point = false;

        for point in points {
            got_second_point = true;

            min = min.min(point);
            max = max.max(point);
        }

        got_second_point.then_some(Self { min, max })
    }

    pub fn volume(&self) -> f32 {
        let size = self.size();
        size.x * size.y * size.z
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn center(&self) -> Vec3 {
        self.max - self.size() / 2.0
    }

    pub fn vertices(&self) -> [Vec3; 8] {
        let size = self.size();
        let mut vertices: [Vec3; 8] = Default::default();
        let mut counter = 0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    vertices[counter] = Vec3::new(
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

    /// Taken from https://gamedev.stackexchange.com/questions/156870/how-do-i-implement-a-aabb-sphere-collision
    /// ClosestPtPointAABB
    pub fn find_closest_surface_point(&self, p: Vec3) -> Vec3 {
        let mut q: Vec3 = Vec3::new(0.0, 0.0, 0.0);
        for i in 0..3 {
            let mut v = p[i];
            v = v.clamp(self.min[i], self.max[i]);
            q[i] = v;
        }
        q
    }

    pub fn contains_point(&self, point: Vec3) -> bool {
        self.min.x < point.x
            && self.max.x > point.x
            && self.min.y < point.y
            && self.max.y > point.y
            && self.min.z < point.z
            && self.max.z > point.z
    }

    /// Returns true if the Aabb fully contains or partially contains the sphere
    pub fn partially_contains_sphere(&self, sphere: Sphere) -> bool {
        if self.contains_point(sphere.center) {
            return true;
        }

        let closest_surface_point = self.find_closest_surface_point(sphere.center);
        let delta = closest_surface_point - sphere.center;
        let distance = delta.length();
        distance < sphere.radius
    }

    pub fn fully_contains_sphere(&self, sphere: Sphere) -> bool {
        let sphere_aabb = sphere.aabb();
        self.contains_point(sphere_aabb.min) && self.contains_point(sphere_aabb.max)
    }

    pub fn subdivide(&self) -> [Self; 8] {
        let mut new_aabbs: [Self; 8] = Default::default();
        for (i, aabb) in self.subdivide_iter().enumerate() {
            new_aabbs[i] = aabb;
        }
        new_aabbs
    }

    pub fn subdivide_iter(&self) -> impl Iterator<Item = Self> + '_ {
        let new_size = (self.max - self.min) / 2.0;
        (0..2).flat_map(move |i| {
            (0..2).flat_map(move |j| {
                (0..2).map(move |k| {
                    let offset = Vec3::new(
                        self.min.x + i as f32 * new_size.x,
                        self.min.y + j as f32 * new_size.y,
                        self.min.z + k as f32 * new_size.z,
                    );
                    Aabb {
                        min: offset,
                        max: offset + new_size,
                    }
                })
            })
        })
    }

    pub fn scale_translate(&self, mut scale: Vec3, translation: Vec3) -> Aabb {
        scale = scale.abs();
        Aabb {
            min: self.min * scale + translation,
            max: self.max * scale + translation,
        }
    }
}

impl Plane {
    pub fn from_normal_and_point(normal: Vec3, point: Vec3) -> Self {
        Self {
            normal: normal.normalize(),
            d: -normal.normalize().dot(point),
        }
    }
}

impl Sphere {
    pub fn aabb(&self) -> Aabb {
        let sphere_bb_half_size = Vec3::new(self.radius, self.radius, self.radius);
        Aabb {
            min: self.center - sphere_bb_half_size,
            max: self.center + sphere_bb_half_size,
        }
    }
}

impl Frustum {
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

    /// See https://gdbooks.gitbooks.io/legacyopengl/content/Chapter8/halfspace.html
    /// and https://gdbooks.gitbooks.io/legacyopengl/content/Chapter8/frustum.html
    pub fn contains_point(&self, point: Vec3) -> bool {
        for plane in self.planes() {
            if plane.normal.dot(point) + plane.d < 0.0 {
                return false;
            }
        }
        true
    }

    /// See https://www.flipcode.com/archives/Frustum_Culling.shtml Frustrum::ContainsAaBox
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

    /// See https://www.flipcode.com/archives/Frustum_Culling.shtml Frustrum::ContainsSphere
    pub fn sphere_intersection_test(&self, sphere: Sphere) -> IntersectionResult {
        let mut f_distance;

        let mut partial = false;
        let planes = self.planes();

        for plane in planes {
            f_distance = plane.normal.dot(sphere.center) + plane.d;

            if f_distance < -sphere.radius {
                return IntersectionResult::NotIntersecting;
            }

            if !partial && f_distance.abs() < sphere.radius {
                partial = true;
            }
        }

        if partial {
            IntersectionResult::PartiallyIntersecting
        } else {
            IntersectionResult::FullyContained
        }
    }

    #[allow(dead_code)]
    pub fn point_cloud_in_frustum(&self, focal_point: Vec3, forward_vector: Vec3) -> Vec<Vec3> {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(
            ((focal_point.x.abs()
                + focal_point.y.abs()
                + focal_point.z.abs()
                + forward_vector.x.abs()
                + forward_vector.y.abs()
                + forward_vector.z.abs())
                * 100.0) as u64,
        );

        let mut result = Vec::new();

        for _ in 0..5000 {
            let point_cloud_radius = 50.0;
            let random_point_near_camera = focal_point
                + point_cloud_radius * forward_vector
                + Vec3::new(
                    point_cloud_radius * (1.0 - rng.gen::<f32>() * 2.0),
                    point_cloud_radius * (1.0 - rng.gen::<f32>() * 2.0),
                    point_cloud_radius * (1.0 - rng.gen::<f32>() * 2.0),
                );
            if self.contains_point(random_point_near_camera) {
                result.push(random_point_near_camera);
            }
        }

        result
    }
}

impl From<CameraFrustumDescriptor> for Frustum {
    fn from(desc: CameraFrustumDescriptor) -> Self {
        let right = desc
            .forward_vector
            .cross(Vec3::new(0.0, 1.0, 0.0))
            .normalize();
        let forward = desc.forward_vector;

        // see https://learnopengl.com/Guest-Articles/2021/Scene/Frustum-Culling
        let up = right.cross(forward).normalize();
        let half_v_side = desc.far_plane_distance * (desc.fov_y_rad * 0.5).tan();
        let half_h_side = half_v_side * desc.aspect_ratio;
        let front_mult_far = desc.far_plane_distance * forward;

        Self {
            left: Plane::from_normal_and_point(
                (front_mult_far - right * half_h_side).cross(up),
                desc.focal_point,
            ),
            right: Plane::from_normal_and_point(
                up.cross(front_mult_far + right * half_h_side),
                desc.focal_point,
            ),
            bottom: Plane::from_normal_and_point(
                (front_mult_far + up * half_v_side).cross(right),
                desc.focal_point,
            ),
            top: Plane::from_normal_and_point(
                right.cross(front_mult_far - up * half_v_side),
                desc.focal_point,
            ),
            near: Plane::from_normal_and_point(
                forward,
                desc.focal_point + desc.near_plane_distance * forward,
            ),
            far: Plane::from_normal_and_point(-forward, desc.focal_point + front_mult_far),
        }
    }
}

impl CameraFrustumDescriptor {
    /// this is SLOW. takes about 2 microseconds. consider caching the result
    pub fn to_convex_polyhedron(&self) -> ConvexPolyhedron {
        let points: Vec<_> = self
            .to_basic_mesh()
            .vertices
            .iter()
            .map(|vertex| {
                Point::from([
                    vertex.position[0] as f64,
                    vertex.position[1] as f64,
                    vertex.position[2] as f64,
                ])
            })
            .collect();
        ColliderBuilder::convex_hull(&points)
            .expect("Failed to construct convex hull for frustum")
            .build()
            .shape()
            .as_convex_polyhedron()
            .unwrap()
            .clone()
    }

    /// here we're less worried about the bounding sphere being optimally tight
    /// and more worried about it having the same radius regardless of how
    /// the frustum is rotated
    pub fn make_rotation_independent_bounding_sphere(&self) -> Sphere {
        let right = self
            .forward_vector
            .cross(Vec3::new(0.0, 1.0, 0.0))
            .normalize();
        let up = right.cross(self.forward_vector);
        let tan_half_fovy = (self.fov_y_rad / 2.0).tan();

        let d_x = tan_half_fovy * right * self.aspect_ratio;
        let d_y = tan_half_fovy * up;

        let d_x_far = self.far_plane_distance * d_x;
        let d_y_far = self.far_plane_distance * d_y;

        let center = Aabb::make_from_points(
            self.to_basic_mesh()
                .vertices
                .iter()
                .map(|vertex| vertex.position.into()),
        )
        .unwrap()
        .center();

        let far_plane_side_length_x = 2.0 * d_x_far.length();
        let far_plane_side_length_y = 2.0 * d_y_far.length();
        let forward_length = self.far_plane_distance - self.near_plane_distance;
        let diagonal_length = ((self.far_plane_distance - self.near_plane_distance)
            / self.far_plane_distance)
            * self.far_plane_distance
            / (self.fov_y_rad / 2.0).cos();
        let longest_side_length = [
            far_plane_side_length_x,
            far_plane_side_length_y,
            forward_length,
            diagonal_length,
        ]
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap();

        let radius = longest_side_length / 2.0;

        Sphere { center, radius }
    }

    pub fn to_basic_mesh(&self) -> BasicMesh {
        let right = self
            .forward_vector
            .cross(Vec3::new(0.0, 1.0, 0.0))
            .normalize();
        let up = right.cross(self.forward_vector);
        let tan_half_fovy = (self.fov_y_rad / 2.0).tan();

        let d_x = tan_half_fovy * right * self.aspect_ratio;
        let d_y = tan_half_fovy * up;

        let d_x_near = self.near_plane_distance * d_x;
        let d_y_near = self.near_plane_distance * d_y;
        let near_vertices = {
            let near_plane_center =
                self.focal_point + self.near_plane_distance * self.forward_vector;
            let top_left = near_plane_center + d_y_near - d_x_near;
            let top_right = near_plane_center + d_y_near + d_x_near;
            let bottom_right = near_plane_center - d_y_near + d_x_near;
            let bottom_left = near_plane_center - d_y_near - d_x_near;

            (top_left, top_right, bottom_right, bottom_left)
        };

        let d_x_far = self.far_plane_distance * d_x;
        let d_y_far = self.far_plane_distance * d_y;
        let far_vertices = {
            let far_plane_center = self.focal_point + self.far_plane_distance * self.forward_vector;
            let top_left = far_plane_center + d_y_far - d_x_far;
            let top_right = far_plane_center + d_y_far + d_x_far;
            let bottom_right = far_plane_center - d_y_far + d_x_far;
            let bottom_left = far_plane_center - d_y_far - d_x_far;

            (top_left, top_right, bottom_right, bottom_left)
        };

        // vertices and indices adapted from cube mesh:
        // https://gist.github.com/MaikKlein/0b6d6bb58772c13593d0a0add6004c1c
        let vertices = [
            near_vertices.2,
            far_vertices.2,
            far_vertices.3,
            near_vertices.3,
            near_vertices.1,
            far_vertices.1,
            far_vertices.0,
            near_vertices.0,
        ];
        let mut indices = [
            1, 2, 3, 7, 6, 5, 4, 5, 1, 5, 6, 2, 2, 6, 7, 0, 3, 7, 0, 1, 3, 4, 7, 5, 0, 4, 1, 1, 5,
            2, 3, 2, 7, 4, 0, 7, 3, 2, 1,
        ];

        indices.reverse();
        BasicMesh {
            vertices: vertices
                .iter()
                .map(|pos| ShaderVertex {
                    position: pos.to_array(),
                    ..Default::default()
                })
                .collect(),
            indices: indices.to_vec(),
        }
    }
}
