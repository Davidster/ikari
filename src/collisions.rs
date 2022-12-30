use cgmath::{Rad, Vector3};

use super::*;

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

impl Default for Aabb {
    fn default() -> Self {
        Self {
            min: Vector3::new(-1.0, -1.0, -1.0),
            max: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

impl Aabb {
    pub fn _volume(&self) -> f32 {
        let size = self.size();
        size.x * size.y * size.z
    }

    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }

    pub fn origin(&self) -> Vector3<f32> {
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
    pub fn _find_closest_surface_point(&self, p: Vector3<f32>) -> Vector3<f32> {
        let mut q: Vector3<f32> = Vector3::zero();
        for i in 0..3 {
            let mut v = p[i];
            v = v.clamp(self.min[i], self.max[i]);
            q[i] = v;
        }
        q
    }

    pub fn contains_point(&self, point: Vector3<f32>) -> bool {
        self.min.x < point.x
            && self.max.x > point.x
            && self.min.y < point.y
            && self.max.y > point.y
            && self.min.z < point.z
            && self.max.z > point.z
    }

    // true if fully contains or partially contains
    pub fn _partially_contains_sphere(&self, sphere: Sphere) -> bool {
        if self.contains_point(sphere.origin) {
            return true;
        }

        let closest_surface_point = self._find_closest_surface_point(sphere.origin);
        let delta = closest_surface_point - sphere.origin;
        let distance = delta.magnitude();
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
                    let offset = Vector3::new(
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
}

impl Plane {
    pub fn from_normal_and_point(normal: Vector3<f32>, point: Vector3<f32>) -> Self {
        Self {
            normal: normal.normalize(),
            d: -normal.normalize().dot(point),
        }
    }
}

impl Sphere {
    pub fn aabb(&self) -> Aabb {
        let sphere_bb_half_size = Vector3::new(self.radius, self.radius, self.radius);
        Aabb {
            min: self.origin - sphere_bb_half_size,
            max: self.origin + sphere_bb_half_size,
        }
    }
}

impl Frustum {
    pub fn from_camera_params(
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
    pub fn _contains_point(&self, point: Vector3<f32>) -> bool {
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