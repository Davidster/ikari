use glam::{
    f32::{Mat4, Quat, Vec3},
    EulerRot,
};

use crate::{
    collisions::CameraFrustumDescriptor,
    player_controller::ControlledViewDirection,
    transform::{
        clear_translation_from_matrix, get_translation_from_matrix, make_orthographic_proj_matrix,
        make_perspective_proj_matrix, TransformBuilder,
    },
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Camera {
    pub horizontal_rotation: f32,
    pub vertical_rotation: f32,
    pub position: Vec3,
}

impl Camera {
    pub fn to_transform(self) -> crate::transform::Transform {
        TransformBuilder::new()
            .position(self.position)
            .rotation(
                Quat::from_euler(EulerRot::XYZ, 0.0, self.horizontal_rotation, 0.0)
                    * Quat::from_euler(EulerRot::XYZ, self.vertical_rotation, 0.0, 0.0),
            )
            .build()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ShaderCameraData {
    pub proj: Mat4,
    pub view: Mat4,
    pub rotation_only_view: Mat4,
    pub position: Vec3,
    pub near_plane_distance: f32,
    pub far_plane_distance: f32,
}

impl ShaderCameraData {
    pub fn perspective(
        transform: Mat4,
        aspect_ratio: f32,
        near_plane_distance: f32,
        far_plane_distance: f32,
        fov_y: f32,
        reverse_z: bool,
    ) -> Self {
        let proj = make_perspective_proj_matrix(
            near_plane_distance,
            far_plane_distance,
            fov_y,
            aspect_ratio,
            reverse_z,
        );
        let rotation_only_matrix = clear_translation_from_matrix(transform);
        let rotation_only_view = rotation_only_matrix.inverse();
        let view = transform.inverse();
        let position = get_translation_from_matrix(transform);
        Self {
            proj,
            view,
            rotation_only_view,
            position,
            near_plane_distance,
            far_plane_distance,
        }
    }

    pub fn orthographic(
        transform: Mat4,
        width: f32,
        height: f32,
        near_plane_distance: f32,
        far_plane_distance: f32,
        reverse_z: bool,
    ) -> Self {
        let proj = make_orthographic_proj_matrix(
            width,
            height,
            near_plane_distance,
            far_plane_distance,
            reverse_z,
        );
        let rotation_only_matrix = clear_translation_from_matrix(transform);
        let rotation_only_view = rotation_only_matrix.inverse();
        let view = transform.inverse();
        let position = get_translation_from_matrix(transform);
        Self {
            proj,
            view,
            rotation_only_view,
            position,
            near_plane_distance,
            far_plane_distance,
        }
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshShaderCameraRaw {
    view_proj: [[f32; 4]; 4],
    position: [f32; 3],
    far_plane_distance: f32,
}

impl From<ShaderCameraData> for MeshShaderCameraRaw {
    fn from(
        ShaderCameraData {
            proj,
            view,
            position,
            far_plane_distance,
            ..
        }: ShaderCameraData,
    ) -> Self {
        Self {
            view_proj: (proj * view).to_cols_array_2d(),
            position: [position.x, position.y, position.z],
            far_plane_distance,
        }
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyboxShaderCameraRaw {
    rotation_only_view_proj: [[f32; 4]; 4],
    position: [f32; 3],
    far_plane_distance: f32,
}

impl From<ShaderCameraData> for SkyboxShaderCameraRaw {
    fn from(
        ShaderCameraData {
            proj,
            position,
            far_plane_distance,
            rotation_only_view,
            ..
        }: ShaderCameraData,
    ) -> Self {
        Self {
            rotation_only_view_proj: (proj * rotation_only_view).to_cols_array_2d(),
            position: [position.x, position.y, position.z],
            far_plane_distance,
        }
    }
}

pub fn build_cubemap_face_camera_view_directions() -> impl Iterator<Item = ControlledViewDirection>
{
    vec![
        (90.0, 0.0),    // right (negative x)
        (-90.0, 0.0),   // left (positive x)
        (180.0, 90.0),  // top (positive y)
        (180.0, -90.0), // bottom (negative y)
        (180.0, 0.0),   // front (position z)
        (0.0, 0.0),     // back (negative z)
    ]
    .into_iter()
    .map(
        move |(horizontal_rotation, vertical_rotation): (f32, f32)| ControlledViewDirection {
            horizontal: horizontal_rotation.to_radians(),
            vertical: vertical_rotation.to_radians(),
        },
    )
}

pub fn build_cubemap_face_camera_views(
    position: Vec3,
    near_plane_distance: f32,
    far_plane_distance: f32,
    reverse_z: bool,
) -> Vec<ShaderCameraData> {
    build_cubemap_face_camera_view_directions()
        .map(|view_direction| {
            ShaderCameraData::perspective(
                Camera {
                    horizontal_rotation: view_direction.horizontal,
                    vertical_rotation: view_direction.vertical,
                    position,
                }
                .to_transform()
                .into(),
                1.0,
                near_plane_distance,
                far_plane_distance,
                90.0_f32.to_radians(),
                reverse_z,
            )
        })
        .collect()
}

pub fn build_cubemap_face_frusta(
    position: Vec3,
    near_plane_distance: f32,
    far_plane_distance: f32,
) -> Vec<CameraFrustumDescriptor> {
    build_cubemap_face_camera_view_directions()
        .map(|view_direction| {
            let forward = view_direction.to_vector();
            CameraFrustumDescriptor {
                focal_point: position,
                forward_vector: forward,
                aspect_ratio: 1.0,
                near_plane_distance,
                far_plane_distance,
                fov_y_rad: 90.0_f32.to_radians(),
            }
        })
        .collect()
}
