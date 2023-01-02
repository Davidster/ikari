use super::*;
use glam::{
    f32::{Mat4, Quat, Vec3},
    EulerRot,
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
pub struct ShaderCameraView {
    pub proj: Mat4,
    pub view: Mat4,
    pub rotation_only_view: Mat4,
    pub position: Vec3,
    pub near_plane_distance: f32,
    pub far_plane_distance: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    rotation_only_view: [[f32; 4]; 4],
    position: [f32; 4],
    near_plane_distance: f32,
    far_plane_distance: f32,
    padding: [f32; 2],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            rotation_only_view: Mat4::IDENTITY.to_cols_array_2d(),
            position: [0.0; 4],
            near_plane_distance: 0.0,
            far_plane_distance: 0.0,
            padding: [0.0; 2],
        }
    }
}

impl ShaderCameraView {
    pub fn from_transform(
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
        // orthographic instead of perspective:
        // build_directional_light_camera_view(
        //     Vec3::new(-0.5, -0.5, 0.1).normalize(),
        //     100.0,
        //     100.0,
        //     100.0,
        // )
        // .into()
    }
}

impl From<ShaderCameraView> for CameraUniform {
    fn from(
        ShaderCameraView {
            proj,
            view,
            rotation_only_view,
            position,
            near_plane_distance,
            far_plane_distance,
        }: ShaderCameraView,
    ) -> CameraUniform {
        Self {
            proj: proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            rotation_only_view: rotation_only_view.to_cols_array_2d(),
            position: [position.x, position.y, position.z, 1.0],
            near_plane_distance,
            far_plane_distance,
            padding: [0.0; 2],
        }
    }
}

pub fn build_cubemap_face_camera_views(
    position: Vec3,
    near_plane_distance: f32,
    far_plane_distance: f32,
    reverse_z: bool,
) -> Vec<ShaderCameraView> {
    vec![
        (90.0, 0.0),    // right
        (-90.0, 0.0),   // left
        (180.0, 90.0),  // top
        (180.0, -90.0), // bottom
        (180.0, 0.0),   // front
        (0.0, 0.0),     // back
    ]
    .iter()
    .map(
        |(horizontal_rotation, vertical_rotation): &(f32, f32)| Camera {
            horizontal_rotation: deg_to_rad(*horizontal_rotation),
            vertical_rotation: deg_to_rad(*vertical_rotation),
            position,
        },
    )
    .map(|camera| {
        ShaderCameraView::from_transform(
            camera.to_transform().matrix(),
            1.0,
            near_plane_distance,
            far_plane_distance,
            deg_to_rad(90.0),
            reverse_z,
        )
    })
    .collect()
}

pub fn build_directional_light_camera_view(
    direction: Vec3,
    width: f32,
    height: f32,
    depth: f32,
) -> ShaderCameraView {
    let proj = make_orthographic_proj_matrix(width, height, -depth / 2.0, depth / 2.0, false);
    let rotation_only_view = direction_vector_to_coordinate_frame_matrix(direction).inverse();
    let view = rotation_only_view;
    ShaderCameraView {
        proj,
        view,
        rotation_only_view,
        position: Vec3::new(0.0, 0.0, 0.0),
        near_plane_distance: -depth / 2.0,
        far_plane_distance: depth / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use glam::f32::Vec4;

    use super::*;

    #[test]
    fn should_i_exist() {
        let reverse_z_mat =
            make_perspective_proj_matrix(0.1, 100000.0, deg_to_rad(90.0), 1.0, true);
        let reg_z_mat = make_perspective_proj_matrix(0.1, 100000.0, deg_to_rad(90.0), 1.0, false);
        let pos = Vec4::new(-0.5, -0.5, -0.11, 1.0);
        let reverse_proj_pos = reverse_z_mat * pos;
        let reg_proj_pos = reg_z_mat * pos;
        let persp_div = |yo: Vec4| yo / yo.w;
        println!("{:?}", reverse_z_mat);
        println!("{:?}", reg_z_mat);
        println!("{:?}", reverse_proj_pos);
        println!("{:?}", reg_proj_pos);
        println!("{:?}", persp_div(reverse_proj_pos));
        println!("{:?}", persp_div(reg_proj_pos));
        assert_eq!(true, true);
    }
}
