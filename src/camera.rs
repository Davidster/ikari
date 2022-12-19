use super::*;
use cgmath::{Deg, Euler, Matrix, Matrix4, Quaternion, Rad, Vector3, Vector4};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Camera {
    pub horizontal_rotation: Rad<f32>,
    pub vertical_rotation: Rad<f32>,
    pub position: Vector3<f32>,
}

impl Camera {
    pub fn to_transform(self) -> crate::transform::Transform {
        TransformBuilder::new()
            .position(self.position)
            .rotation(
                Quaternion::from(Euler::new(Rad(0.0), self.horizontal_rotation, Rad(0.0)))
                    * Quaternion::from(Euler::new(self.vertical_rotation, Rad(0.0), Rad(0.0))),
            )
            .build()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ShaderCameraView {
    pub proj: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub rotation_only_view: Matrix4<f32>,
    pub position: Vector3<f32>,
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
            proj: Matrix4::one().into(),
            view: Matrix4::one().into(),
            rotation_only_view: Matrix4::one().into(),
            position: [0.0; 4],
            near_plane_distance: 0.0,
            far_plane_distance: 0.0,
            padding: [0.0; 2],
        }
    }
}

impl ShaderCameraView {
    pub fn from_transform(
        transform: Matrix4<f32>,
        aspect_ratio: f32,
        near_plane_distance: f32,
        far_plane_distance: f32,
        fov_y: Rad<f32>,
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
        let rotation_only_view = rotation_only_matrix.inverse_transform().unwrap();
        let view = transform.inverse_transform().unwrap();
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
        //     Vector3::new(-0.5, -0.5, 0.1).normalize(),
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
            proj: proj.into(),
            view: view.into(),
            rotation_only_view: rotation_only_view.into(),
            position: [position.x, position.y, position.z, 1.0],
            near_plane_distance,
            far_plane_distance,
            padding: [0.0; 2],
        }
    }
}

impl Frustum {
    pub fn from_camera_params(
        camera_position: Vector3<f32>,
        transform: Matrix4<f32>,
        aspect_ratio: f32,
        near_plane_distance: f32,
        far_plane_distance: f32,
        fov_y: Rad<f32>,
    ) -> Self {
        // see https://learnopengl.com/Guest-Articles/2021/Scene/Frustum-Culling
        let transform_t = transform;
        let up = transform_t.y.truncate();
        let forward = transform_t.z.truncate();
        let right = forward.cross(up);
        // dbg!(up, forward, right);
        let half_v_side = far_plane_distance * (fov_y.0 * 0.5).tan();
        let half_h_side = half_v_side * aspect_ratio;
        let front_mult_far = far_plane_distance * forward;

        Self {
            left: Plane::from_normal_and_point(
                (front_mult_far - right * half_h_side).cross(up),
                camera_position,
            ),
            right: Plane::from_normal_and_point(
                up.cross(front_mult_far + right * half_h_side),
                camera_position,
            ),
            bottom: Plane::from_normal_and_point(
                (front_mult_far + up * half_v_side).cross(right),
                camera_position,
            ),
            top: Plane::from_normal_and_point(
                right.cross(front_mult_far - up * half_v_side),
                camera_position,
            ),
            near: Plane::from_normal_and_point(
                forward,
                camera_position + near_plane_distance * forward,
            ),
            far: Plane::from_normal_and_point(-forward, camera_position + front_mult_far),
        }

        // gdbooks version
        // let proj = make_perspective_proj_matrix(
        //     near_plane_distance,
        //     far_plane_distance,
        //     fov_y,
        //     aspect_ratio,
        //     false,
        // );
        // let view = transform.inverse_transform().unwrap();
        // Self::from_view_proj(view, proj)
    }

    // see https://gdbooks.gitbooks.io/legacyopengl/content/Chapter8/frustum.html
    // fn from_view_proj(view: Matrix4<f32>, proj: Matrix4<f32>) -> Self {
    //     let mv = (proj * view).transpose();

    //     let to_plane = |p: Vector4<f32>| Plane {
    //         normal: Vector3::new(p.x, p.y, p.z).normalize(),
    //         d: p.w,
    //     };

    //     Self {
    //         left: to_plane(mv.w + mv.x),
    //         right: to_plane(mv.w - mv.x),
    //         bottom: to_plane(mv.w + mv.y),
    //         top: to_plane(mv.w - mv.y),
    //         near: to_plane(mv.w + mv.z),
    //         far: to_plane(mv.w - mv.z),
    //     }
    // }

    // fn from_view_proj(view: Matrix4<f32>, proj: Matrix4<f32>) -> Self {
    //     todo!()
    // }
}

// impl From<ShaderCameraView> for Frustum {
//     fn from(ShaderCameraView { position, far_plane_distance, near_plane_distance }: ShaderCameraView) -> Frustum {
//         Frustum::from_view_proj(view, proj)
//     }
// }

pub fn build_cubemap_face_camera_views(
    position: Vector3<f32>,
    near_plane_distance: f32,
    far_plane_distance: f32,
    reverse_z: bool,
) -> Vec<ShaderCameraView> {
    vec![
        (Deg(90.0), Deg(0.0)),    // right
        (Deg(-90.0), Deg(0.0)),   // left
        (Deg(180.0), Deg(90.0)),  // top
        (Deg(180.0), Deg(-90.0)), // bottom
        (Deg(180.0), Deg(0.0)),   // front
        (Deg(0.0), Deg(0.0)),     // back
    ]
    .iter()
    .map(
        |(horizontal_rotation, vertical_rotation): &(Deg<f32>, Deg<f32>)| Camera {
            horizontal_rotation: (*horizontal_rotation).into(),
            vertical_rotation: (*vertical_rotation).into(),
            position,
        },
    )
    .map(|camera| {
        ShaderCameraView::from_transform(
            camera.to_transform().matrix(),
            1.0,
            near_plane_distance,
            far_plane_distance,
            Deg(90.0).into(),
            reverse_z,
        )
    })
    .collect()
}

pub fn build_directional_light_camera_view(
    direction: Vector3<f32>,
    width: f32,
    height: f32,
    depth: f32,
) -> ShaderCameraView {
    let proj = make_orthographic_proj_matrix(width, height, -depth / 2.0, depth / 2.0, false);
    let rotation_only_view = direction_vector_to_coordinate_frame_matrix(direction)
        .inverse_transform()
        .unwrap();
    let view = rotation_only_view;
    ShaderCameraView {
        proj,
        view,
        rotation_only_view,
        position: Vector3::new(0.0, 0.0, 0.0),
        near_plane_distance: -depth / 2.0,
        far_plane_distance: depth / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use cgmath::Vector4;

    use super::*;

    #[test]
    fn should_i_exist() {
        let reverse_z_mat =
            make_perspective_proj_matrix(0.1, 100000.0, cgmath::Deg(90.0).into(), 1.0, true);
        let reg_z_mat =
            make_perspective_proj_matrix(0.1, 100000.0, cgmath::Deg(90.0).into(), 1.0, false);
        let pos = Vector4::new(-0.5, -0.5, -0.11, 1.0);
        let reverse_proj_pos = reverse_z_mat * pos;
        let reg_proj_pos = reg_z_mat * pos;
        let persp_div = |yo: Vector4<f32>| yo / yo.w;
        println!("{:?}", reverse_z_mat);
        println!("{:?}", reg_z_mat);
        println!("{:?}", reverse_proj_pos);
        println!("{:?}", reg_proj_pos);
        println!("{:?}", persp_div(reverse_proj_pos));
        println!("{:?}", persp_div(reg_proj_pos));
        assert_eq!(true, true);
    }
}
