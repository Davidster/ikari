use super::*;

use anyhow::Result;
use cgmath::{Deg, Rad, Vector3, Vector4};
use rapier3d::prelude::*;
use winit::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};

pub const INITIAL_RENDER_SCALE: f32 = 1.0;
pub const INITIAL_TONE_MAPPING_EXPOSURE: f32 = 0.5;
pub const INITIAL_BLOOM_THRESHOLD: f32 = 0.8;
pub const INITIAL_BLOOM_RAMP_SIZE: f32 = 0.2;
pub const ARENA_SIDE_LENGTH: f32 = 25.0;
pub const LIGHT_COLOR_A: Vector3<f32> = Vector3::new(0.996, 0.973, 0.663);
pub const LIGHT_COLOR_B: Vector3<f32> = Vector3::new(0.25, 0.973, 0.663);

#[allow(clippy::let_and_return)]
fn get_gltf_path() -> &'static str {
    // let gltf_path = "/home/david/Downloads/adamHead/adamHead.gltf";
    // let gltf_path = "/home/david/Downloads/free_low_poly_forest/scene_2.glb";
    // let gltf_path = "/home/david/Downloads/free_low_poly_forest/scene.gltf";
    // let gltf_path = "./src/models/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf";
    // let gltf_path = "./src/models/gltf/SimpleMeshes/SimpleMeshes.gltf";
    // let gltf_path = "./src/models/gltf/Triangle/Triangle.gltf";
    // let gltf_path = "./src/models/gltf/TriangleWithoutIndices/TriangleWithoutIndices.gltf";
    // let gltf_path = "./src/models/gltf/Sponza/Sponza.gltf";
    // let gltf_path = "./src/models/gltf/EnvironmentTest/EnvironmentTest.gltf";
    // let gltf_path = "./src/models/gltf/Arrow/Arrow.gltf";
    // let gltf_path = "./src/models/gltf/DamagedHelmet/DamagedHelmet.gltf";
    // let gltf_path = "./src/models/gltf/VertexColorTest/VertexColorTest.gltf";
    // let gltf_path = "./src/models/gltf/Revolver/revolver_low_poly.gltf";
    // let gltf_path =
    //     "/home/david/Programming/glTF-Sample-Models/2.0/BoomBoxWithAxes/glTF/BoomBoxWithAxes.gltf";
    // let gltf_path =
    //     "./src/models/gltf/TextureLinearInterpolationTest/TextureLinearInterpolationTest.glb";
    // let gltf_path = "../glTF-Sample-Models/2.0/RiggedFigure/glTF/RiggedFigure.gltf";
    // let gltf_path = "../glTF-Sample-Models/2.0/RiggedSimple/glTF/RiggedSimple.gltf";
    let gltf_path = "../glTF-Sample-Models/2.0/CesiumMan/glTF/CesiumMan.gltf";
    // let gltf_path = "../glTF-Sample-Models/2.0/Fox/glTF/Fox.gltf";
    // let gltf_path = "../glTF-Sample-Models/2.0/RecursiveSkeletons/glTF/RecursiveSkeletons.gltf";
    // let gltf_path = "../glTF-Sample-Models/2.0/BrainStem/glTF/BrainStem.gltf";
    // let gltf_path =
    //     "/home/david/Programming/glTF-Sample-Models/2.0/BoxAnimated/glTF/BoxAnimated.gltf";
    // let gltf_path = "/home/david/Programming/glTF-Sample-Models/2.0/InterpolationTest/glTF/InterpolationTest.gltf";
    // let gltf_path = "./src/models/gltf/VC/VC.gltf";
    // let gltf_path =
    //     "../glTF-Sample-Models-master/2.0/InterpolationTest/glTF/InterpolationTest.gltf";
    gltf_path
}

pub fn get_skybox_path() -> (
    SkyboxBackground<'static>,
    Option<SkyboxHDREnvironment<'static>>,
) {
    // Mountains
    // src: https://github.com/JoeyDeVries/LearnOpenGL/tree/master/resources/textures/skybox
    let _skybox_background = SkyboxBackground::Cube {
        face_image_paths: [
            "./src/textures/skybox/right.jpg",
            "./src/textures/skybox/left.jpg",
            "./src/textures/skybox/top.jpg",
            "./src/textures/skybox/bottom.jpg",
            "./src/textures/skybox/front.jpg",
            "./src/textures/skybox/back.jpg",
        ],
    };
    let _skybox_hdr_environment: Option<SkyboxHDREnvironment> = None;

    // Newport Loft
    // src: http://www.hdrlabs.com/sibl/archive/
    let skybox_background = SkyboxBackground::Equirectangular {
        image_path: "./src/textures/newport_loft/background.jpg",
    };
    let skybox_hdr_environment: Option<SkyboxHDREnvironment> =
        Some(SkyboxHDREnvironment::Equirectangular {
            image_path: "./src/textures/newport_loft/radiance.hdr",
        });

    // My photosphere pic
    // src: me
    let _skybox_background = SkyboxBackground::Equirectangular {
        image_path: "./src/textures/photosphere_skybox.jpg",
    };
    let _skybox_hdr_environment: Option<SkyboxHDREnvironment> =
        Some(SkyboxHDREnvironment::Equirectangular {
            image_path: "./src/textures/photosphere_skybox_small.jpg",
        });

    (skybox_background, skybox_hdr_environment)
}

pub fn init_game_state(
    mut scene: GameScene,
    renderer_state: &mut RendererState,
    logger: &mut Logger,
) -> Result<GameState> {
    let sphere_mesh = BasicMesh::new("./src/models/sphere.obj")?;
    let plane_mesh = BasicMesh::new("./src/models/plane.obj")?;
    let _cube_mesh = BasicMesh::new("./src/models/cube.obj")?;

    let mut physics_state = PhysicsState::new();

    let mut camera = Camera::new((0.0, 16.0, 33.0).into());
    camera.vertical_rotation = Rad(-0.53);
    let camera_controller = CameraController::new(6.0, camera);
    let camera_node_id = scene.add_node(GameNodeDesc::default()).id();

    // add lights to the scene
    let directional_lights = vec![DirectionalLightComponent {
        position: Vector3::new(10.0, 5.0, 0.0) * 10.0,
        direction: Vector3::new(-1.0, -0.7, 0.0).normalize(),
        color: LIGHT_COLOR_A,
        intensity: 1.0,
    }];
    // let directional_lights: Vec<DirectionalLightComponent> = vec![];

    let point_lights: Vec<(transform::Transform, Vector3<f32>, f32)> = vec![
        (
            TransformBuilder::new()
                .scale(Vector3::new(0.05, 0.05, 0.05))
                .position(Vector3::new(0.0, 12.0, 0.0))
                .build(),
            LIGHT_COLOR_A,
            1.0,
        ),
        (
            TransformBuilder::new()
                .scale(Vector3::new(0.1, 0.1, 0.1))
                .position(Vector3::new(0.0, 15.0, 0.0))
                .build(),
            LIGHT_COLOR_B,
            1.0,
        ),
    ];
    // let point_lights: Vec<(transform::Transform, Vector3<f32>)> = vec![];

    let point_light_unlit_mesh_index = renderer_state.bind_basic_unlit_mesh(&sphere_mesh)?;
    let mut point_light_node_ids: Vec<GameNodeId> = Vec::new();
    let mut point_light_components: Vec<PointLightComponent> = Vec::new();
    for (transform, color, intensity) in &point_lights {
        let node_id = scene
            .add_node(
                GameNodeDescBuilder::new()
                    .mesh(Some(GameNodeMesh::Unlit {
                        mesh_indices: vec![point_light_unlit_mesh_index],
                        color: color * *intensity,
                    }))
                    .transform(*transform)
                    .build(),
            )
            .id();
        point_light_node_ids.push(node_id);
        point_light_components.push(PointLightComponent {
            node_id,
            color: LIGHT_COLOR_A,
            intensity: *intensity,
        });
    }

    // rotate the animated character 90 deg
    if let Some(node_0) = scene._get_node_mut_by_index(0) {
        // node_0.transform.set_rotation(make_quat_from_axis_angle(
        //     Vector3::new(0.0, 1.0, 0.0),
        //     Deg(90.0).into(),
        // ));
        // node_0.transform.set_scale(Vector3::new(0.0, 0.0, 0.0));
        node_0.transform.set_position(Vector3::new(2.0, 0.0, 0.0));
    }

    // let simple_normal_map_path = "./src/textures/simple_normal_map.jpg";
    // let simple_normal_map_bytes = std::fs::read(simple_normal_map_path)?;
    // let simple_normal_map = Texture::from_encoded_image(
    //     &renderer_state.base.device,
    //     &renderer_state.base.queue,
    //     &simple_normal_map_bytes,
    //     simple_normal_map_path,
    //     wgpu::TextureFormat::Rgba8Unorm.into(),
    //     false,
    //     &Default::default(),
    // )?;

    // let brick_normal_map_path = "./src/textures/brick_normal_map.jpg";
    // let brick_normal_map_bytes = std::fs::read(brick_normal_map_path)?;
    // let brick_normal_map = Texture::from_encoded_image(
    //     &renderer_state.base.device,
    //     &renderer_state.base.queue,
    //     &brick_normal_map_bytes,
    //     brick_normal_map_path,
    //     wgpu::TextureFormat::Rgba8Unorm.into(),
    //     false,
    //     &Default::default(),
    // )?;

    // add test object to scene
    let earth_texture_path = "./src/textures/8k_earth.jpg";
    let earth_texture_bytes = std::fs::read(earth_texture_path)?;
    let earth_texture = Texture::from_encoded_image(
        &renderer_state.base.device,
        &renderer_state.base.queue,
        &earth_texture_bytes,
        earth_texture_path,
        None,
        true,
        &Default::default(),
    )?;

    let earth_normal_map_path = "./src/textures/8k_earth_normal_map.jpg";
    let earth_normal_map_bytes = std::fs::read(earth_normal_map_path)?;
    let earth_normal_map = Texture::from_encoded_image(
        &renderer_state.base.device,
        &renderer_state.base.queue,
        &earth_normal_map_bytes,
        earth_normal_map_path,
        wgpu::TextureFormat::Rgba8Unorm.into(),
        false,
        &Default::default(),
    )?;

    let test_object_metallic_roughness_map = Texture::from_color(
        &renderer_state.base.device,
        &renderer_state.base.queue,
        [
            255,
            (0.12 * 255.0f32).round() as u8,
            (0.8 * 255.0f32).round() as u8,
            255,
        ],
    )?;

    let test_object_pbr_mesh_index = renderer_state.bind_basic_pbr_mesh(
        &sphere_mesh,
        &PbrMaterial {
            base_color: Some(&earth_texture),
            normal: Some(&earth_normal_map),
            metallic_roughness: Some(&test_object_metallic_roughness_map),
            ..Default::default()
        },
        Default::default(),
    )?;
    let test_object_node_id = scene
        .add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(GameNodeMesh::Pbr {
                    mesh_indices: vec![test_object_pbr_mesh_index],
                    material_override: None,
                }))
                .transform(
                    TransformBuilder::new()
                        .position(Vector3::new(4.0, 10.0, 4.0))
                        // .scale(Vector3::new(0.0, 0.0, 0.0))
                        .build(),
                )
                .build(),
        )
        .id();

    // add floor to scene
    let big_checkerboard_texture_img = {
        let mut img = image::RgbaImage::new(4096, 4096);
        for x in 0..img.width() {
            for y in 0..img.height() {
                let scale = 10;
                let x_scaled = x / scale;
                let y_scaled = y / scale;
                img.put_pixel(
                    x,
                    y,
                    if (x_scaled + y_scaled) % 2 == 0 {
                        [100, 100, 100, 100].into()
                    } else {
                        [150, 150, 150, 150].into()
                    },
                );
            }
        }
        img
    };
    let big_checkerboard_texture = Texture::from_decoded_image(
        &renderer_state.base.device,
        &renderer_state.base.queue,
        &big_checkerboard_texture_img,
        big_checkerboard_texture_img.dimensions(),
        Some("big_checkerboard_texture"),
        None,
        true,
        &texture::SamplerDescriptor(wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            ..texture::SamplerDescriptor::default().0
        }),
    )?;

    let small_checkerboard_texture_img = {
        let mut img = image::RgbaImage::new(1080, 1080);
        for x in 0..img.width() {
            for y in 0..img.height() {
                let scale = 25;
                let x_scaled = x / scale;
                let y_scaled = y / scale;
                img.put_pixel(
                    x,
                    y,
                    if (x_scaled + y_scaled) % 2 == 0 {
                        [100, 100, 100, 100].into()
                    } else {
                        [150, 150, 150, 150].into()
                    },
                );
            }
        }
        img
    };
    let small_checkerboard_texture = Texture::from_decoded_image(
        &renderer_state.base.device,
        &renderer_state.base.queue,
        &small_checkerboard_texture_img,
        small_checkerboard_texture_img.dimensions(),
        Some("small_checkerboard_texture"),
        None,
        true,
        &texture::SamplerDescriptor(wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            ..texture::SamplerDescriptor::default().0
        }),
    )?;

    // add balls to scene

    // source: https://www.solarsystemscope.com/textures/
    let mars_texture_path = "./src/textures/8k_mars.jpg";
    let mars_texture_bytes = std::fs::read(mars_texture_path)?;
    let mars_texture = Texture::from_encoded_image(
        &renderer_state.base.device,
        &renderer_state.base.queue,
        &mars_texture_bytes,
        mars_texture_path,
        None,
        true,
        &Default::default(),
    )?;

    let ball_count = 0;
    let balls: Vec<_> = (0..ball_count)
        .into_iter()
        .map(|_| BallComponent::rand())
        .collect();

    let ball_pbr_mesh_index = renderer_state.bind_basic_pbr_mesh(
        &sphere_mesh,
        &PbrMaterial {
            base_color: Some(&mars_texture),
            ..Default::default()
        },
        Default::default(),
    )?;

    let mut ball_node_ids: Vec<GameNodeId> = Vec::new();
    for ball in &balls {
        let node = scene.add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(GameNodeMesh::Pbr {
                    mesh_indices: vec![ball_pbr_mesh_index],
                    material_override: None,
                }))
                .transform(ball.transform)
                .build(),
        );
        ball_node_ids.push(node.id());
    }

    let physics_ball_count = 25;
    let physics_balls: Vec<_> = (0..physics_ball_count)
        .into_iter()
        .map(|_| {
            PhysicsBall::new_random(
                &mut scene,
                &mut physics_state,
                GameNodeMesh::Pbr {
                    mesh_indices: vec![ball_pbr_mesh_index],
                    material_override: None,
                },
            )
        })
        .collect();

    // let box_pbr_mesh_index = renderer_state.bind_basic_pbr_mesh(
    //     &cube_mesh,
    //     &PbrMaterial {
    //         diffuse: Some(&checkerboard_texture),
    //         ..Default::default()
    //     },
    //     Default::default(),
    // )?;
    // scene.nodes.push(
    //     GameNodeBuilder::new()
    //         .mesh(Some(GameNodeMesh::Pbr {
    //             mesh_indices: vec![box_pbr_mesh_index],
    //             material_override: None,
    //         }))
    //         .transform(
    //             TransformBuilder::new()
    //                 .scale(Vector3::new(0.5, 0.5, 0.5))
    //                 .position(Vector3::new(0.0, 0.5, 0.0))
    //                 .build(),
    //         )
    //         .build(),
    // );

    // create the floor and add it to the scene
    let floor_pbr_mesh_index = renderer_state.bind_basic_pbr_mesh(
        &plane_mesh,
        &PbrMaterial {
            base_color: Some(&big_checkerboard_texture),
            ..Default::default()
        },
        Default::default(),
    )?;
    let floor_node = scene.add_node(
        GameNodeDescBuilder::new()
            .mesh(Some(GameNodeMesh::Pbr {
                mesh_indices: vec![floor_pbr_mesh_index],
                material_override: None,
            }))
            .transform(
                TransformBuilder::new()
                    .scale(Vector3::new(ARENA_SIDE_LENGTH, 1.0, ARENA_SIDE_LENGTH))
                    .build(),
            )
            .build(),
    );
    let floor_thickness = 0.1;
    let floor_rigid_body = RigidBodyBuilder::fixed()
        .translation(vector![
            floor_node.transform.position().x / 2.0,
            floor_node.transform.position().y - floor_thickness / 2.0,
            floor_node.transform.position().z / 2.0
        ])
        .build();
    let floor_collider = ColliderBuilder::cuboid(
        floor_node.transform.scale().x,
        floor_thickness / 2.0,
        floor_node.transform.scale().z,
    )
    .friction(1.0)
    .restitution(1.0)
    .build();
    let floor_body_handle = physics_state.rigid_body_set.insert(floor_rigid_body);
    physics_state.collider_set.insert_with_parent(
        floor_collider,
        floor_body_handle,
        &mut physics_state.rigid_body_set,
    );

    // create the checkerboarded bouncing ball and add it to the scene
    let (bouncing_ball_node_id, bouncing_ball_body_handle) = {
        let bouncing_ball_pbr_mesh_index = renderer_state.bind_basic_pbr_mesh(
            &sphere_mesh,
            &PbrMaterial {
                base_color: Some(&small_checkerboard_texture),
                ..Default::default()
            },
            Default::default(),
        )?;
        let bouncing_ball_radius = 0.5;
        let bouncing_ball_node = scene.add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(GameNodeMesh::Pbr {
                    mesh_indices: vec![bouncing_ball_pbr_mesh_index],
                    material_override: None,
                }))
                .transform(
                    TransformBuilder::new()
                        .scale(Vector3::new(
                            bouncing_ball_radius,
                            bouncing_ball_radius,
                            bouncing_ball_radius,
                        ))
                        .position(Vector3::new(-1.0, 10.0, 0.0))
                        .build(),
                )
                .build(),
        );
        let bouncing_ball_rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![
                bouncing_ball_node.transform.position().x,
                bouncing_ball_node.transform.position().y,
                bouncing_ball_node.transform.position().z
            ])
            .build();
        let bouncing_ball_collider = ColliderBuilder::ball(bouncing_ball_radius)
            .restitution(0.9)
            .build();
        let bouncing_ball_body_handle = physics_state
            .rigid_body_set
            .insert(bouncing_ball_rigid_body);
        physics_state.collider_set.insert_with_parent(
            bouncing_ball_collider,
            bouncing_ball_body_handle,
            &mut physics_state.rigid_body_set,
        );
        (bouncing_ball_node.id(), bouncing_ball_body_handle)
    };

    // add crosshair to scene
    let crosshair_texture_img = {
        let thickness = 8;
        let gap = 30;
        let bar_length = 14;
        let texture_size = 512;
        let mut img = image::RgbaImage::new(texture_size, texture_size);
        for x in 0..img.width() {
            for y in 0..img.height() {
                let d_x = ((x as i32) - ((texture_size / 2) as i32)).abs();
                let d_y = ((y as i32) - ((texture_size / 2) as i32)).abs();

                let inside_vertical_bar =
                    d_x < thickness / 2 && d_y > gap / 2 && d_y < gap + bar_length;
                let inside_horizontal_bar =
                    d_y < thickness / 2 && d_x > gap / 2 && d_x < gap + bar_length;

                img.put_pixel(
                    x,
                    y,
                    if inside_vertical_bar || inside_horizontal_bar {
                        [255, 255, 255, 255].into()
                    } else {
                        [0, 0, 0, 0].into()
                    },
                );
            }
        }
        img
    };
    let crosshair_texture = Texture::from_decoded_image(
        &renderer_state.base.device,
        &renderer_state.base.queue,
        &crosshair_texture_img,
        crosshair_texture_img.dimensions(),
        Some("crosshair_texture"),
        None,
        false,
        &texture::SamplerDescriptor(wgpu::SamplerDescriptor {
            // mag_filter: wgpu::FilterMode::Nearest,
            // min_filter: wgpu::FilterMode::Nearest,
            // mipmap_filter: wgpu::FilterMode::Nearest,
            ..texture::SamplerDescriptor::default().0
        }),
    )?;
    let crosshair_quad = BasicMesh {
        vertices: vec![[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]]
            .iter()
            .map(|position| Vertex {
                position: [0.0, position[1], position[0]],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.5 * (position[0] + 1.0), 0.5 * (1.0 - position[1])],
                tangent: [1.0, 0.0, 0.0],
                bitangent: [0.0, -1.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
                bone_indices: [0, 1, 2, 3],
                bone_weights: [1.0, 0.0, 0.0, 0.0],
            })
            .collect(),
        indices: vec![0, 2, 1, 0, 3, 2],
    };
    let pbr_mesh_index = renderer_state.bind_basic_pbr_mesh(
        &crosshair_quad,
        &PbrMaterial {
            ambient_occlusion: Some(&Texture::from_color(
                &renderer_state.base.device,
                &renderer_state.base.queue,
                [0, 0, 0, 0],
            )?),
            metallic_roughness: Some(&Texture::from_color(
                &renderer_state.base.device,
                &renderer_state.base.queue,
                [0, 0, 255, 0],
            )?),
            base_color: Some(&crosshair_texture),
            emissive: Some(&crosshair_texture),
            ..Default::default()
        },
        Default::default(),
    )?;
    let crosshair_color = Vector3::new(1.0, 0.0, 0.0);
    let crosshair_node_id = scene
        .add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(GameNodeMesh::Pbr {
                    mesh_indices: vec![pbr_mesh_index],
                    material_override: Some(DynamicPbrParams {
                        emissive_factor: crosshair_color,
                        base_color_factor: Vector4::new(0.0, 0.0, 0.0, 1.0),
                        alpha_cutoff: 0.5,
                        ..Default::default()
                    }),
                }))
                .build(),
        )
        .id();

    // merge revolver scene into current scene
    // let (document, buffers, images) =
    //     gltf::import("./src/models/gltf/Revolver/revolver_low_poly.gltf")?;
    let (document, buffers, images) =
        gltf::import("../glTF-Sample-Models/2.0/BrainStem/glTF/BrainStem.gltf")?;
    validate_animation_property_counts(&document, logger);
    let (other_game_scene, other_render_buffers) =
        build_scene(&renderer_state.base, (&document, &buffers, &images))?;
    scene.merge_scene(renderer_state, other_game_scene, other_render_buffers);

    Ok(GameState {
        scene,
        time_tracker: None,
        state_update_time_accumulator: 0.0,

        camera_controller,
        camera_node_id,

        point_lights: point_light_components,
        point_light_node_ids,
        directional_lights,

        next_balls: balls.clone(),
        prev_balls: balls.clone(),
        actual_balls: balls,
        ball_node_ids,
        ball_pbr_mesh_index,

        ball_spawner_acc: 0.0,

        test_object_node_id,
        crosshair_node_id,

        bouncing_ball_node_id,
        bouncing_ball_body_handle,

        physics_state,

        physics_balls,
        mouse_button_pressed: false,
    })
}

pub fn process_device_input(
    game_state: &mut GameState,
    event: &winit::event::DeviceEvent,
    logger: &mut Logger,
) {
    game_state
        .camera_controller
        .process_device_events(event, logger);
}

pub fn process_window_input(
    game_state: &mut GameState,
    renderer_state: &mut RendererState,
    event: &winit::event::WindowEvent,
    window: &mut winit::window::Window,
    logger: &mut Logger,
) {
    if let WindowEvent::MouseInput {
        state,
        button: MouseButton::Left,
        ..
    } = event
    {
        game_state.mouse_button_pressed = *state == ElementState::Pressed;
    }
    if let WindowEvent::KeyboardInput {
        input:
            KeyboardInput {
                state,
                virtual_keycode: Some(keycode),
                ..
            },
        ..
    } = event
    {
        if *state == ElementState::Released {
            match keycode {
                VirtualKeyCode::Z => {
                    renderer_state.increment_render_scale(false, logger);
                }
                VirtualKeyCode::X => {
                    renderer_state.increment_render_scale(true, logger);
                }
                VirtualKeyCode::E => {
                    renderer_state.increment_exposure(false, logger);
                }
                VirtualKeyCode::R => {
                    renderer_state.increment_exposure(true, logger);
                }
                VirtualKeyCode::T => {
                    renderer_state.increment_bloom_threshold(false, logger);
                }
                VirtualKeyCode::Y => {
                    renderer_state.increment_bloom_threshold(true, logger);
                }
                VirtualKeyCode::P => {
                    renderer_state.toggle_animations();
                }
                VirtualKeyCode::M => {
                    renderer_state.toggle_shadows();
                }
                VirtualKeyCode::B => {
                    renderer_state.toggle_bloom();
                }
                _ => {}
            }
        }
    }
    game_state
        .camera_controller
        .process_window_events(event, window, logger);
}

pub fn update_game_state(
    game_state: &mut GameState,
    renderer_state: &RendererState,
    logger: &mut Logger,
) {
    let time_tracker = game_state.time();
    let global_time_seconds = time_tracker.global_time_seconds();

    // results in ~60 state changes per second
    let min_update_timestep_seconds = 1.0 / 60.0;
    // if frametime takes longer than this, we give up on trying to catch up completely
    // prevents the game from getting stuck in a spiral of death
    let max_delay_catchup_seconds = 0.25;
    let mut frame_time_seconds = time_tracker.last_frame_time_seconds();
    if frame_time_seconds > max_delay_catchup_seconds {
        frame_time_seconds = max_delay_catchup_seconds;
    }
    game_state.state_update_time_accumulator += frame_time_seconds;

    game_state.physics_state.step();

    game_state.camera_controller.update(frame_time_seconds);
    // logger.log(&format!(
    //     "camera pose: {:?}",
    //     game_state.camera_controller.current_pose
    // ));
    let new_camera_transform = game_state.camera_controller.current_pose.to_transform();
    if let Some(camera_transform) = game_state.scene.get_node_mut(game_state.camera_node_id) {
        camera_transform.transform = new_camera_transform.into();
    }

    // update ball positions
    while game_state.state_update_time_accumulator >= min_update_timestep_seconds {
        if game_state.state_update_time_accumulator < min_update_timestep_seconds * 2.0 {
            game_state.prev_balls = game_state.next_balls.clone();
        }
        game_state.prev_balls = game_state.next_balls.clone();
        game_state
            .next_balls
            .iter_mut()
            .for_each(|ball| ball.update(min_update_timestep_seconds, logger));
        game_state.state_update_time_accumulator -= min_update_timestep_seconds;
    }
    let alpha = game_state.state_update_time_accumulator / min_update_timestep_seconds;
    game_state.actual_balls = game_state
        .prev_balls
        .iter()
        .zip(game_state.next_balls.iter())
        .map(|(prev_ball, next_ball)| prev_ball.lerp(next_ball, alpha))
        .collect();
    game_state
        .ball_node_ids
        .iter()
        .zip(game_state.actual_balls.iter())
        .for_each(|(node_id, ball)| {
            if let Some(node) = game_state.scene.get_node_mut(*node_id) {
                node.transform = ball.transform;
            }
        });

    if let Some(point_light_0) = game_state.point_lights.get_mut(0) {
        point_light_0.color = lerp_vec(
            LIGHT_COLOR_A,
            LIGHT_COLOR_B,
            (global_time_seconds * 2.0).sin(),
        );
        if let Some(node) = game_state.scene.get_node_mut(point_light_0.node_id) {
            node.transform.set_position(Vector3::new(
                1.5 * (global_time_seconds * 0.25 + std::f32::consts::PI).cos(),
                node.transform.position().y - frame_time_seconds * 0.25,
                1.5 * (global_time_seconds * 0.25 + std::f32::consts::PI).sin(),
            ));
        }
    }

    if let Some(point_light_1) = game_state.point_lights.get_mut(1) {
        point_light_1.color = lerp_vec(
            LIGHT_COLOR_B,
            LIGHT_COLOR_A,
            (global_time_seconds * 2.0).sin(),
        );
        // let transform = &mut game_state.scene.nodes[point_light_1.node_id].transform;
        // transform.set_position(Vector3::new(
        //     1.1 * (global_time_seconds * 0.25 + std::f32::consts::PI).cos(),
        //     transform.position().y,
        //     1.1 * (global_time_seconds * 0.25 + std::f32::consts::PI).sin(),
        // ));
    }

    // sync unlit mesh config with point light component
    game_state
        .point_light_node_ids
        .iter()
        .zip(game_state.point_lights.iter())
        .for_each(|(node_id, point_light)| {
            if let Some(GameNodeMesh::Unlit { ref mut color, .. }) = game_state
                .scene
                .get_node_mut(*node_id)
                .and_then(|node| node.mesh.as_mut())
            {
                *color = point_light.color * point_light.intensity;
            }
        });

    let directional_light_0 = game_state
        .directional_lights
        .get(0)
        .map(|directional_light_0| {
            let direction = directional_light_0.direction;
            // transform.set_position(Vector3::new(
            //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).cos(),
            //     transform.position.get().y,
            //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).sin(),
            // ));
            // let color = lerp_vec(LIGHT_COLOR_B, LIGHT_COLOR_A, (time_seconds * 2.0).sin());

            DirectionalLightComponent {
                direction: Vector3::new(direction.x, direction.y + 0.0001, direction.z),
                ..*directional_light_0
            }
        });
    if let Some(directional_light_0) = directional_light_0 {
        game_state.directional_lights[0] = directional_light_0;
    }

    // rotate the test object
    let rotational_displacement =
        make_quat_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(frame_time_seconds / 5.0));
    if let Some(node) = game_state
        .scene
        .get_node_mut(game_state.test_object_node_id)
    {
        node.transform
            .set_rotation(rotational_displacement * node.transform.rotation());
    }

    // logger.log(&format!("Frame time: {:?}", frame_time_seconds));
    // logger.log(&format!(
    //     "state_update_time_accumulator: {:?}",
    //     game_state.state_update_time_accumulator
    // ));

    // remove physics balls over time
    game_state.ball_spawner_acc += frame_time_seconds;
    let rate = 0.01;
    let prev_ball_count = game_state.physics_balls.len();
    while game_state.ball_spawner_acc > rate {
        // let new_ball = BallComponent::rand();
        // let new_ball_transform = new_ball.transform;
        // game_state.next_balls.push(new_ball);
        // game_state.scene.nodes.push(
        //     GameNodeBuilder::new()
        //         .mesh(Some(GameNodeMesh::Pbr {
        //             mesh_indices: vec![game_state.ball_pbr_mesh_index],
        //             material_override: None,
        //         }))
        //         .transform(new_ball_transform)
        //         .build(),
        // );
        // game_state
        //     .ball_node_indices
        //     .push(game_state.scene.nodes.len() - 1);
        // if let Some(physics_ball) = game_state.physics_balls.pop() {
        //     physics_ball.destroy(&mut game_state.scene, &mut game_state.physics_state);
        // }
        game_state.ball_spawner_acc -= rate;
    }
    let new_ball_count = game_state.physics_balls.len();
    if prev_ball_count != new_ball_count {
        logger.log(&format!("Ball count: {:?}", new_ball_count));
    }

    // let physics_time_step_start = Instant::now();

    // logger.log(&format!("Physics step time: {:?}", physics_time_step_start.elapsed()));
    let physics_state = &mut game_state.physics_state;
    let ball_body = &physics_state.rigid_body_set[game_state.bouncing_ball_body_handle];
    if let Some(node) = game_state
        .scene
        .get_node_mut(game_state.bouncing_ball_node_id)
    {
        node.transform.apply_isometry(*ball_body.position());
    }

    physics_state.integration_parameters.dt = frame_time_seconds;
    game_state
        .physics_balls
        .iter()
        .for_each(|physics_ball| physics_ball.update(&mut game_state.scene, physics_state));

    if let Some(crosshair_node) = game_state.scene.get_node_mut(game_state.crosshair_node_id) {
        crosshair_node.transform = crate::transform::Transform::from(new_camera_transform)
            * TransformBuilder::new()
                .position(Vector3::new(0.0, 0.0, -NEAR_PLANE_DISTANCE * 2.0))
                .rotation(make_quat_from_axis_angle(
                    Vector3::new(0.0, 1.0, 0.0),
                    Deg(90.0).into(),
                ))
                .scale(
                    (1080.0 / renderer_state.base.window_size.height as f32)
                        * 0.0001f32
                        * Vector3::new(1.0, 1.0, 1.0),
                )
                .build();
    }

    if game_state.mouse_button_pressed {
        let camera_position = game_state.camera_controller.current_pose.position;
        let direction_vec = game_state
            .camera_controller
            .current_pose
            .get_direction_vector();
        let ray = Ray::new(
            point![camera_position.x, camera_position.y, camera_position.z],
            vector![direction_vec.x, direction_vec.y, direction_vec.z],
        );
        let max_distance = ARENA_SIDE_LENGTH * 10.0;
        let solid = true;
        if let Some((collider_handle, collision_point_distance)) =
            game_state.physics_state.query_pipeline.cast_ray(
                &game_state.physics_state.collider_set,
                &ray,
                max_distance,
                solid,
                InteractionGroups::all(),
                None,
            )
        {
            // The first collider hit has the handle `handle` and it hit after
            // the ray travelled a distance equal to `ray.dir * toi`.
            let hit_point = ray.point_at(collision_point_distance); // Same as: `ray.origin + ray.dir * toi`
            logger.log(&format!(
                "Collider {:?} hit at point {}",
                collider_handle, hit_point
            ));
            if let Some(rigid_body_handle) = game_state
                .physics_state
                .collider_set
                .get(collider_handle)
                .unwrap()
                .parent()
            {
                if let Some((ball_index, ball)) = game_state
                    .physics_balls
                    .iter()
                    .enumerate()
                    .find(|(_, ball)| ball.rigid_body_handle() == rigid_body_handle)
                {
                    logger.log(&format!(
                        "Hit physics ball {:?} hit at point {}",
                        ball_index, hit_point
                    ));
                    ball.destroy(&mut game_state.scene, &mut game_state.physics_state);
                    game_state.physics_balls.remove(ball_index);
                }
            }
        }
    }
    game_state.mouse_button_pressed = false;
}

pub fn init_scene(
    base_renderer_state: &mut BaseRendererState,
    logger: &mut Logger,
) -> Result<(GameScene, RenderBuffers)> {
    let (document, buffers, images) = gltf::import(get_gltf_path())?;
    validate_animation_property_counts(&document, logger);
    build_scene(base_renderer_state, (&document, &buffers, &images))
}
