use super::*;

use anyhow::Result;
use cgmath::{Deg, Rad, Vector3, Vector4};
use rapier3d::prelude::*;
use winit::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};

pub const INITIAL_RENDER_SCALE: f32 = 1.0;
pub const INITIAL_TONE_MAPPING_EXPOSURE: f32 = 0.3;
pub const INITIAL_BLOOM_THRESHOLD: f32 = 0.8;
pub const INITIAL_BLOOM_RAMP_SIZE: f32 = 0.2;
pub const ARENA_SIDE_LENGTH: f32 = 500.0;
// pub const LIGHT_COLOR_A: Vector3<f32> = Vector3::new(0.996, 0.973, 0.663);
// pub const LIGHT_COLOR_B: Vector3<f32> = Vector3::new(0.25, 0.973, 0.663);

// linear colors, not srgb
pub const DIRECTIONAL_LIGHT_COLOR_A: Vector3<f32> = Vector3::new(0.84922975, 0.81581426, 0.8832506);
pub const DIRECTIONAL_LIGHT_COLOR_B: Vector3<f32> = Vector3::new(0.81115574, 0.77142686, 0.8088144);
pub const POINT_LIGHT_COLOR: Vector3<f32> = Vector3::new(0.93126976, 0.7402633, 0.49407062);
// pub const LIGHT_COLOR_C: Vector3<f32> =
//     Vector3::new(from_srgb(0.631), from_srgb(0.565), from_srgb(0.627));

pub const COLLISION_GROUP_PLAYER_UNSHOOTABLE: Group = Group::GROUP_1;

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
    let _skybox_background = SkyboxBackground::Equirectangular {
        image_path: "./src/textures/newport_loft/background.jpg",
    };
    let _skybox_hdr_environment: Option<SkyboxHDREnvironment> =
        Some(SkyboxHDREnvironment::Equirectangular {
            image_path: "./src/textures/newport_loft/radiance.hdr",
        });

    // Milkyway
    // src: http://www.hdrlabs.com/sibl/archive/
    let skybox_background = SkyboxBackground::Equirectangular {
        image_path: "./src/textures/milkyway/background.jpg",
    };
    let skybox_hdr_environment: Option<SkyboxHDREnvironment> =
        Some(SkyboxHDREnvironment::Equirectangular {
            image_path: "./src/textures/milkyway/radiance.hdr",
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

pub fn init_game_state(mut scene: Scene, renderer_state: &mut RendererState) -> Result<GameState> {
    let mut physics_state = PhysicsState::new();

    // create player
    let player_node_id = scene.add_node(GameNodeDesc::default()).id();
    let player_controller = PlayerController::new(
        &mut physics_state,
        6.0,
        Vector3::new(8.0, 30.0, -13.0),
        ControlledViewDirection {
            horizontal: Deg(180.0).into(),
            vertical: Rad(0.0),
        },
    );

    // load in gltf files

    // player's revolver
    #[allow(unused_assignments)]
    let mut revolver: Option<Revolver> = None;
    {
        // or ./src/models/gltf/Revolver/revolver_low_poly.gltf
        let (document, buffers, images) =
            gltf::import("./src/models/gltf/ColtPython/colt_python.gltf")?;
        let (other_scene, other_render_buffers) =
            build_scene(&mut renderer_state.base, (&document, &buffers, &images))?;
        scene.merge_scene(renderer_state, other_scene, other_render_buffers);

        let node_id = scene.nodes().last().unwrap().id();
        let animation_index = scene.animations.len() - 1;
        // revolver_indices = Some((revolver_model_node_id, animation_index));
        revolver = Some(Revolver::new(
            &mut scene,
            player_node_id,
            node_id,
            animation_index,
            // revolver model
            // TransformBuilder::new()
            //     .position(Vector3::new(0.21, -0.09, -1.0))
            //     .rotation(make_quat_from_axis_angle(
            //         Vector3::new(0.0, 1.0, 0.0),
            //         Deg(180.0).into(),
            //     ))
            //     .scale(0.17f32 * Vector3::new(1.0, 1.0, 1.0))
            //     .build(),
            // colt python model
            TransformBuilder::new()
                .position(Vector3::new(0.21, -0.13, -1.0))
                .rotation(
                    make_quat_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Deg(180.0).into())
                        * make_quat_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(0.1)),
                )
                .scale(2.0f32 * Vector3::new(1.0, 1.0, 1.0))
                .build(),
        ));
    }

    // forest
    {
        let (document, buffers, images) =
            gltf::import("./src/models/gltf/free_low_poly_forest/scene.gltf")?;
        let (mut other_scene, other_render_buffers) =
            build_scene(&mut renderer_state.base, (&document, &buffers, &images))?;
        // hack to get the terrain to be at the same height as the ground.
        let node_has_parent: Vec<_> = other_scene
            .nodes()
            .map(|node| other_scene.get_node_parent(node.id()).is_some())
            .collect();
        for (i, node) in other_scene.nodes_mut().enumerate() {
            if node_has_parent[i] {
                continue;
            }
            node.transform
                .set_position(node.transform.position() + Vector3::new(0.0, 29.0, 0.0));
        }
        scene.merge_scene(renderer_state, other_scene, other_render_buffers);
    }

    // robot
    // https://www.cgtrader.com/free-3d-models/character/sci-fi-character/legendary-robot-free-low-poly-3d-model
    {
        let (document, buffers, images) =
            gltf::import("./src/models/gltf/LegendaryRobot/Legendary_Robot.gltf")?;
        let (mut other_scene, other_render_buffers) =
            build_scene(&mut renderer_state.base, (&document, &buffers, &images))?;
        if let Some(jump_up_animation) = other_scene
            .animations
            .iter_mut()
            .find(|animation| animation.name == Some(String::from("jump_up_root_motion")))
        {
            jump_up_animation.speed = 0.25;
            jump_up_animation.state.is_playing = true;
            jump_up_animation.state.loop_type = LoopType::Wrap;
        }
        scene.merge_scene(renderer_state, other_scene, other_render_buffers);
    }

    // maze
    {
        let skip_nodes = scene.node_count();
        let (document, buffers, images) =
            gltf::import("./src/models/gltf/TestLevel/test_level.gltf")?;
        let (other_scene, other_render_buffers) =
            build_scene(&mut renderer_state.base, (&document, &buffers, &images))?;
        scene.merge_scene(renderer_state, other_scene, other_render_buffers);

        let test_level_node_ids: Vec<_> = scene
            .nodes()
            .skip(skip_nodes)
            .map(|node| node.id())
            .collect();
        for node_id in test_level_node_ids {
            if let Some(_mesh) = scene.get_node_mut(node_id).unwrap().mesh.as_mut() {
                // _mesh.wireframe = true;
            }
            physics_state.add_static_box(&scene, renderer_state, node_id);
        }
    }

    // other
    {
        // let gltf_path = "/home/david/Downloads/adamHead/adamHead.gltf";
        // let gltf_path = "./src/models/gltf/free_low_poly_forest/scene.gltf";
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
        // let gltf_path = "../glTF-Sample-Models/2.0/CesiumMan/glTF/CesiumMan.gltf";
        // let gltf_path = "../glTF-Sample-Models/2.0/Fox/glTF/Fox.gltf";
        // let gltf_path = "../glTF-Sample-Models/2.0/RecursiveSkeletons/glTF/RecursiveSkeletons.gltf";
        // let gltf_path = "../glTF-Sample-Models/2.0/BrainStem/glTF/BrainStem.gltf";
        // let gltf_path =
        //     "/home/david/Programming/glTF-Sample-Models/2.0/BoxAnimated/glTF/BoxAnimated.gltf";
        // let gltf_path = "/home/david/Programming/glTF-Sample-Models/2.0/Lantern/glTF/Lantern.gltf";
        // let gltf_path = "./src/models/gltf/VC/VC.gltf";
        // let gltf_path =
        //     "../glTF-Sample-Models-master/2.0/InterpolationTest/glTF/InterpolationTest.gltf";
        // let (document, buffers, images) = gltf::import(gltf_path)?;
        // let (mut other_scene, other_render_buffers) =
        //     build_scene(&mut renderer_state.base, (&document, &buffers, &images))?;
        // for animation in other_scene.animations.iter_mut() {
        //     animation.state.is_playing = true;
        //     animation.state.loop_type = LoopType::Wrap;
        // }
        // scene.merge_scene(renderer_state, other_scene, other_render_buffers);
    }

    let sphere_mesh = BasicMesh::new("./src/models/sphere.obj")?;
    let plane_mesh = BasicMesh::new("./src/models/plane.obj")?;
    let cube_mesh = BasicMesh::new("./src/models/cube.obj")?;

    // add lights to the scene
    let directional_lights = vec![
        DirectionalLightComponent {
            position: Vector3::new(1.0, 5.0, -10.0) * 10.0,
            direction: (-Vector3::new(1.0, 5.0, -10.0)).normalize(),
            color: DIRECTIONAL_LIGHT_COLOR_A,
            intensity: 1.0,
        },
        DirectionalLightComponent {
            position: Vector3::new(-1.0, 10.0, 10.0) * 10.0,
            direction: (-Vector3::new(-1.0, 10.0, 10.0)).normalize(),
            color: DIRECTIONAL_LIGHT_COLOR_B,
            intensity: 1.0,
        },
    ];
    // let directional_lights: Vec<DirectionalLightComponent> = vec![];

    let point_lights: Vec<(transform::Transform, Vector3<f32>, f32)> = vec![
        (
            TransformBuilder::new()
                .scale(Vector3::new(0.05, 0.05, 0.05))
                .position(Vector3::new(0.0, 12.0, 0.0))
                .build(),
            POINT_LIGHT_COLOR,
            1.0,
        ),
        // (
        //     TransformBuilder::new()
        //         .scale(Vector3::new(0.1, 0.1, 0.1))
        //         .position(Vector3::new(0.0, 15.0, 0.0))
        //         .build(),
        //     LIGHT_COLOR_B,
        //     1.0,
        // ),
    ];
    // let point_lights: Vec<(transform::Transform, Vector3<f32>, f32)> = vec![];

    let point_light_unlit_mesh_index = renderer_state.bind_basic_unlit_mesh(&sphere_mesh);
    let mut point_light_node_ids: Vec<GameNodeId> = Vec::new();
    let mut point_light_components: Vec<PointLightComponent> = Vec::new();
    for (transform, color, intensity) in point_lights {
        let node_id = scene
            .add_node(
                GameNodeDescBuilder::new()
                    .mesh(Some(GameNodeMesh {
                        mesh_indices: vec![point_light_unlit_mesh_index],
                        mesh_type: GameNodeMeshType::Unlit {
                            color: color * intensity,
                        },
                        ..Default::default()
                    }))
                    .transform(transform)
                    .build(),
            )
            .id();
        point_light_node_ids.push(node_id);
        point_light_components.push(PointLightComponent {
            node_id,
            color: POINT_LIGHT_COLOR,
            intensity,
        });
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
                .mesh(Some(GameNodeMesh::from_pbr_mesh_index(
                    test_object_pbr_mesh_index,
                )))
                .transform(
                    TransformBuilder::new()
                        .position(Vector3::new(4.0, 10.0, 4.0))
                        // .scale(Vector3::new(0.0, 0.0, 0.0))
                        .build(),
                )
                .build(),
        )
        .id();
    scene.remove_node(test_object_node_id);

    let legendary_robot_root_node_id = scene
        .nodes()
        .find(|node| node.name == Some(String::from("robot")))
        .map(|legendary_robot_root_node| legendary_robot_root_node.id());

    let legendary_robot = legendary_robot_root_node_id.map(|legendary_robot_root_node_id| {
        scene
            .get_node_mut(legendary_robot_root_node_id)
            .unwrap()
            .transform
            .set_position(Vector3::new(2.0, 0.0, 0.0));

        let legendary_robot_skin_index = 0;
        Character::new(
            &mut scene,
            &mut physics_state,
            renderer_state,
            legendary_robot_root_node_id,
            legendary_robot_skin_index,
            &cube_mesh,
        )
    });
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
                .mesh(Some(GameNodeMesh::from_pbr_mesh_index(ball_pbr_mesh_index)))
                .transform(ball.transform)
                .build(),
        );
        ball_node_ids.push(node.id());
    }

    let physics_ball_count = 500;
    let physics_balls: Vec<_> = (0..physics_ball_count)
        .into_iter()
        .map(|_| {
            PhysicsBall::new_random(
                &mut scene,
                &mut physics_state,
                GameNodeMesh::from_pbr_mesh_index(ball_pbr_mesh_index),
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
    let floor_transform = TransformBuilder::new()
        .scale(Vector3::new(ARENA_SIDE_LENGTH, 1.0, ARENA_SIDE_LENGTH))
        .build();
    let _floor_node = scene.add_node(
        GameNodeDescBuilder::new()
            .mesh(Some(GameNodeMesh::from_pbr_mesh_index(
                floor_pbr_mesh_index,
            )))
            .transform(floor_transform)
            .build(),
    );
    let floor_thickness = 0.1;
    let floor_collider = ColliderBuilder::cuboid(
        floor_transform.scale().x,
        floor_thickness / 2.0,
        floor_transform.scale().z,
    )
    .translation(vector![
        floor_transform.position().x / 2.0,
        floor_transform.position().y - floor_thickness / 2.0,
        floor_transform.position().z / 2.0
    ])
    .collision_groups(
        InteractionGroups::all().with_memberships(!COLLISION_GROUP_PLAYER_UNSHOOTABLE),
    )
    .friction(1.0)
    .restitution(1.0)
    .build();
    physics_state.collider_set.insert(floor_collider);

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
                .mesh(Some(GameNodeMesh::from_pbr_mesh_index(
                    bouncing_ball_pbr_mesh_index,
                )))
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
            .collision_groups(
                InteractionGroups::all().with_memberships(!COLLISION_GROUP_PLAYER_UNSHOOTABLE),
            )
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
    scene.remove_node(bouncing_ball_node_id);

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
    #[allow(unused_assignments)]
    let mut crosshair_node_id: Option<GameNodeId> = None;
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
    let crosshair_mesh_index = renderer_state.bind_basic_pbr_mesh(
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
    crosshair_node_id = Some(
        scene
            .add_node(
                GameNodeDescBuilder::new()
                    .mesh(Some(GameNodeMesh {
                        mesh_indices: vec![crosshair_mesh_index],
                        mesh_type: GameNodeMeshType::Pbr {
                            material_override: Some(DynamicPbrParams {
                                emissive_factor: crosshair_color,
                                base_color_factor: Vector4::new(0.0, 0.0, 0.0, 1.0),
                                alpha_cutoff: 0.5,
                                ..Default::default()
                            }),
                        },
                        wireframe: false,
                        ..Default::default()
                    }))
                    .build(),
            )
            .id(),
    );

    let mut audio_manager = AudioManager::new()?;

    let bgm_data =
        AudioManager::decode_mp3(audio_manager.device_sample_rate(), "./src/sounds/bgm.mp3")?;
    let bgm_sound_index = audio_manager.add_sound(&bgm_data, 0.5, false, None);
    audio_manager.play_sound(bgm_sound_index);

    let gunshot_sound_data = AudioManager::decode_wav(
        audio_manager.device_sample_rate(),
        "./src/sounds/gunshot.wav",
    )?;
    let gunshot_sound_index = audio_manager.add_sound(&gunshot_sound_data, 0.75, true, None);

    // logger_log(&format!("{:?}", &revolver));

    Ok(GameState {
        scene,
        time_tracker: None,
        state_update_time_accumulator: 0.0,
        is_playing_animations: true,

        audio_manager: Some(audio_manager),
        bgm_sound_index,
        gunshot_sound_index,
        gunshot_sound_data,

        player_node_id,

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
        revolver,

        bouncing_ball_node_id,
        bouncing_ball_body_handle,

        physics_state,

        physics_balls,
        mouse_button_pressed: false,

        character: legendary_robot,
        player_controller,
    })
}

pub fn process_device_input(game_state: &mut GameState, event: &winit::event::DeviceEvent) {
    game_state.player_controller.process_device_events(event);
}

pub fn increment_render_scale(renderer_state: &mut RendererState, increase: bool) {
    let delta = 0.1;
    let change = if increase { delta } else { -delta };
    renderer_state.render_scale = (renderer_state.render_scale + change).clamp(0.1, 4.0);
    logger_log(&format!(
        "Render scale: {:?} ({:?}x{:?})",
        renderer_state.render_scale,
        (renderer_state.base.surface_config.width as f32 * renderer_state.render_scale.sqrt())
            .round() as u32,
        (renderer_state.base.surface_config.height as f32 * renderer_state.render_scale.sqrt())
            .round() as u32,
    ));
    renderer_state.resize(renderer_state.base.window_size);
}

pub fn increment_exposure(renderer_state: &mut RendererState, increase: bool) {
    let delta = 0.05;
    let change = if increase { delta } else { -delta };
    renderer_state.tone_mapping_exposure =
        (renderer_state.tone_mapping_exposure + change).clamp(0.0, 20.0);
    logger_log(&format!(
        "Exposure: {:?}",
        renderer_state.tone_mapping_exposure
    ));
}

pub fn increment_bloom_threshold(renderer_state: &mut RendererState, increase: bool) {
    let delta = 0.05;
    let change = if increase { delta } else { -delta };
    renderer_state.bloom_threshold = (renderer_state.bloom_threshold + change).clamp(0.0, 20.0);
    logger_log(&format!(
        "Bloom Threshold: {:?}",
        renderer_state.bloom_threshold
    ));
}

pub fn process_window_input(
    game_state: &mut GameState,
    renderer_state: &mut RendererState,
    event: &winit::event::WindowEvent,
    window: &mut winit::window::Window,
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
                    increment_render_scale(renderer_state, false);
                }
                VirtualKeyCode::X => {
                    increment_render_scale(renderer_state, true);
                }
                VirtualKeyCode::E => {
                    increment_exposure(renderer_state, false);
                }
                VirtualKeyCode::R => {
                    increment_exposure(renderer_state, true);
                }
                VirtualKeyCode::T => {
                    increment_bloom_threshold(renderer_state, false);
                }
                VirtualKeyCode::Y => {
                    increment_bloom_threshold(renderer_state, true);
                }
                VirtualKeyCode::P => {
                    game_state.is_playing_animations = !game_state.is_playing_animations;
                }
                VirtualKeyCode::M => {
                    renderer_state.enable_shadows = !renderer_state.enable_shadows;
                }
                VirtualKeyCode::B => {
                    renderer_state.enable_bloom = !renderer_state.enable_bloom;
                }
                VirtualKeyCode::F => {
                    renderer_state.enable_wireframe_mode = !renderer_state.enable_wireframe_mode;
                }
                VirtualKeyCode::J => {
                    renderer_state.draw_node_bounding_spheres =
                        !renderer_state.draw_node_bounding_spheres;
                }
                VirtualKeyCode::C => {
                    if let Some(character) = game_state.character.as_mut() {
                        character.toggle_collision_box_display(&mut game_state.scene);
                    }
                }
                _ => {}
            }
        }
    }
    game_state
        .player_controller
        .process_window_events(event, window);
}

#[profiling::function]
pub fn update_game_state(game_state: &mut GameState, renderer_state: &RendererState) {
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

    game_state
        .player_controller
        .update(&mut game_state.physics_state);
    // logger_log(&format!(
    //     "camera pose: {:?}",
    //     game_state.camera_controller.current_pose
    // ));

    let new_player_transform = game_state
        .player_controller
        .transform(&game_state.physics_state);
    if let Some(player_transform) = game_state.scene.get_node_mut(game_state.player_node_id) {
        player_transform.transform = new_player_transform;
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
            .for_each(|ball| ball.update(min_update_timestep_seconds));
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
        // point_light_0.color = lerp_vec(
        //     LIGHT_COLOR_A,
        //     LIGHT_COLOR_B,
        //     (global_time_seconds * 2.0).sin(),
        // );
        if let Some(node) = game_state.scene.get_node_mut(point_light_0.node_id) {
            node.transform.set_position(Vector3::new(
                1.5 * (global_time_seconds * 0.25 + std::f32::consts::PI).cos(),
                node.transform.position().y - frame_time_seconds * 0.25,
                1.5 * (global_time_seconds * 0.25 + std::f32::consts::PI).sin(),
            ));
        }
    }

    if let Some(_point_light_1) = game_state.point_lights.get_mut(1) {
        // _point_light_1.color = lerp_vec(
        //     LIGHT_COLOR_B,
        //     LIGHT_COLOR_A,
        //     (global_time_seconds * 2.0).sin(),
        // );
    }

    // sync unlit mesh config with point light component
    game_state
        .point_light_node_ids
        .iter()
        .zip(game_state.point_lights.iter())
        .for_each(|(node_id, point_light)| {
            if let Some(GameNodeMesh {
                mesh_type: GameNodeMeshType::Unlit { ref mut color },
                ..
            }) = game_state
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

    // logger_log(&format!("Frame time: {:?}", frame_time_seconds));
    // logger_log(&format!(
    //     "state_update_time_accumulator: {:?}",
    //     game_state.state_update_time_accumulator
    // ));

    // remove physics balls over time
    game_state.ball_spawner_acc += frame_time_seconds;
    let rate = 0.1; // lower value spawns balls more quickly
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
        // game_state.physics_balls.push(PhysicsBall::new_random(
        //     &mut game_state.scene,
        //     &mut game_state.physics_state,
        //     GameNodeMesh::from_pbr_mesh_index(game_state.ball_pbr_mesh_index),
        // ));
        game_state.ball_spawner_acc -= rate;
    }
    let new_ball_count = game_state.physics_balls.len();
    if prev_ball_count != new_ball_count {
        // logger_log(&format!("Ball count: {:?}", new_ball_count));
    }

    // let physics_time_step_start = Instant::now();

    // logger_log(&format!("Physics step time: {:?}", physics_time_step_start.elapsed()));
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

    if let Some(crosshair_node) = game_state
        .crosshair_node_id
        .and_then(|crosshair_node_id| game_state.scene.get_node_mut(crosshair_node_id))
    {
        crosshair_node.transform = new_player_transform
            * TransformBuilder::new()
                .position(Vector3::new(0.0, 0.0, -1.0))
                .rotation(make_quat_from_axis_angle(
                    Vector3::new(0.0, 1.0, 0.0),
                    Deg(90.0).into(),
                ))
                .scale(
                    (1080.0 / renderer_state.base.window_size.height as f32)
                        * 0.06
                        * Vector3::new(1.0, 1.0, 1.0),
                )
                .build();
    }

    if let Some(revolver) = game_state.revolver.as_mut() {
        revolver.update(
            game_state.player_controller.view_direction,
            &mut game_state.scene,
        );

        if game_state.mouse_button_pressed && revolver.fire(&mut game_state.scene) {
            if let Some(audio_manager) = game_state.audio_manager.as_mut() {
                audio_manager.play_sound(game_state.gunshot_sound_index)
            }
            // setting gunshot_sound_index to 0 is a hacky way to deal with the audio_manager not being initialized
            // such as when we dont want to play audio
            game_state.gunshot_sound_index = game_state
                .audio_manager
                .as_mut()
                .map(|audio_manager| {
                    audio_manager.add_sound(&game_state.gunshot_sound_data, 0.75, true, None)
                })
                .unwrap_or(0);

            // logger_log("Fired!");
            let player_position = game_state
                .player_controller
                .position(&game_state.physics_state);
            let direction_vec = game_state
                .player_controller
                .view_direction
                .to_direction_vector();
            let ray = Ray::new(
                point![player_position.x, player_position.y, player_position.z],
                vector![direction_vec.x, direction_vec.y, direction_vec.z],
            );
            let max_distance = ARENA_SIDE_LENGTH * 10.0;
            let solid = true;
            if let Some((collider_handle, collision_point_distance)) =
                game_state.physics_state.query_pipeline.cast_ray(
                    &game_state.physics_state.rigid_body_set,
                    &game_state.physics_state.collider_set,
                    &ray,
                    max_distance,
                    solid,
                    QueryFilter::from(
                        InteractionGroups::all().with_filter(!COLLISION_GROUP_PLAYER_UNSHOOTABLE),
                    ),
                )
            {
                // The first collider hit has the handle `handle` and it hit after
                // the ray travelled a distance equal to `ray.dir * toi`.
                let _hit_point = ray.point_at(collision_point_distance); // Same as: `ray.origin + ray.dir * toi`

                // logger_log(&format!(
                //     "Collider {:?} hit at point {}",
                //     collider_handle, _hit_point
                // ));
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
                        // logger_log(&format!(
                        //     "Hit physics ball {:?} hit at point {}",
                        //     ball_index, hit_point
                        // ));
                        // ball.toggle_wireframe(&mut game_state.scene);
                        ball.destroy(&mut game_state.scene, &mut game_state.physics_state);
                        game_state.physics_balls.remove(ball_index);
                    }
                }
                if let Some(character) = game_state.character.as_mut() {
                    character.handle_hit(&mut game_state.scene, collider_handle);
                }
            }
        }
    }

    // step animatons
    let scene = &mut game_state.scene;
    if game_state.is_playing_animations {
        step_animations(scene, frame_time_seconds)
    }

    if let Some(character) = game_state.character.as_mut() {
        character.update(scene, &mut game_state.physics_state);
    }
}
