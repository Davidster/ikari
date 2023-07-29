use crate::animation::*;
use crate::asset_loader::*;
use crate::audio::*;
use crate::ball::*;
use crate::character::*;
use crate::game_state::*;
use crate::light::*;
use crate::math::*;
use crate::mesh::*;
use crate::physics::*;
use crate::physics_ball::*;
use crate::player_controller::*;
use crate::renderer::*;
use crate::revolver::*;
use crate::sampler_cache::*;
use crate::scene::*;
use crate::texture::*;
#[cfg(not(target_arch = "wasm32"))]
use crate::texture_compression::*;
use crate::transform::*;

use std::{
    collections::hash_map::Entry,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use glam::f32::{Vec3, Vec4};
use rapier3d::prelude::*;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

pub const INITIAL_ENABLE_VSYNC: bool = true;
pub const INITIAL_RENDER_SCALE: f32 = 1.0;
pub const INITIAL_TONE_MAPPING_EXPOSURE: f32 = 1.0;
pub const INITIAL_BLOOM_THRESHOLD: f32 = 0.8;
pub const INITIAL_BLOOM_RAMP_SIZE: f32 = 0.2;
pub const ARENA_SIDE_LENGTH: f32 = 500.0;
pub const INITIAL_IS_SHOWING_CAMERA_POSE: bool = true;
pub const INITIAL_ENABLE_SHADOW_DEBUG: bool = false;
pub const INITIAL_ENABLE_CULLING_FRUSTUM_DEBUG: bool = false;
pub const INITIAL_ENABLE_POINT_LIGHT_CULLING_FRUSTUM_DEBUG: bool = false;
pub const INITIAL_ENABLE_SOFT_SHADOWS: bool = true;
pub const INITIAL_SHADOW_BIAS: f32 = 0.0005;
pub const INITIAL_SOFT_SHADOW_FACTOR: f32 = 0.0015;
pub const INITIAL_SOFT_SHADOW_GRID_DIMS: u32 = 4;

pub const CREATE_POINT_SHADOW_MAP_DEBUG_OBJECTS: bool = false;

// pub const LIGHT_COLOR_A: Vec3 = Vec3::new(0.996, 0.973, 0.663);
// pub const LIGHT_COLOR_B: Vec3 = Vec3::new(0.25, 0.973, 0.663);

// linear colors, not srgb
pub const DIRECTIONAL_LIGHT_COLOR_A: Vec3 = Vec3::new(0.84922975, 0.81581426, 0.8832506);
pub const DIRECTIONAL_LIGHT_COLOR_B: Vec3 = Vec3::new(0.81115574, 0.77142686, 0.8088144);
pub const POINT_LIGHT_COLOR: Vec3 = Vec3::new(0.93126976, 0.7402633, 0.49407062);
// pub const LIGHT_COLOR_C: Vec3 =
//     Vec3::new(from_srgb(0.631), from_srgb(0.565), from_srgb(0.627));

pub const COLLISION_GROUP_PLAYER_UNSHOOTABLE: Group = Group::GROUP_1;

pub fn get_skybox_path() -> (
    SkyboxBackgroundPath<'static>,
    Option<SkyboxHDREnvironmentPath<'static>>,
) {
    // Mountains
    // src: https://github.com/JoeyDeVries/LearnOpenGL/tree/master/resources/textures/skybox
    let _skybox_background = SkyboxBackgroundPath::Cube {
        face_image_paths: [
            "src/textures/skybox/right.jpg",
            "src/textures/skybox/left.jpg",
            "src/textures/skybox/top.jpg",
            "src/textures/skybox/bottom.jpg",
            "src/textures/skybox/front.jpg",
            "src/textures/skybox/back.jpg",
        ],
    };
    let _skybox_hdr_environment: Option<SkyboxHDREnvironmentPath> = None;

    // Newport Loft
    // src: http://www.hdrlabs.com/sibl/archive/
    let _skybox_background = SkyboxBackgroundPath::Equirectangular {
        image_path: "src/textures/newport_loft/background.jpg",
    };
    let _skybox_hdr_environment: Option<SkyboxHDREnvironmentPath> =
        Some(SkyboxHDREnvironmentPath::Equirectangular {
            image_path: "src/textures/newport_loft/radiance.hdr",
        });

    // Milkyway
    // src: http://www.hdrlabs.com/sibl/archive/
    let skybox_background = SkyboxBackgroundPath::Equirectangular {
        image_path: "src/textures/milkyway/background.jpg",
    };
    let skybox_hdr_environment: Option<SkyboxHDREnvironmentPath> =
        Some(SkyboxHDREnvironmentPath::Equirectangular {
            image_path: "src/textures/milkyway/radiance.hdr",
        });

    // My photosphere pic
    // src: me
    let _skybox_background = SkyboxBackgroundPath::Equirectangular {
        image_path: "src/textures/photosphere_skybox_small.jpg",
    };
    let _skybox_hdr_environment: Option<SkyboxHDREnvironmentPath> = None;

    (skybox_background, skybox_hdr_environment)
}

fn get_misc_gltf_path() -> &'static str {
    // "/home/david/Downloads/adamHead/adamHead.gltf"
    // "src/models/gltf/free_low_poly_forest/scene.gltf"
    // "src/models/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf"
    // "src/models/gltf/SimpleMeshes/SimpleMeshes.gltf"
    // "src/models/gltf/Triangle/Triangle.gltf"
    // "src/models/gltf/TriangleWithoutIndices/TriangleWithoutIndices.gltf"
    // "src/models/gltf/EnvironmentTest/EnvironmentTest.gltf"
    // "src/models/gltf/Arrow/Arrow.gltf"
    "src/models/gltf/DamagedHelmet/DamagedHelmet.gltf"
    // "src/models/gltf/VertexColorTest/VertexColorTest.gltf"
    // "src/models/gltf/Revolver/revolver_low_poly.gltf"
    // "src/models/gltf/NormalTangentMirrorTest/NormalTangentMirrorTest.gltf"
    // "src/models/gltf/TextureLinearInterpolationTest/TextureLinearInterpolationTest.glb"
    // "../glTF-Sample-Models/2.0/RiggedFigure/glTF/RiggedFigure.gltf"
    // "../glTF-Sample-Models/2.0/RiggedSimple/glTF/RiggedSimple.gltf"
    // "../glTF-Sample-Models/2.0/CesiumMan/glTF/CesiumMan.gltf"
    // "../glTF-Sample-Models/2.0/Fox/glTF/Fox.gltf"
    // "../glTF-Sample-Models/2.0/RecursiveSkeletons/glTF/RecursiveSkeletons.gltf"
    // "../glTF-Sample-Models/2.0/BrainStem/glTF/BrainStem.gltf"
    // "/home/david/Programming/glTF-Sample-Models/2.0/BoxAnimated/glTF/BoxAnimated.gltf"
    // "/home/david/Programming/glTF-Sample-Models/2.0/Lantern/glTF/Lantern.gltf"
    // "src/models/gltf/VC/VC.gltf"
    //  "../glTF-Sample-Models-master/2.0/InterpolationTest/glTF/InterpolationTest.gltf"
    // "src/models/gltf/Sponza/Sponza.gltf"
}

#[cfg(not(target_arch = "wasm32"))]
async fn get_rainbow_texture(renderer_base: &BaseRenderer) -> Result<Texture> {
    let texture_compressor = TextureCompressor::new();
    let rainbow_texture_path = "src/textures/rainbow_gradient_vertical_compressed.bin";
    let rainbow_texture_bytes = crate::file_loader::read(rainbow_texture_path).await?;
    let rainbow_texture_decompressed =
        texture_compressor.transcode_image(&rainbow_texture_bytes, false)?;
    Texture::from_decoded_image(
        renderer_base,
        &rainbow_texture_decompressed.raw,
        (
            rainbow_texture_decompressed.width,
            rainbow_texture_decompressed.height,
        ),
        rainbow_texture_decompressed.mip_count,
        Some(rainbow_texture_path),
        Some(wgpu::TextureFormat::Bc7RgbaUnormSrgb),
        false,
        &Default::default(),
    )
}

#[cfg(target_arch = "wasm32")]
async fn get_rainbow_texture(renderer_base: &BaseRenderer) -> Result<Texture> {
    let rainbow_texture_path = "src/textures/rainbow_gradient_vertical.jpg";
    let rainbow_texture_bytes = crate::file_loader::read(rainbow_texture_path).await?;
    Texture::from_encoded_image(
        renderer_base,
        &rainbow_texture_bytes,
        Some(rainbow_texture_path),
        wgpu::TextureFormat::Rgba8Unorm.into(),
        false,
        &Default::default(),
    )
}

pub async fn init_game_state(mut scene: Scene, renderer: &mut Renderer) -> Result<GameState> {
    let mut physics_state = PhysicsState::new();

    // create player
    let player_node_id = scene.add_node(GameNodeDesc::default()).id();
    let player_controller = PlayerController::new(
        &mut physics_state,
        6.0,
        Vec3::new(8.0, 30.0, -13.0),
        ControlledViewDirection {
            horizontal: deg_to_rad(180.0),
            vertical: 0.0,
        },
    );

    let (audio_manager, audio_streams) = AudioManager::new()?;

    let audio_manager_mutex = Arc::new(Mutex::new(audio_manager));

    let asset_loader = Arc::new(AssetLoader::new(audio_manager_mutex.clone()));

    let asset_loader_clone = asset_loader.clone();

    crate::thread::spawn(move || {
        crate::block_on(async move {
            // crate::thread::sleep_async(crate::time::Duration::from_secs_f32(5.0)).await;

            // load in gltf files

            // player's revolver
            // https://done3d.com/colt-python-8-inch/
            asset_loader.load_gltf_scene("src/models/gltf/ColtPython/colt_python.glb");
            // forest
            // https://sketchfab.com/3d-models/free-low-poly-forest-6dc8c85121234cb59dbd53a673fa2b8f
            asset_loader.load_gltf_scene("src/models/gltf/free_low_poly_forest/scene.glb");
            // legendary robot
            // https://www.cgtrader.com/free-3d-models/character/sci-fi-character/legendary-robot-free-low-poly-3d-model
            asset_loader.load_gltf_scene("src/models/gltf/LegendaryRobot/Legendary_Robot.glb");
            // maze
            asset_loader.load_gltf_scene("src/models/gltf/TestLevel/test_level.glb");
            // other
            // asset_loader.load_gltf_scene(get_misc_gltf_path());

            asset_loader.load_audio(
                "src/sounds/bgm.mp3",
                AudioFileFormat::Mp3,
                SoundParams {
                    initial_volume: 0.3,
                    fixed_volume: false,
                    spacial_params: None,
                    stream: !cfg!(target_arch = "wasm32"),
                },
            );
            asset_loader.load_audio(
                "src/sounds/gunshot.wav",
                AudioFileFormat::Wav,
                SoundParams {
                    initial_volume: 0.4,
                    fixed_volume: true,
                    spacial_params: None,
                    stream: false,
                },
            );

            crate::thread::sleep_async(crate::time::Duration::from_secs_f32(10.0)).await;
            let (background, environment_hdr) = get_skybox_path();
            asset_loader.load_skybox("skybox".to_string(), background, environment_hdr);
        })
    });

    let sphere_mesh = BasicMesh::new("src/models/sphere.obj").await?;
    let plane_mesh = BasicMesh::new("src/models/plane.obj").await?;
    let cube_mesh = BasicMesh::new("src/models/cube.obj").await?;

    // add lights to the scene
    let directional_lights = vec![
        // DirectionalLightComponent {
        //     position: Vec3::new(1.0, 5.0, -10.0) * 10.0,
        //     direction: (-Vec3::new(1.0, 5.0, -10.0)).normalize(),
        //     color: DIRECTIONAL_LIGHT_COLOR_A,
        //     intensity: 1.0,
        // },
        DirectionalLightComponent {
            position: Vec3::new(-1.0, 10.0, 10.0) * 10.0,
            direction: (-Vec3::new(-1.0, 10.0, 10.0)).normalize(),
            color: DIRECTIONAL_LIGHT_COLOR_B,
            intensity: 0.2,
        },
    ];
    // let directional_lights: Vec<DirectionalLightComponent> = vec![];

    let point_lights: Vec<(crate::transform::Transform, Vec3, f32)> = vec![
        (
            TransformBuilder::new()
                .scale(Vec3::new(0.05, 0.05, 0.05))
                .position(Vec3::new(0.0, 0.0, 0.0))
                .build(),
            POINT_LIGHT_COLOR,
            1.0,
        ),
        // (
        //     TransformBuilder::new()
        //         .scale(Vec3::new(0.1, 0.1, 0.1))
        //         .position(Vec3::new(0.0, 15.0, 0.0))
        //         .build(),
        //     DIRECTIONAL_LIGHT_COLOR_B,
        //     1.0,
        // ),
    ];
    // let point_lights: Vec<(crate::transform::Transform, Vec3, f32)> = vec![];

    let point_light_unlit_mesh_index = Renderer::bind_basic_unlit_mesh(
        &renderer.base,
        &mut renderer.data.lock().unwrap(),
        &sphere_mesh,
    );
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

    // let simple_normal_map_path = "src/textures/simple_normal_map.jpg";
    // let simple_normal_map_bytes = crate::file_loader::read(simple_normal_map_path).await?;
    // let simple_normal_map = Texture::from_encoded_image(
    //     &renderer.base.device,
    //     &renderer.base.queue,
    //     &simple_normal_map_bytes,
    //     simple_normal_map_path,
    //     wgpu::TextureFormat::Rgba8Unorm.into(),
    //     false,
    //     &Default::default(),
    // )?;

    let rainbow_texture = get_rainbow_texture(&renderer.base).await?;

    let brick_normal_map_path = "src/textures/brick_normal_map.jpg";
    let brick_normal_map_bytes = crate::file_loader::read(brick_normal_map_path).await?;
    let brick_normal_map = Texture::from_encoded_image(
        &renderer.base,
        &brick_normal_map_bytes,
        Some(brick_normal_map_path),
        wgpu::TextureFormat::Rgba8Unorm.into(),
        false,
        &Default::default(),
    )?;

    // add test object to scene
    /* let earth_texture_path = "src/textures/8k_earth.jpg";
    let earth_texture_bytes = crate::file_loader::read(earth_texture_path).await?;
    let earth_texture = Texture::from_encoded_image(
        &renderer.base,
        &earth_texture_bytes,
        earth_texture_path,
        None,
        true,
        &SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        },
    )?; */

    /* let earth_normal_map_path = "src/textures/8k_earth_normal_map.jpg";
    let earth_normal_map_bytes = crate::file_loader::read(earth_normal_map_path).await?;
    let earth_normal_map = Texture::from_encoded_image(
        &renderer.base,
        &earth_normal_map_bytes,
        earth_normal_map_path,
        wgpu::TextureFormat::Rgba8Unorm.into(),
        false,
        &SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        },
    )?; */

    let checkerboard_texture_img = {
        let mut img = image::RgbaImage::new(1024, 1024);
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
    let checkerboard_texture = Texture::from_decoded_image(
        &renderer.base,
        &checkerboard_texture_img,
        checkerboard_texture_img.dimensions(),
        1,
        USE_LABELS.then_some("checkerboard_texture"),
        None,
        true,
        &SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        },
    )?;

    // add balls to scene

    // source: https://www.solarsystemscope.com/textures/
    /* let mars_texture_path = "src/textures/8k_mars.jpg";
    let mars_texture_bytes = crate::file_loader::read(mars_texture_path).await?;
    let mars_texture = Texture::from_encoded_image(
        &renderer.base,
        &mars_texture_bytes,
        mars_texture_path,
        None,
        true,
        &SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        },
    )?; */

    let test_object_metallic_roughness_map = Texture::from_color(
        &renderer.base,
        [
            255,
            (0.12 * 255.0f32).round() as u8,
            (0.8 * 255.0f32).round() as u8,
            255,
        ],
    )?;

    let test_object_pbr_mesh_index = Renderer::bind_basic_pbr_mesh(
        &renderer.base,
        &mut renderer.data.lock().unwrap(),
        &sphere_mesh,
        &PbrMaterial {
            base_color: Some(&rainbow_texture),
            normal: Some(&brick_normal_map),
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
                        .position(Vec3::new(4.0, 1.5, 3.0))
                        .scale(0.2 * Vec3::new(1.0, 1.0, 1.0))
                        .build(),
                )
                .build(),
        )
        .id();
    // scene.remove_node(test_object_node_id);

    // add floor to scene

    let ball_count = 0;
    let balls: Vec<_> = (0..ball_count).map(|_| BallComponent::rand()).collect();

    let ball_pbr_mesh_index = Renderer::bind_basic_pbr_mesh(
        &renderer.base,
        &mut renderer.data.lock().unwrap(),
        &sphere_mesh,
        &PbrMaterial {
            base_color: Some(&rainbow_texture),
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

    let physics_ball_count = 0;
    let physics_balls: Vec<_> = (0..physics_ball_count)
        .map(|_| {
            PhysicsBall::new_random(
                &mut scene,
                &mut physics_state,
                GameNodeMesh::from_pbr_mesh_index(ball_pbr_mesh_index),
            )
        })
        .collect();

    if CREATE_POINT_SHADOW_MAP_DEBUG_OBJECTS {
        let cube_radius = 4.0;
        let cube_center = Vec3::new(20.0, cube_radius, -4.5);

        let ball_metallic_roughness_map = Texture::from_color(
            &renderer.base,
            [
                255,
                (0.12 * 255.0f32).round() as u8,
                (0.8 * 255.0f32).round() as u8,
                255,
            ],
        )?;
        let ball_pbr_mesh_index = Renderer::bind_basic_pbr_mesh(
            &renderer.base,
            &mut renderer.data.lock().unwrap(),
            &sphere_mesh,
            &PbrMaterial {
                base_color: Some(&rainbow_texture),
                normal: Some(&brick_normal_map),
                metallic_roughness: Some(&ball_metallic_roughness_map),
                ..Default::default()
            },
            Default::default(),
        )?;
        let _ball_node_id = scene
            .add_node(
                GameNodeDescBuilder::new()
                    .mesh(Some(GameNodeMesh::from_pbr_mesh_index(ball_pbr_mesh_index)))
                    .transform(
                        TransformBuilder::new()
                            .position(cube_center - Vec3::new(0.0, 1.0, 0.0))
                            .scale(0.2 * Vec3::new(1.0, 1.0, 1.0))
                            .build(),
                    )
                    .build(),
            )
            .id();

        let wall_mesh_index = Renderer::bind_basic_pbr_mesh(
            &renderer.base,
            &mut renderer.data.lock().unwrap(),
            &plane_mesh,
            &PbrMaterial {
                base_color: Some(&checkerboard_texture),
                ..Default::default()
            },
            Default::default(),
        )?;
        let game_node_mesh = GameNodeMesh::from_pbr_mesh_index(wall_mesh_index);

        let ceiling_transform = TransformBuilder::new()
            .position(cube_center + Vec3::new(0.0, cube_radius, 0.0))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(make_quat_from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                deg_to_rad(180.0),
            ))
            .build();
        let ceiling_game_node_mesh = GameNodeMesh {
            mesh_type: GameNodeMeshType::Pbr {
                material_override: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(1.0, 0.5, 0.5, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_mesh.clone()
        };
        let _ceiling_node = scene.add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(ceiling_game_node_mesh))
                .transform(ceiling_transform)
                .build(),
        );

        let wall_1_node_mesh = GameNodeMesh {
            mesh_type: GameNodeMeshType::Pbr {
                material_override: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(0.5, 1.0, 0.5, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_mesh.clone()
        };
        let wall_transform_1 = TransformBuilder::new()
            .position(cube_center + Vec3::new(0.0, 0.0, -cube_radius))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(make_quat_from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                deg_to_rad(90.0),
            ))
            .build();
        let _wall_1_node = scene.add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(wall_1_node_mesh))
                .transform(wall_transform_1)
                .build(),
        );

        let wall_2_node_mesh = GameNodeMesh {
            mesh_type: GameNodeMeshType::Pbr {
                material_override: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(0.5, 0.5, 1.0, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_mesh.clone()
        };
        let wall_transform_2 = TransformBuilder::new()
            .position(cube_center + Vec3::new(0.0, 0.0, cube_radius))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(make_quat_from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                deg_to_rad(270.0),
            ))
            .build();
        let _wall_2_node = scene.add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(wall_2_node_mesh))
                .transform(wall_transform_2)
                .build(),
        );

        let wall_3_node_mesh = GameNodeMesh {
            mesh_type: GameNodeMeshType::Pbr {
                material_override: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(1.0, 0.5, 1.0, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_mesh.clone()
        };
        let wall_transform_3 = TransformBuilder::new()
            .position(cube_center + Vec3::new(cube_radius, 0.0, 0.0))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(make_quat_from_axis_angle(
                Vec3::new(0.0, 0.0, 1.0),
                deg_to_rad(90.0),
            ))
            .build();
        let _wall_3_node = scene.add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(wall_3_node_mesh))
                .transform(wall_transform_3)
                .build(),
        );

        let wall_4_node_mesh = GameNodeMesh {
            mesh_type: GameNodeMeshType::Pbr {
                material_override: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(1.0, 1.0, 0.5, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_mesh
        };
        let wall_transform_4 = TransformBuilder::new()
            .position(cube_center + Vec3::new(-cube_radius, 0.0, 0.0))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(make_quat_from_axis_angle(
                Vec3::new(0.0, 0.0, 1.0),
                deg_to_rad(270.0),
            ))
            .build();
        let _wall_4_node = scene.add_node(
            GameNodeDescBuilder::new()
                .mesh(Some(wall_4_node_mesh))
                .transform(wall_transform_4)
                .build(),
        );
    }

    // let box_pbr_mesh_index = renderer.bind_basic_pbr_mesh(
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
    //                 .scale(Vec3::new(0.5, 0.5, 0.5))
    //                 .position(Vec3::new(0.0, 0.5, 0.0))
    //                 .build(),
    //         )
    //         .build(),
    // );

    // create the floor and add it to the scene
    let floor_pbr_mesh_index = Renderer::bind_basic_pbr_mesh(
        &renderer.base,
        &mut renderer.data.lock().unwrap(),
        &plane_mesh,
        &PbrMaterial {
            base_color: Some(&checkerboard_texture),
            ..Default::default()
        },
        Default::default(),
    )?;
    let floor_transform = TransformBuilder::new()
        .position(Vec3::new(0.0, -0.01, 0.0))
        .scale(Vec3::new(ARENA_SIDE_LENGTH, 1.0, ARENA_SIDE_LENGTH))
        .build();
    let _floor_node = scene.add_node(
        GameNodeDescBuilder::new()
            .mesh(Some(GameNodeMesh::from_pbr_mesh_index(
                floor_pbr_mesh_index,
            )))
            .transform(floor_transform)
            .build(),
    );
    // let ceiling_transform = TransformBuilder::new()
    //     .position(Vec3::new(0.0, 10.0, 0.0))
    //     .scale(Vec3::new(ARENA_SIDE_LENGTH, 1.0, ARENA_SIDE_LENGTH))
    //     .rotation(make_quat_from_axis_angle(
    //         Vec3::new(1.0, 0.0, 0.0),
    //         deg_to_rad(180.0),
    //     ))
    //     .build();
    // let _ceiling_node = scene.add_node(
    //     GameNodeDescBuilder::new()
    //         .mesh(Some(GameNodeMesh::from_pbr_mesh_index(
    //             floor_pbr_mesh_index,
    //         )))
    //         .transform(ceiling_transform)
    //         .build(),
    // );
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
        let bouncing_ball_pbr_mesh_index = Renderer::bind_basic_pbr_mesh(
            &renderer.base,
            &mut renderer.data.lock().unwrap(),
            &sphere_mesh,
            &PbrMaterial {
                base_color: Some(&checkerboard_texture),
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
                        .scale(Vec3::new(
                            bouncing_ball_radius,
                            bouncing_ball_radius,
                            bouncing_ball_radius,
                        ))
                        .position(Vec3::new(-1.0, 10.0, 0.0))
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
        &renderer.base,
        &crosshair_texture_img,
        crosshair_texture_img.dimensions(),
        1,
        Some("crosshair_texture"),
        None,
        false,
        &SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        },
    )?;
    let crosshair_quad = BasicMesh {
        vertices: [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]]
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
    let crosshair_ambient_occlusion = Texture::from_color(&renderer.base, [0, 0, 0, 0])?;
    let crosshair_metallic_roughness = Texture::from_color(&renderer.base, [0, 0, 255, 0])?;
    let crosshair_mesh_index = Renderer::bind_basic_pbr_mesh(
        &renderer.base,
        &mut renderer.data.lock().unwrap(),
        &crosshair_quad,
        &PbrMaterial {
            ambient_occlusion: Some(&crosshair_ambient_occlusion),
            metallic_roughness: Some(&crosshair_metallic_roughness),
            base_color: Some(&crosshair_texture),
            emissive: Some(&crosshair_texture),
            ..Default::default()
        },
        Default::default(),
    )?;
    let crosshair_color = Vec3::new(1.0, 0.0, 0.0);
    crosshair_node_id = Some(
        scene
            .add_node(
                GameNodeDescBuilder::new()
                    .mesh(Some(GameNodeMesh {
                        mesh_indices: vec![crosshair_mesh_index],
                        mesh_type: GameNodeMeshType::Pbr {
                            material_override: Some(DynamicPbrParams {
                                emissive_factor: crosshair_color,
                                base_color_factor: Vec4::new(0.0, 0.0, 0.0, 1.0),
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

    // logger_log(&format!("{:?}", &revolver));

    // anyhow::bail!("suhh dude");

    Ok(GameState {
        scene,
        time_tracker: None,
        state_update_time_accumulator: 0.0,
        is_playing_animations: true,

        audio_streams,
        audio_manager: audio_manager_mutex,
        bgm_sound_index: None,
        gunshot_sound_index: None,
        // gunshot_sound_data,
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
        revolver: None,

        bouncing_ball_node_id,
        bouncing_ball_body_handle,

        physics_state,

        physics_balls,

        character: None,
        player_controller,

        cube_mesh,

        asset_loader: asset_loader_clone,
    })
}

pub fn process_device_input(
    game_state: &mut GameState,
    renderer: &Renderer,
    event: &winit::event::DeviceEvent,
) {
    game_state
        .player_controller
        .process_device_events(event, &renderer.ui_overlay);
}

pub fn process_window_input(
    game_state: &mut GameState,
    renderer: &mut Renderer,
    event: &winit::event::WindowEvent,
    window: &winit::window::Window,
) {
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
            let mut render_data_guard = renderer.data.lock().unwrap();

            match keycode {
                VirtualKeyCode::Z => {
                    drop(render_data_guard);
                    increment_render_scale(renderer, false, window);
                }
                VirtualKeyCode::X => {
                    drop(render_data_guard);
                    increment_render_scale(renderer, true, window);
                }
                VirtualKeyCode::E => {
                    increment_exposure(&mut render_data_guard, false);
                }
                VirtualKeyCode::R => {
                    increment_exposure(&mut render_data_guard, true);
                }
                VirtualKeyCode::T => {
                    increment_bloom_threshold(&mut render_data_guard, false);
                }
                VirtualKeyCode::Y => {
                    increment_bloom_threshold(&mut render_data_guard, true);
                }
                VirtualKeyCode::P => {
                    game_state.is_playing_animations = !game_state.is_playing_animations;
                }
                VirtualKeyCode::M => {
                    render_data_guard.enable_shadows = !render_data_guard.enable_shadows;
                }
                VirtualKeyCode::B => {
                    render_data_guard.enable_bloom = !render_data_guard.enable_bloom;
                }
                VirtualKeyCode::F => {
                    render_data_guard.enable_wireframe_mode =
                        !render_data_guard.enable_wireframe_mode;
                }
                VirtualKeyCode::J => {
                    render_data_guard.draw_node_bounding_spheres =
                        !render_data_guard.draw_node_bounding_spheres;
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
        .process_window_events(event, window, &mut renderer.ui_overlay);
}

pub fn increment_render_scale(
    renderer: &mut Renderer,
    increase: bool,
    window: &winit::window::Window,
) {
    let delta = 0.1;
    let change = if increase { delta } else { -delta };

    {
        let mut renderer_data_guard = renderer.data.lock().unwrap();
        let surface_config_guard = renderer.base.surface_config.lock().unwrap();
        renderer_data_guard.render_scale =
            (renderer_data_guard.render_scale + change).clamp(0.1, 4.0);
        log::info!(
            "Render scale: {:?} ({:?}x{:?})",
            renderer_data_guard.render_scale,
            (surface_config_guard.width as f32 * renderer_data_guard.render_scale.sqrt()).round()
                as u32,
            (surface_config_guard.height as f32 * renderer_data_guard.render_scale.sqrt()).round()
                as u32,
        )
    }
    renderer.resize(window.inner_size(), window.scale_factor());
}

pub fn increment_exposure(renderer_data: &mut RendererData, increase: bool) {
    let delta = 0.05;
    let change = if increase { delta } else { -delta };
    renderer_data.tone_mapping_exposure =
        (renderer_data.tone_mapping_exposure + change).clamp(0.0, 20.0);
    log::info!("Exposure: {:?}", renderer_data.tone_mapping_exposure);
}

pub fn increment_bloom_threshold(renderer_data: &mut RendererData, increase: bool) {
    let delta = 0.05;
    let change = if increase { delta } else { -delta };
    renderer_data.bloom_threshold = (renderer_data.bloom_threshold + change).clamp(0.0, 20.0);
    log::info!("Bloom Threshold: {:?}", renderer_data.bloom_threshold);
}

#[profiling::function]
pub fn update_game_state(game_state: &mut GameState, renderer: &mut Renderer) {
    let base_renderer = renderer.base.clone();
    let renderer_data = renderer.data.clone();
    let renderer_constant_data = renderer.constant_data.clone();

    game_state
        .asset_loader
        .update(base_renderer.clone(), renderer_constant_data.clone());

    {
        let loaded_skyboxes = game_state.asset_loader.loaded_skyboxes();
        let mut loaded_skyboxes_guard = loaded_skyboxes.lock().unwrap();

        if let Entry::Occupied(entry) = loaded_skyboxes_guard.entry("skybox".to_string()) {
            let (_, skybox) = entry.remove_entry();
            renderer.set_skybox(skybox);
        }
    }

    {
        let loaded_scenes = game_state.asset_loader.loaded_scenes();
        let mut loaded_assets_guard = loaded_scenes.lock().unwrap();
        let mut renderer_data_guard = renderer_data.lock().unwrap();
        if game_state.gunshot_sound_index.is_some() {
            if let Entry::Occupied(entry) =
                loaded_assets_guard.entry("src/models/gltf/ColtPython/colt_python.glb".to_string())
            {
                let (_, (other_scene, other_render_buffers)) = entry.remove_entry();
                game_state.scene.merge_scene(
                    &mut renderer_data_guard,
                    other_scene,
                    other_render_buffers,
                );

                let node_id = game_state.scene.nodes().last().unwrap().id();
                let animation_index = game_state.scene.animations.len() - 1;
                // revolver_indices = Some((revolver_model_node_id, animation_index));
                game_state.revolver = Some(Revolver::new(
                    &mut game_state.scene,
                    game_state.player_node_id,
                    node_id,
                    animation_index,
                    // revolver model
                    // TransformBuilder::new()
                    //     .position(Vec3::new(0.21, -0.09, -1.0))
                    //     .rotation(make_quat_from_axis_angle(
                    //         Vec3::new(0.0, 1.0, 0.0),
                    //         deg_to_rad(180.0).into(),
                    //     ))
                    //     .scale(0.17f32 * Vec3::new(1.0, 1.0, 1.0))
                    //     .build(),
                    // colt python model
                    TransformBuilder::new()
                        .position(Vec3::new(0.21, -0.13, -1.0))
                        .rotation(
                            make_quat_from_axis_angle(Vec3::new(0.0, 1.0, 0.0), deg_to_rad(180.0))
                                * make_quat_from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.1),
                        )
                        .scale(2.0f32 * Vec3::new(1.0, 1.0, 1.0))
                        .build(),
                ));
            }
        }

        if let Entry::Occupied(entry) =
            loaded_assets_guard.entry("src/models/gltf/free_low_poly_forest/scene.glb".to_string())
        {
            let (_, (mut other_scene, other_render_buffers)) = entry.remove_entry();
            // hack to get the terrain to be at the same height as the ground.
            let node_has_parent: Vec<_> = other_scene
                .nodes()
                .map(|node| node.parent_id.is_some())
                .collect();
            for (i, node) in other_scene.nodes_mut().enumerate() {
                if node_has_parent[i] {
                    continue;
                }
                node.transform
                    .set_position(node.transform.position() + Vec3::new(0.0, 29.0, 0.0));
            }
            game_state.scene.merge_scene(
                &mut renderer_data_guard,
                other_scene,
                other_render_buffers,
            );
        }

        if let Entry::Occupied(entry) = loaded_assets_guard
            .entry("src/models/gltf/LegendaryRobot/Legendary_Robot.glb".to_string())
        {
            let (_, (mut other_scene, other_render_buffers)) = entry.remove_entry();
            if let Some(jump_up_animation) = other_scene
                .animations
                .iter_mut()
                .find(|animation| animation.name == Some(String::from("jump_up_root_motion")))
            {
                jump_up_animation.speed = 0.25;
                jump_up_animation.state.is_playing = true;
                jump_up_animation.state.loop_type = LoopType::Wrap;
            }
            game_state.scene.merge_scene(
                &mut renderer_data_guard,
                other_scene,
                other_render_buffers,
            );
        }

        if let Entry::Occupied(entry) =
            loaded_assets_guard.entry("src/models/gltf/TestLevel/test_level.glb".to_string())
        {
            let (_, (other_scene, other_render_buffers)) = entry.remove_entry();
            let skip_nodes = game_state.scene.node_count();
            game_state.scene.merge_scene(
                &mut renderer_data_guard,
                other_scene,
                other_render_buffers,
            );

            let test_level_node_ids: Vec<_> = game_state
                .scene
                .nodes()
                .skip(skip_nodes)
                .map(|node| node.id())
                .collect();
            for node_id in test_level_node_ids {
                if let Some(mesh) = game_state
                    .scene
                    .get_node_mut(node_id)
                    .unwrap()
                    .mesh
                    .as_mut()
                {
                    mesh.wireframe = true;
                }
                game_state.physics_state.add_static_box(
                    &game_state.scene,
                    &renderer_data_guard,
                    node_id,
                );
            }
        }

        if let Entry::Occupied(entry) = loaded_assets_guard.entry(get_misc_gltf_path().to_string())
        {
            let (_, (mut other_scene, other_render_buffers)) = entry.remove_entry();
            for animation in other_scene.animations.iter_mut() {
                animation.state.is_playing = true;
                animation.state.loop_type = LoopType::Wrap;
            }
            game_state.scene.merge_scene(
                &mut renderer_data_guard,
                other_scene,
                other_render_buffers,
            );
        }
    }

    // if game_state
    //     .asset_loader
    //     .pending_gltf_scenes
    //     .lock()
    //     .unwrap()
    //     .is_empty()
    {
        let mut loaded_audio_guard = game_state.asset_loader.loaded_audio.lock().unwrap();
        // let mut audio_manager_guard = game_state.audio_manager.lock().unwrap();

        if let Entry::Occupied(entry) = loaded_audio_guard.entry("src/sounds/bgm.mp3".to_string()) {
            let (_, bgm_sound_index) = entry.remove_entry();
            game_state.bgm_sound_index = Some(bgm_sound_index);

            let audio_manager_clone = game_state.audio_manager.clone();
            let bgm_sound_index_clone = bgm_sound_index;
            crate::thread::spawn(move || {
                // #[cfg(not(target_arch = "wasm32"))]
                // crate::thread::sleep(crate::time::Duration::from_secs_f32(5.0));

                let mut audio_manager_guard = audio_manager_clone.lock().unwrap();
                audio_manager_guard.play_sound(bgm_sound_index_clone);
            });
            // logger_log("loaded bgm sound");
        }

        if let Entry::Occupied(entry) =
            loaded_audio_guard.entry("src/sounds/gunshot.wav".to_string())
        {
            let (_, gunshot_sound_index) = entry.remove_entry();
            game_state.gunshot_sound_index = Some(gunshot_sound_index);
            // logger_log("loaded gunshot sound");
            // audio_manager_guard.set_sound_volume(gunshot_sound_index, 0.001);
        }
    }

    if game_state.character.is_none() {
        let legendary_robot_root_node_id = game_state
            .scene
            .nodes()
            .find(|node| node.name == Some(String::from("robot")))
            .map(|legendary_robot_root_node| legendary_robot_root_node.id());

        game_state.character = legendary_robot_root_node_id.map(|legendary_robot_root_node_id| {
            game_state
                .scene
                .get_node_mut(legendary_robot_root_node_id)
                .unwrap()
                .transform
                .set_position(Vec3::new(2.0, 0.0, 0.0));

            let legendary_robot_skin_index = 0;

            Character::new(
                &mut game_state.scene,
                &mut game_state.physics_state,
                &base_renderer,
                &mut renderer_data.lock().unwrap(),
                legendary_robot_root_node_id,
                legendary_robot_skin_index,
                &game_state.cube_mesh,
            )
        });
    }

    // "src/models/gltf/free_low_poly_forest/scene.gltf"

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
            let t = global_time_seconds;
            // let t = game_state.player_controller.speed;
            node.transform.set_position(
                Vec3::new(0.0, 6.5, 0.0)
                    + Vec3::new(
                        (t * 2.0).cos() * (t * 0.25).cos(),
                        2.0 * (t * 1.0).cos(),
                        (t * 2.0).sin() * (t * 0.5).sin(),
                    ),
            );
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
            // transform.set_position(Vec3::new(
            //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).cos(),
            //     transform.position.get().y,
            //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).sin(),
            // ));
            // let color = lerp_vec(LIGHT_COLOR_B, LIGHT_COLOR_A, (time_seconds * 2.0).sin());

            DirectionalLightComponent {
                direction: Vec3::new(direction.x, direction.y + 0.00001, direction.z),
                ..*directional_light_0
            }
        });
    if let Some(directional_light_0) = directional_light_0 {
        game_state.directional_lights[0] = directional_light_0;
    }

    // rotate the test object
    let rotational_displacement =
        make_quat_from_axis_angle(Vec3::new(0.0, 1.0, 0.0), frame_time_seconds / 5.0);
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
                .position(Vec3::new(0.0, 0.0, -1.0))
                .rotation(make_quat_from_axis_angle(
                    Vec3::new(0.0, 1.0, 0.0),
                    deg_to_rad(90.0),
                ))
                .scale(
                    (1080.0 / base_renderer.window_size.lock().unwrap().height as f32)
                        * 0.06
                        * Vec3::new(1.0, 1.0, 1.0),
                )
                .build();
    }

    if let Some(revolver) = game_state.revolver.as_mut() {
        revolver.update(
            game_state.player_controller.view_direction,
            &mut game_state.scene,
        );

        if game_state.player_controller.mouse_button_pressed && revolver.fire(&mut game_state.scene)
        {
            /* if let Some(bgm_sound_index) = game_state.bgm_sound_index {
                if time_tracker.global_time_seconds() > 30.0 {
                    game_state
                        .audio_manager
                        .lock()
                        .unwrap()
                        .play_sound(bgm_sound_index);
                    game_state.bgm_sound_index = None;
                }
            } */

            if let Some(gunshot_sound_index) = game_state.gunshot_sound_index {
                {
                    let mut audio_manager_guard = game_state.audio_manager.lock().unwrap();
                    audio_manager_guard.play_sound(gunshot_sound_index);
                    audio_manager_guard.reload_sound(
                        gunshot_sound_index,
                        SoundParams {
                            initial_volume: 0.4,
                            fixed_volume: true,
                            spacial_params: None,
                            stream: false,
                        },
                    )
                }
            }

            // logger_log("Fired!");
            let player_position = game_state
                .player_controller
                .position(&game_state.physics_state);
            let direction_vec = game_state.player_controller.view_direction.to_vector();
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
