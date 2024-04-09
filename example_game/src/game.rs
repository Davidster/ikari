use crate::ball::BallComponent;
use crate::character::Character;
use crate::game_state::AssetIds;
use crate::game_state::GameState;
use crate::physics_ball::PhysicsBall;
use crate::revolver::Revolver;
use crate::ui_overlay::AudioSoundStats;
use crate::ui_overlay::FrameStats;
use crate::ui_overlay::Message;
use crate::ui_overlay::UiOverlay;
use crate::ui_overlay::LATO_BOLD_FONT_BYTES;
use crate::ui_overlay::LATO_FONT_BYTES;
use crate::ui_overlay::LATO_FONT_NAME;
use crate::ui_overlay::PACIFICO_FONT_BYTES;

use std::{collections::hash_map::Entry, sync::Arc};

use anyhow::Result;
use glam::f32::{Vec3, Vec4};
use glam::Mat4;
use glam::Quat;
use ikari::animation::step_animations;
use ikari::animation::LoopType;
use ikari::asset_loader::AudioAssetLoadParams;
use ikari::asset_loader::SceneAssetLoadParams;
use ikari::audio::SoundParams;
use ikari::engine_state::EngineState;
use ikari::file_manager::{FileManager, GamePathMaker};
use ikari::gameloop::GameContext;
use ikari::mesh::BasicMesh;
use ikari::mesh::DynamicPbrParams;
use ikari::mesh::PbrTextures;
use ikari::mesh::Vertex;
use ikari::mutex::Mutex;
use ikari::physics::rapier3d_f64::prelude::*;
use ikari::physics::PhysicsState;
use ikari::player_controller::ControlledViewDirection;
use ikari::player_controller::PlayerController;
use ikari::raw_image::RawImage;
use ikari::renderer::BloomType;
use ikari::renderer::DirectionalLight;
use ikari::renderer::DirectionalLightShadowMappingConfig;
use ikari::renderer::PointLight;
use ikari::renderer::RendererData;
use ikari::renderer::SkyboxPaths;
use ikari::renderer::SkyboxSlot;
use ikari::renderer::{Renderer, SkyboxBackgroundPath, SkyboxEnvironmentHDRPath, SurfaceData};
use ikari::sampler_cache::SamplerDescriptor;
use ikari::scene::GameNodeDesc;
use ikari::scene::GameNodeDescBuilder;
use ikari::scene::GameNodeId;
use ikari::scene::GameNodeVisual;
use ikari::scene::Material;
use ikari::scene::Scene;
use ikari::texture::Texture;
use ikari::transform::Transform;
use ikari::transform::TransformBuilder;
use ikari::ui::IkariUiContainer;
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::Key;
use winit::keyboard::NamedKey;

// graphics settings
// TODO: replace the ones that are using the defaults with a call to ::default
pub const INITIAL_ENABLE_VSYNC: bool = true;
pub const INITIAL_NEAR_PLANE_DISTANCE: f32 = 0.001;
pub const INITIAL_FAR_PLANE_DISTANCE: f32 = 100000.0;
pub const INITIAL_FOV_X: f32 = 103.0 * std::f32::consts::PI / 180.0;
pub const INITIAL_ENABLE_DEPTH_PREPASS: bool = false;
pub const INITIAL_ENABLE_SHADOWS: bool = false;
pub const INITIAL_RENDER_SCALE: f32 = 1.0;
pub const INITIAL_TONE_MAPPING_EXPOSURE: f32 = 1.0;
pub const INITIAL_SHADOW_SMALL_OBJECT_CULLING_SIZE_PIXELS: f32 = 16.0;
pub const INITIAL_OLD_BLOOM_THRESHOLD: f32 = 0.8;
pub const INITIAL_OLD_BLOOM_RAMP_SIZE: f32 = 0.2;
pub const INITIAL_NEW_BLOOM_RADIUS: f32 = 0.005;
pub const INITIAL_NEW_BLOOM_INTENSITY: f32 = 0.04;
pub const INITIAL_BLOOM_TYPE: BloomType = BloomType::New;
pub const INITIAL_IS_SHOWING_CAMERA_POSE: bool = false;
pub const INITIAL_IS_SHOWING_CURSOR_MARKER: bool = false;
pub const INITIAL_ENABLE_SHADOW_DEBUG: bool = false;
pub const INITIAL_ENABLE_CASCADE_DEBUG: bool = false;
pub const INITIAL_ENABLE_CULLING_FRUSTUM_DEBUG: bool = false;
pub const INITIAL_ENABLE_POINT_LIGHT_CULLING_FRUSTUM_DEBUG: bool = false;
pub const INITIAL_ENABLE_DIRECTIONAL_LIGHT_CULLING_FRUSTUM_DEBUG: bool = false;
pub const INITIAL_ENABLE_SOFT_SHADOWS: bool = true;
pub const INITIAL_SHADOW_BIAS: f32 = 0.001;
pub const INITIAL_SKYBOX_WEIGHT: f32 = 1.0;
pub const INITIAL_SOFT_SHADOW_FACTOR: f32 = 0.00003;
pub const INITIAL_SOFT_SHADOW_GRID_DIMS: u32 = 4;
pub const INITIAL_SOFT_SHADOWS_MAX_DISTANCE: f32 = 100.0;

// game settings
pub const ARENA_SIDE_LENGTH: f32 = 500.0;
pub const ENABLE_GRAVITY: bool = true;
pub const ENABLE_GRAVITY_ON_PLAYER: bool = false;
pub const PLAYER_MOVEMENT_SPEED: f32 = 6.0;

pub const CREATE_POINT_SHADOW_MAP_DEBUG_OBJECTS: bool = false;
pub const REMOVE_LARGE_OBJECTS_FROM_FOREST: bool = false;
pub const ARROW_KEYS_ADJUST_WEAPON_POSITION: bool = false;

// linear colors, not srgb
pub const _DIRECTIONAL_LIGHT_COLOR_A: Vec3 = Vec3::new(0.84922975, 0.81581426, 0.8832506);
pub const DIRECTIONAL_LIGHT_COLOR_B: Vec3 = Vec3::new(0.81115574, 0.77142686, 0.8088144);
pub const POINT_LIGHT_COLOR: Vec3 = Vec3::new(0.93126976, 0.7402633, 0.49407062);
pub const POINT_LIGHT_COLOR_B: Vec3 = Vec3::new(0.25, 0.973, 0.663);
// pub const LIGHT_COLOR_C: Vec3 =
//     Vec3::new(from_srgb(0.631), from_srgb(0.565), from_srgb(0.627));

pub const COLLISION_GROUP_PLAYER_UNSHOOTABLE: Group = Group::GROUP_1;

#[cfg(not(target_arch = "wasm32"))]
lazy_static::lazy_static! {
    pub static ref GAME_PATH_MAKER: GamePathMaker = GamePathMaker::new(Some("ikari".into()));
}

#[cfg(target_arch = "wasm32")]
lazy_static::lazy_static! {
    pub static ref GAME_PATH_MAKER: GamePathMaker = GamePathMaker::new(Some("ikari".into()), String::from("http://localhost:8000"));
}

// order of the images for a cubemap is documented here:
// https://www.khronos.org/opengl/wiki/Cubemap_Texture
pub fn get_skybox_path() -> SkyboxPaths {
    // Mountains
    // src: https://github.com/JoeyDeVries/LearnOpenGL/tree/master/resources/textures/skybox
    let _background = SkyboxBackgroundPath::Cube([
        GAME_PATH_MAKER.make("src/textures/skybox/right.jpg"),
        GAME_PATH_MAKER.make("src/textures/skybox/left.jpg"),
        GAME_PATH_MAKER.make("src/textures/skybox/top.jpg"),
        GAME_PATH_MAKER.make("src/textures/skybox/bottom.jpg"),
        GAME_PATH_MAKER.make("src/textures/skybox/front.jpg"),
        GAME_PATH_MAKER.make("src/textures/skybox/back.jpg"),
    ]);
    let _environment_hdr: Option<SkyboxEnvironmentHDRPath> = None;

    // Newport Loft
    // src: http://www.hdrlabs.com/sibl/archive/
    let _background = SkyboxBackgroundPath::Equirectangular(
        GAME_PATH_MAKER.make("src/textures/newport_loft/background.jpg"),
    );
    let _environment_hdr: Option<SkyboxEnvironmentHDRPath> =
        Some(SkyboxEnvironmentHDRPath::Equirectangular(
            GAME_PATH_MAKER.make("src/textures/newport_loft/radiance.hdr"),
        ));

    // Milkyway
    // src: http://www.hdrlabs.com/sibl/archive/
    let _background = SkyboxBackgroundPath::Equirectangular(
        GAME_PATH_MAKER.make("src/textures/milkyway/background.jpg"),
    );
    let _environment_hdr: Option<SkyboxEnvironmentHDRPath> =
        Some(SkyboxEnvironmentHDRPath::Equirectangular(
            GAME_PATH_MAKER.make("src/textures/milkyway/radiance.hdr"),
        ));

    let preprocessed_skybox_folder = "milkyway";
    // let preprocessed_skybox_folder = "photosphere_small";

    let background = SkyboxBackgroundPath::ProcessedCube([
        GAME_PATH_MAKER.make(format!(
            "src/skyboxes/{preprocessed_skybox_folder}/background/pos_x.png"
        )),
        GAME_PATH_MAKER.make(format!(
            "src/skyboxes/{preprocessed_skybox_folder}/background/neg_x.png"
        )),
        GAME_PATH_MAKER.make(format!(
            "src/skyboxes/{preprocessed_skybox_folder}/background/pos_y.png"
        )),
        GAME_PATH_MAKER.make(format!(
            "src/skyboxes/{preprocessed_skybox_folder}/background/neg_y.png"
        )),
        GAME_PATH_MAKER.make(format!(
            "src/skyboxes/{preprocessed_skybox_folder}/background/pos_z.png"
        )),
        GAME_PATH_MAKER.make(format!(
            "src/skyboxes/{preprocessed_skybox_folder}/background/neg_z.png"
        )),
    ]);
    let environment_hdr: Option<SkyboxEnvironmentHDRPath> =
        Some(SkyboxEnvironmentHDRPath::ProcessedCube {
            diffuse: GAME_PATH_MAKER.make(format!(
                "src/skyboxes/{preprocessed_skybox_folder}/diffuse_environment_map_compressed.bin"
            )),
            specular: GAME_PATH_MAKER.make(format!(
                "src/skyboxes/{preprocessed_skybox_folder}/specular_environment_map_compressed.bin"
            )),
        });

    // My photosphere pic
    // src: me
    let _background = SkyboxBackgroundPath::Equirectangular(
        GAME_PATH_MAKER.make("src/textures/photosphere_skybox_small.jpg"),
    );
    let _environment_hdr: Option<SkyboxEnvironmentHDRPath> = None;

    SkyboxPaths {
        background,
        environment_hdr,
    }
}

fn get_misc_gltf_paths() -> &'static [&'static str] {
    &[
        // "/home/david/Downloads/adamHead/adamHead.gltf",
        // "src/models/gltf/free_low_poly_forest/scene.gltf",
        // "src/models/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf",
        // "src/models/gltf/SimpleMeshes/SimpleMeshes.gltf",
        // "src/models/gltf/Triangle/Triangle.gltf",
        // "src/models/gltf/TriangleWithoutIndices/TriangleWithoutIndices.gltf",
        // "src/models/gltf/EnvironmentTest/EnvironmentTest.gltf",
        // "src/models/gltf/Arrow/Arrow.gltf",
        "src/models/gltf/DamagedHelmet/DamagedHelmet.gltf",
        // "src/models/gltf/VertexColorTest/VertexColorTest.gltf",
        // "src/models/gltf/Revolver/revolver_low_poly.gltf",
        // "src/models/gltf/NormalTangentMirrorTest/NormalTangentMirrorTest.gltf",
        // "src/models/gltf/NormalTangentTest/NormalTangentTest.glb",
        // "src/models/gltf/TextureLinearInterpolationTest/TextureLinearInterpolationTest.glb",
        // "../glTF-Sample-Models/2.0/RiggedFigure/glTF/RiggedFigure.gltf",
        // "../glTF-Sample-Models/2.0/RiggedSimple/glTF/RiggedSimple.gltf",
        // "../glTF-Sample-Models/2.0/CesiumMan/glTF/CesiumMan.gltf",
        // "../glTF-Sample-Models/2.0/Fox/glTF/Fox.gltf",
        // "../glTF-Sample-Models/2.0/RecursiveSkeletons/glTF/RecursiveSkeletons.gltf",
        // "../glTF-Sample-Models/2.0/BrainStem/glTF/BrainStem.gltf",
        // "/home/david/Programming/glTF-Sample-Models/2.0/BoxAnimated/glTF/BoxAnimated.gltf",
        // "/home/david/Programming/glTF-Sample-Models/2.0/Lantern/glTF/Lantern.gltf",
        // "src/models/gltf/VC/VC.gltf",
        //  "../glTF-Sample-Models-master/2.0/InterpolationTest/glTF/InterpolationTest.gltf",
        // "src/models/gltf/Sponza/Sponza.gltf",
    ]
}

#[cfg(not(target_arch = "wasm32"))]
async fn get_rainbow_img_raw() -> Result<RawImage> {
    let texture_compressor = ikari::texture_compression::TextureCompressor;
    texture_compressor.transcode_image(
        &FileManager::read(
            &GAME_PATH_MAKER.make("src/textures/rainbow_gradient_vertical_compressed.bin"),
        )
        .await?,
        true,
        false,
    )
}

#[cfg(target_arch = "wasm32")]
async fn get_rainbow_img_raw() -> Result<RawImage> {
    Ok(RawImage::from_dynamic_image(
        image::load_from_memory(
            &FileManager::read(&GAME_PATH_MAKER.make("src/textures/rainbow_gradient_vertical.jpg"))
                .await?,
        )?,
        true,
    ))
}

pub async fn init_game_state(
    engine_state: &mut EngineState,
    renderer: &mut Renderer,
    surface_data: &mut SurfaceData,
    window: &winit::window::Window,
) -> Result<GameState> {
    log::info!("Controls:");
    [
        "Look Around:             Mouse",
        "Move Around:             WASD, E, Space Bar, LCtrl, Q",
        "Adjust Speed:            Scroll or Up/Down Arrow Keys",
        "Adjust Render Scale:     Z / X",
        "Adjust Exposure:         R / T",
        "Adjust Bloom Threshold:  Y / U",
        "Pause/Resume Animations: P",
        "Toggle Bloom Effect:     B",
        "Toggle Shadows:          M",
        "Toggle Wireframe:        F",
        "Toggle Collision Boxes:  C",
        "Draw Bounding Spheres:   J",
        "Open Options Menu:       Tab",
    ]
    .iter()
    .for_each(|line| {
        log::info!("  {line}");
    });

    {
        let mut renderer_data_guard = renderer.data.lock();
        renderer_data_guard.general_settings.render_scale = INITIAL_RENDER_SCALE;

        renderer_data_guard
            .post_effect_settings
            .tone_mapping_exposure = INITIAL_TONE_MAPPING_EXPOSURE;

        renderer_data_guard.post_effect_settings.bloom_type = INITIAL_BLOOM_TYPE;
        renderer_data_guard.post_effect_settings.old_bloom_threshold = INITIAL_OLD_BLOOM_THRESHOLD;
        renderer_data_guard.post_effect_settings.old_bloom_ramp_size = INITIAL_OLD_BLOOM_RAMP_SIZE;

        renderer_data_guard.shadow_settings.enable_shadows = INITIAL_ENABLE_SHADOWS;
    }

    // dbg!(&surface_data.surface_config);

    let unscaled_framebuffer_size = winit::dpi::PhysicalSize::new(
        surface_data.surface_config.width,
        surface_data.surface_config.height,
    );
    // must call this after changing the render scale
    renderer.resize_surface(surface_data, unscaled_framebuffer_size);

    let asset_loader = engine_state.asset_loader.clone();
    let asset_ids = Arc::new(Mutex::new(AssetIds::default()));
    let asset_ids_clone = asset_ids.clone();

    ikari::thread::spawn(move || {
        #[allow(clippy::vec_init_then_push)]
        ikari::block_on(async move {
            // ikari::thread::sleep_async(ikari::time::Duration::from_secs_f32(5.0)).await;

            // load in gltf files
            let load_gltf = |path| {
                asset_loader.load_gltf_scene(SceneAssetLoadParams {
                    path: GAME_PATH_MAKER.make(path),
                    generate_wireframe_meshes: true,
                })
            };

            let mut asset_ids = asset_ids_clone.lock();

            // player's revolver
            // https://done3d.com/colt-python-8-inch/
            asset_ids.gun = load_gltf("src/models/gltf/ColtPython/colt_python.glb").into();
            // asset_ids.gun = load_gltf("src/models/gltf/Revolver/revolver_low_poly.gltf").into();

            // forest
            // https://sketchfab.com/3d-models/free-low-poly-forest-6dc8c85121234cb59dbd53a673fa2b8f
            asset_ids.forest = load_gltf("src/models/gltf/free_low_poly_forest/scene.gltf").into();

            // legendary robot
            // https://www.cgtrader.com/free-3d-models/character/sci-fi-character/legendary-robot-free-low-poly-3d-model
            asset_ids.legendary_robot =
                load_gltf("src/models/gltf/LegendaryRobot/Legendary_Robot.gltf").into();

            asset_ids.test_level = load_gltf("src/models/gltf/TestLevel/test_level.gltf").into();

            for path in get_misc_gltf_paths() {
                asset_ids.anonymous_scenes.push(load_gltf(path));
            }

            asset_ids.gunshot = asset_loader
                .load_audio(AudioAssetLoadParams {
                    path: GAME_PATH_MAKER.make("src/sounds/gunshot.wav"),
                    sound_params: SoundParams {
                        initial_volume: 0.4,
                        fixed_volume: true,
                        spacial_params: None,
                        stream: false,
                    },
                })
                .into();

            asset_ids.bgm = asset_loader
                .load_audio(AudioAssetLoadParams {
                    path: GAME_PATH_MAKER.make("src/sounds/bgm.mp3"),
                    sound_params: SoundParams {
                        initial_volume: 0.3,
                        fixed_volume: false,
                        spacial_params: None,
                        stream: true,
                    },
                })
                .into();

            asset_ids.skybox = asset_loader.load_skybox(get_skybox_path()).into();
        })
    });

    // add lights to the scene
    engine_state.scene.directional_lights = vec![
        DirectionalLight {
            direction: (-Vec3::new(-1.0, 10.0, 10.0)).normalize(),
            color: DIRECTIONAL_LIGHT_COLOR_B,
            intensity: 1.0,
            shadow_mapping_config: DirectionalLightShadowMappingConfig {
                // num_cascades: 2,
                // maximum_distance: 100.0,
                // first_cascade_far_bound: 10.0,
                ..Default::default()
            },
        },
        // DirectionalLight {
        //     direction: (-Vec3::new(1.0, 5.0, -10.0)).normalize(),
        //     color: _DIRECTIONAL_LIGHT_COLOR_A,
        //     intensity: 1.0,
        //     shadow_mapping_config: Default::default(),
        // },
        // DirectionalLight {
        //     direction: (-Vec3::new(10.0, 10.0, 1.0)).normalize(),
        //     color: DIRECTIONAL_LIGHT_COLOR_B,
        //     intensity: 0.2,
        //     shadow_mapping_config: Default::default(),
        // },
    ];
    // let directional_lights: Vec<DirectionalLightComponent> = vec![];

    let point_lights: Vec<(Transform, Vec3, f32)> = vec![
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
        // (
        //     TransformBuilder::new()
        //         .scale(Vec3::new(0.1, 0.1, 0.1))
        //         .position(Vec3::new(0.0, 10.0, 0.0))
        //         .build(),
        //     DIRECTIONAL_LIGHT_COLOR_B,
        //     1.0,
        // ),
    ];
    // let point_lights: Vec<(ikari::transform::Transform, Vec3, f32)> = vec![];

    let physics_state = &mut engine_state.physics_state;
    let scene = &mut engine_state.scene;

    let player_node_id = scene.add_node(GameNodeDesc::default()).id();
    let player_controller = PlayerController::new(
        physics_state,
        PLAYER_MOVEMENT_SPEED,
        Vec3::new(8.0, 30.0, -13.0),
        ControlledViewDirection {
            horizontal: 180.0_f32.to_radians(),
            vertical: 0.0,
        },
        ColliderBuilder::capsule_y(0.5, 0.25)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .friction_combine_rule(CoefficientCombineRule::Min)
            .collision_groups(
                InteractionGroups::all().with_memberships(COLLISION_GROUP_PLAYER_UNSHOOTABLE),
            )
            .friction(0.0)
            .restitution(0.0)
            .build(),
    );

    physics_state.set_gravity_is_enabled(ENABLE_GRAVITY);
    player_controller.set_is_gravity_enabled(physics_state, ENABLE_GRAVITY_ON_PLAYER);

    renderer.data.lock().camera_node_id = Some(player_node_id);

    let mut point_light_node_ids: Vec<GameNodeId> = Vec::new();
    for (transform, color, intensity) in point_lights {
        let node_id = scene
            .add_node(
                GameNodeDescBuilder::new()
                    .visual(Some(GameNodeVisual::from_mesh_mat(
                        renderer.constant_data.sphere_mesh_index,
                        Material::Unlit {
                            color: color * intensity * 10.0,
                        },
                    )))
                    .transform(transform)
                    .build(),
            )
            .id();
        point_light_node_ids.push(node_id);
        scene.point_lights.push(PointLight {
            node_id,
            color,
            intensity,
        });
    }

    // let simple_normal_map_path = "src/textures/simple_normal_map.jpg";
    // let simple_normal_map_bytes = FileLoader::read(simple_normal_map_path).await?;
    // let simple_normal_map = Texture::from_encoded_image(
    //     &renderer.base.device,
    //     &renderer.base.queue,
    //     &simple_normal_map_bytes,
    //     simple_normal_map_path,
    //     wgpu::TextureFormat::Rgba8Unorm.into(),
    //     false,
    //     &Default::default(),
    // )?;

    let rainbow_img_raw = get_rainbow_img_raw().await?;
    let rainbow_texture = Texture::from_raw_image(
        &renderer.base,
        &rainbow_img_raw,
        Some("rainbow_texture"),
        false,
        &Default::default(),
    )?;

    let brick_normal_map_path = "src/textures/brick_normal_map.jpg";

    let brick_normal_map_raw = RawImage::from_dynamic_image(
        image::load_from_memory(
            &FileManager::read(&GAME_PATH_MAKER.make(brick_normal_map_path)).await?,
        )?,
        false,
    );
    let brick_normal_map = Texture::from_raw_image(
        &renderer.base,
        &brick_normal_map_raw,
        Some(brick_normal_map_path),
        false,
        &Default::default(),
    )?;

    // add test object to scene
    /* let earth_texture_path = "src/textures/8k_earth.jpg";
    let earth_texture_bytes = FileLoader::read(earth_texture_path).await?;
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
    let earth_normal_map_bytes = FileLoader::read(earth_normal_map_path).await?;
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

    let checkerboard_img_raw = {
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
        RawImage::from_dynamic_image(img.into(), true)
    };
    let checkerboard_texture = Texture::from_raw_image(
        &renderer.base,
        &checkerboard_img_raw,
        Some("checkerboard_texture"),
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
    let mars_texture_bytes = FileLoader::read(mars_texture_path).await?;
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

    let test_object_pbr_material_index = Renderer::bind_pbr_material(
        &renderer.base,
        &renderer.constant_data,
        &mut renderer.data.lock(),
        &PbrTextures {
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
                .visual(Some(GameNodeVisual::make_pbr(
                    renderer.constant_data.sphere_mesh_index,
                    test_object_pbr_material_index,
                )))
                .transform(
                    TransformBuilder::new()
                        .position(Vec3::new(4.0, 20.5, 3.0))
                        .scale(0.2 * Vec3::new(1.0, 1.0, 1.0))
                        .build(),
                )
                .name(Some("test_object".into()))
                .build(),
        )
        .id();
    // scene.remove_node(test_object_node_id);

    // add floor to scene

    let ball_count = 0;
    let balls: Vec<_> = (0..ball_count).map(|_| BallComponent::rand()).collect();

    let ball_pbr_material_index = Renderer::bind_pbr_material(
        &renderer.base,
        &renderer.constant_data,
        &mut renderer.data.lock(),
        &PbrTextures {
            base_color: Some(&rainbow_texture),
            ..Default::default()
        },
        Default::default(),
    )?;

    let mut ball_node_ids: Vec<GameNodeId> = Vec::new();
    for ball in &balls {
        let node = scene.add_node(
            GameNodeDescBuilder::new()
                .visual(Some(GameNodeVisual::make_pbr(
                    renderer.constant_data.sphere_mesh_index,
                    ball_pbr_material_index,
                )))
                .transform(ball.transform)
                .build(),
        );
        ball_node_ids.push(node.id());
    }

    let physics_ball_count = 500;
    let physics_balls: Vec<_> = (0..physics_ball_count)
        .map(|_| {
            PhysicsBall::new_random(
                scene,
                physics_state,
                GameNodeVisual::make_pbr(
                    renderer.constant_data.sphere_mesh_index,
                    ball_pbr_material_index,
                ),
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
        let ball_pbr_mesh_index = Renderer::bind_pbr_material(
            &renderer.base,
            &renderer.constant_data,
            &mut renderer.data.lock(),
            &PbrTextures {
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
                    .visual(Some(GameNodeVisual::make_pbr(
                        renderer.constant_data.sphere_mesh_index,
                        ball_pbr_mesh_index,
                    )))
                    .transform(
                        TransformBuilder::new()
                            .position(cube_center - Vec3::new(0.0, 1.0, 0.0))
                            .scale(0.2 * Vec3::new(1.0, 1.0, 1.0))
                            .build(),
                    )
                    .build(),
            )
            .id();

        let wall_pbr_material_index = Renderer::bind_pbr_material(
            &renderer.base,
            &renderer.constant_data,
            &mut renderer.data.lock(),
            &PbrTextures {
                base_color: Some(&checkerboard_texture),
                ..Default::default()
            },
            Default::default(),
        )?;
        let game_node_visual = GameNodeVisual::make_pbr(
            renderer.constant_data.plane_mesh_index,
            wall_pbr_material_index,
        );

        let ceiling_transform = TransformBuilder::new()
            .position(cube_center + Vec3::new(0.0, cube_radius, 0.0))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(Quat::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                180.0_f32.to_radians(),
            ))
            .build();
        let ceiling_game_node_mesh = GameNodeVisual {
            material: Material::Pbr {
                binded_material_index: wall_pbr_material_index,
                dynamic_pbr_params: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(1.0, 0.5, 0.5, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_visual.clone()
        };
        let _ceiling_node = scene.add_node(
            GameNodeDescBuilder::new()
                .visual(Some(ceiling_game_node_mesh))
                .transform(ceiling_transform)
                .build(),
        );

        let wall_1_node_mesh = GameNodeVisual {
            material: Material::Pbr {
                binded_material_index: wall_pbr_material_index,
                dynamic_pbr_params: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(0.5, 1.0, 0.5, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_visual.clone()
        };
        let wall_transform_1 = TransformBuilder::new()
            .position(cube_center + Vec3::new(0.0, 0.0, -cube_radius))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(Quat::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                90.0_f32.to_radians(),
            ))
            .build();
        let _wall_1_node = scene.add_node(
            GameNodeDescBuilder::new()
                .visual(Some(wall_1_node_mesh))
                .transform(wall_transform_1)
                .build(),
        );

        let wall_2_node_mesh = GameNodeVisual {
            material: Material::Pbr {
                binded_material_index: wall_pbr_material_index,
                dynamic_pbr_params: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(0.5, 0.5, 1.0, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_visual.clone()
        };
        let wall_transform_2 = TransformBuilder::new()
            .position(cube_center + Vec3::new(0.0, 0.0, cube_radius))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(Quat::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                270.0_f32.to_radians(),
            ))
            .build();
        let _wall_2_node = scene.add_node(
            GameNodeDescBuilder::new()
                .visual(Some(wall_2_node_mesh))
                .transform(wall_transform_2)
                .build(),
        );

        let wall_3_node_mesh = GameNodeVisual {
            material: Material::Pbr {
                binded_material_index: wall_pbr_material_index,
                dynamic_pbr_params: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(1.0, 0.5, 1.0, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_visual.clone()
        };
        let wall_transform_3 = TransformBuilder::new()
            .position(cube_center + Vec3::new(cube_radius, 0.0, 0.0))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(Quat::from_axis_angle(
                Vec3::new(0.0, 0.0, 1.0),
                90.0_f32.to_radians(),
            ))
            .build();
        let _wall_3_node = scene.add_node(
            GameNodeDescBuilder::new()
                .visual(Some(wall_3_node_mesh))
                .transform(wall_transform_3)
                .build(),
        );

        let wall_4_node_mesh = GameNodeVisual {
            material: Material::Pbr {
                binded_material_index: wall_pbr_material_index,
                dynamic_pbr_params: Some(DynamicPbrParams {
                    base_color_factor: Vec4::new(1.0, 1.0, 0.5, 1.0),
                    ..Default::default()
                }),
            },
            ..game_node_visual
        };
        let wall_transform_4 = TransformBuilder::new()
            .position(cube_center + Vec3::new(-cube_radius, 0.0, 0.0))
            .scale(Vec3::new(cube_radius, 1.0, cube_radius))
            .rotation(Quat::from_axis_angle(
                Vec3::new(0.0, 0.0, 1.0),
                270.0_f32.to_radians(),
            ))
            .build();
        let _wall_4_node = scene.add_node(
            GameNodeDescBuilder::new()
                .visual(Some(wall_4_node_mesh))
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
    let floor_material_index = Renderer::bind_pbr_material(
        &renderer.base,
        &renderer.constant_data,
        &mut renderer.data.lock(),
        &PbrTextures {
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
            .visual(Some(GameNodeVisual::make_pbr(
                renderer.constant_data.plane_mesh_index,
                floor_material_index,
            )))
            .transform(floor_transform)
            .build(),
    );
    // let ceiling_transform = TransformBuilder::new()
    //     .position(Vec3::new(0.0, 10.0, 0.0))
    //     .scale(Vec3::new(ARENA_SIDE_LENGTH, 1.0, ARENA_SIDE_LENGTH))
    //     .rotation(Quat::from_axis_angle(
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
        floor_transform.scale().x as f64,
        floor_thickness / 2.0,
        floor_transform.scale().z as f64,
    )
    .translation(vector![
        floor_transform.position().x as f64 / 2.0,
        floor_transform.position().y as f64 - (floor_thickness / 2.0),
        floor_transform.position().z as f64 / 2.0
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
        let bouncing_ball_pbr_mesh_index = Renderer::bind_pbr_material(
            &renderer.base,
            &renderer.constant_data,
            &mut renderer.data.lock(),
            &PbrTextures {
                base_color: Some(&checkerboard_texture),
                ..Default::default()
            },
            Default::default(),
        )?;
        let bouncing_ball_radius = 0.5;
        let bouncing_ball_node = scene.add_node(
            GameNodeDescBuilder::new()
                .visual(Some(GameNodeVisual::make_pbr(
                    renderer.constant_data.sphere_mesh_index,
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
                bouncing_ball_node.transform.position().x as f64,
                bouncing_ball_node.transform.position().y as f64,
                bouncing_ball_node.transform.position().z as f64
            ])
            .build();
        let bouncing_ball_collider = ColliderBuilder::ball(bouncing_ball_radius as f64)
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
    let crosshair_img_raw = {
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

        RawImage::from_dynamic_image(img.into(), true)
    };
    #[allow(unused_assignments)]
    let mut crosshair_node_id: Option<GameNodeId> = None;
    let crosshair_texture = Texture::from_raw_image(
        &renderer.base,
        &crosshair_img_raw,
        Some("crosshair_texture"),
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
            .map(|position| {
                Vertex {
                    position: [0.0, position[1], position[0]].into(),
                    tex_coords: [0.5 * (position[0] + 1.0), 0.5 * (1.0 - position[1])].into(),
                    color: [1.0, 1.0, 1.0].into(),
                    ..Default::default()
                }
                .into()
            })
            .collect(),
        indices: vec![0, 2, 1, 0, 3, 2],
    };
    let crosshair_mesh_index = Renderer::bind_basic_mesh(
        &renderer.base,
        &mut renderer.data.lock(),
        &crosshair_quad,
        false,
    );
    let crosshair_ambient_occlusion = Texture::from_color(&renderer.base, [0, 0, 0, 0])?;
    let crosshair_metallic_roughness = Texture::from_color(&renderer.base, [0, 0, 255, 0])?;
    let crosshair_material_index = Renderer::bind_pbr_material(
        &renderer.base,
        &renderer.constant_data,
        &mut renderer.data.lock(),
        &PbrTextures {
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
                    .visual(Some(GameNodeVisual::from_mesh_mat(
                        crosshair_mesh_index,
                        Material::Pbr {
                            binded_material_index: crosshair_material_index,
                            dynamic_pbr_params: Some(DynamicPbrParams {
                                emissive_factor: crosshair_color,
                                base_color_factor: Vec4::new(0.0, 0.0, 0.0, 1.0),
                                alpha_cutoff: 0.5,
                                ..Default::default()
                            }),
                        },
                    )))
                    .build(),
            )
            .id(),
    );

    let ui_overlay = {
        let surface_format = surface_data.surface_config.format;
        IkariUiContainer::new(
            window,
            &renderer.base.device,
            &renderer.base.queue,
            surface_format,
            UiOverlay::new(window),
            Some(LATO_FONT_NAME),
            vec![
                LATO_FONT_BYTES,
                LATO_BOLD_FONT_BYTES,
                PACIFICO_FONT_BYTES,
                iced_aw::graphics::icons::BOOTSTRAP_FONT_BYTES,
            ],
            crate::ui_overlay::THEME,
        )
    };

    Ok(GameState {
        state_update_time_accumulator: 0.0,
        is_playing_animations: true,

        bgm_sound_index: None,
        gunshot_sound_index: None,
        // gunshot_sound_data,
        point_light_node_ids,

        next_balls: balls.clone(),
        prev_balls: balls.clone(),
        actual_balls: balls,
        ball_node_ids,
        ball_pbr_mesh_index: ball_pbr_material_index,

        ball_spawner_acc: 0.0,

        test_object_node_id,
        crosshair_node_id,
        revolver: None,

        bouncing_ball_node_id,
        bouncing_ball_body_handle,

        physics_balls,

        // player_node_id,
        player_controller,
        character: None,

        asset_ids,

        ui_overlay,
    })
}

pub fn process_window_input(
    GameContext {
        game_state,
        engine_state,
        renderer,
        surface_data,
        window,
        elwt,
        ..
    }: GameContext<GameState>,
    event: &winit::event::WindowEvent,
) {
    #[allow(clippy::single_match)]
    match event {
        WindowEvent::KeyboardInput { event, .. } => {
            let key = event.logical_key.as_ref();
            if event.state == ElementState::Pressed && key == Key::Named(NamedKey::Tab) {
                game_state
                    .ui_overlay
                    .queue_message(Message::TogglePopupMenu);
            }

            if event.state == ElementState::Released {
                let mut render_data_guard = renderer.data.lock();
                match key {
                    Key::Character(character) => match character.to_lowercase().as_str() {
                        "z" => {
                            drop(render_data_guard);
                            increment_render_scale(
                                renderer,
                                surface_data,
                                false,
                                window,
                                &mut game_state.ui_overlay,
                            );
                        }
                        "x" => {
                            drop(render_data_guard);
                            increment_render_scale(
                                renderer,
                                surface_data,
                                true,
                                window,
                                &mut game_state.ui_overlay,
                            );
                        }
                        "r" => {
                            increment_exposure(&mut render_data_guard, false);
                        }
                        "t" => {
                            increment_exposure(&mut render_data_guard, true);
                        }
                        "y" => {
                            increment_old_bloom_threshold(&mut render_data_guard, false);
                        }
                        "u" => {
                            increment_old_bloom_threshold(&mut render_data_guard, true);
                        }
                        "p" => {
                            game_state.is_playing_animations = !game_state.is_playing_animations;
                        }
                        "m" => {
                            render_data_guard.shadow_settings.enable_shadows =
                                !render_data_guard.shadow_settings.enable_shadows;
                        }
                        "b" => {
                            game_state
                                .ui_overlay
                                .queue_message(Message::BloomTypeChanged(match render_data_guard
                                    .post_effect_settings
                                    .bloom_type
                                {
                                    BloomType::Disabled => BloomType::Old,
                                    BloomType::Old => BloomType::New,
                                    BloomType::New => BloomType::Disabled,
                                }));
                        }
                        "f" => {
                            render_data_guard.debug_settings.enable_wireframe_mode =
                                !render_data_guard.debug_settings.enable_wireframe_mode;
                        }
                        "j" => {
                            render_data_guard.debug_settings.draw_node_bounding_spheres =
                                !render_data_guard.debug_settings.draw_node_bounding_spheres;
                        }
                        "c" => {
                            if let Some(character) = game_state.character.as_mut() {
                                character.toggle_collision_box_display(&mut engine_state.scene);
                            }
                        }
                        _ => {}
                    },
                    Key::Named(NamedKey::Escape) => {
                        elwt.exit();
                    }
                    _ => {}
                };

                if ARROW_KEYS_ADJUST_WEAPON_POSITION {
                    if let Some(revolver) = game_state.revolver.as_mut() {
                        if let Some(node) = engine_state.scene.get_node_mut(revolver.node_id) {
                            match key {
                                Key::Named(NamedKey::ArrowUp) => {
                                    node.transform.set_position(dbg!(Vec3::new(
                                        node.transform.position().x,
                                        node.transform.position().y,
                                        node.transform.position().z - 0.01,
                                    )));
                                }
                                Key::Named(NamedKey::ArrowDown) => {
                                    node.transform.set_position(dbg!(Vec3::new(
                                        node.transform.position().x,
                                        node.transform.position().y,
                                        node.transform.position().z + 0.01,
                                    )));
                                }
                                Key::Named(NamedKey::ArrowRight) => {
                                    node.transform.set_position(dbg!(Vec3::new(
                                        node.transform.position().x + 0.01,
                                        node.transform.position().y,
                                        node.transform.position().z,
                                    )));
                                }
                                Key::Named(NamedKey::ArrowLeft) => {
                                    node.transform.set_position(dbg!(Vec3::new(
                                        node.transform.position().x - 0.01,
                                        node.transform.position().y,
                                        node.transform.position().z,
                                    )));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    };

    game_state
        .player_controller
        .process_window_event(event, window);

    game_state.ui_overlay.handle_window_event(window, event);
}

pub fn process_device_input(
    GameContext { game_state, .. }: GameContext<GameState>,
    event: &winit::event::DeviceEvent,
) {
    game_state.player_controller.process_device_event(event);
}

pub fn handle_window_resize(
    GameContext {
        game_state, window, ..
    }: GameContext<GameState>,
    new_size: winit::dpi::PhysicalSize<u32>,
) {
    resize_ui_overlay(&mut game_state.ui_overlay, window, new_size);
}

pub fn resize_ui_overlay(
    ui_overlay: &mut IkariUiContainer<UiOverlay>,
    window: &winit::window::Window,
    new_size: winit::dpi::PhysicalSize<u32>,
) {
    ui_overlay.resize(new_size, window.scale_factor());
}

pub fn increment_render_scale(
    renderer: &mut Renderer,
    surface_data: &mut SurfaceData,
    increase: bool,
    window: &winit::window::Window,
    ui_overlay: &mut IkariUiContainer<UiOverlay>,
) {
    let delta = 0.1;
    let change = if increase { delta } else { -delta };

    {
        let mut renderer_data_guard = renderer.data.lock();

        let render_scale = &mut renderer_data_guard.general_settings.render_scale;
        *render_scale = (*render_scale + change).clamp(0.1, 4.0);
        log::info!(
            "Render scale: {:?} ({:?}x{:?})",
            render_scale,
            (surface_data.surface_config.width as f32 * render_scale.sqrt()).round() as u32,
            (surface_data.surface_config.height as f32 * render_scale.sqrt()).round() as u32,
        );
    }

    let unscaled_framebuffer_size = winit::dpi::PhysicalSize::new(
        surface_data.surface_config.width,
        surface_data.surface_config.height,
    );
    // must call this after changing the render scale
    renderer.resize_surface(surface_data, unscaled_framebuffer_size);
    ui_overlay.resize(unscaled_framebuffer_size, window.scale_factor());
}

pub fn increment_exposure(renderer_data: &mut RendererData, increase: bool) {
    let delta = 0.05;
    let change = if increase { delta } else { -delta };
    let tone_mapping_exposure = &mut renderer_data.post_effect_settings.tone_mapping_exposure;
    *tone_mapping_exposure = (*tone_mapping_exposure + change).clamp(0.0, 20.0);
    log::info!("Exposure: {:?}", tone_mapping_exposure);
}

pub fn increment_old_bloom_threshold(renderer_data: &mut RendererData, increase: bool) {
    let delta = 0.05;
    let change = if increase { delta } else { -delta };
    let old_bloom_threshold = &mut renderer_data.post_effect_settings.old_bloom_threshold;
    *old_bloom_threshold = (*old_bloom_threshold + change).clamp(0.0, 20.0);
    log::info!("Old Bloom Threshold: {:?}", old_bloom_threshold);
}

#[profiling::function]
pub fn update_game_state(
    GameContext {
        game_state,
        engine_state,
        renderer,
        surface_data,
        window,
        elwt,
        ..
    }: GameContext<GameState>,
) {
    let renderer_data = renderer.data.clone();

    {
        let loaded_skyboxes = engine_state.asset_binder.loaded_skyboxes();
        let mut loaded_skyboxes_guard = loaded_skyboxes.lock();
        let asset_ids_guard = game_state.asset_ids.lock();

        if let Some(asset_id) = asset_ids_guard.skybox {
            if let Entry::Occupied(entry) = loaded_skyboxes_guard.entry(asset_id) {
                let (_, skybox) = entry.remove_entry();
                renderer.set_skybox(SkyboxSlot::Two, skybox);
            }
        }
    }

    {
        let loaded_scenes = engine_state.asset_binder.loaded_scenes();
        let mut loaded_assets_guard = loaded_scenes.lock();
        let asset_ids_guard = game_state.asset_ids.lock();
        let mut renderer_data_guard = renderer_data.lock();

        if let (Some(_gunshot_sound_index), Some(camera_node_id)) = (
            game_state.gunshot_sound_index,
            renderer_data_guard.camera_node_id,
        ) {
            if let Some(asset_id) = asset_ids_guard.gun {
                if let Entry::Occupied(entry) = loaded_assets_guard.entry(asset_id) {
                    let (_, other_loaded_scene) = entry.remove_entry();
                    let revolver_scene_node_count = other_loaded_scene.scene.node_count();

                    engine_state
                        .scene
                        .merge_scene(&mut renderer_data_guard, other_loaded_scene);

                    let merged_scene_node_count = engine_state.scene.node_count();
                    let node_id = engine_state
                        .scene
                        .nodes()
                        .nth(merged_scene_node_count - revolver_scene_node_count)
                        .unwrap()
                        .id();
                    let animation_index = engine_state.scene.animations.len() - 1;
                    game_state.revolver = Some(Revolver::new(
                        &mut engine_state.scene,
                        camera_node_id,
                        node_id,
                        animation_index,
                        // revolver model
                        // TransformBuilder::new()
                        //     .position(Vec3::new(0.23, -0.09, -0.81))
                        //     .rotation(
                        //         Quat::from_axis_angle(
                        //             Vec3::new(0.0, 1.0, 0.0),
                        //             180.0_f32.to_radians(),
                        //         ) * Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.1),
                        //     )
                        //     .scale(0.17f32 * Vec3::new(1.0, 1.0, 1.0))
                        //     .build(),
                        // colt python model
                        TransformBuilder::new()
                            .position(Vec3::new(0.19, -0.13, -0.75))
                            .rotation(
                                Quat::from_axis_angle(
                                    Vec3::new(0.0, 1.0, 0.0),
                                    180.0_f32.to_radians(),
                                ) * Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.1),
                            )
                            .scale(2.0f32 * Vec3::new(1.0, 1.0, 1.0))
                            .build(),
                    ));
                }
            }
        }

        if let Some(asset_id) = asset_ids_guard.forest {
            if let Entry::Occupied(entry) = loaded_assets_guard.entry(asset_id) {
                let (_, mut other_loaded_scene) = entry.remove_entry();
                // hack to get the terrain to be at the same height as the ground.
                let node_has_parent: Vec<_> = other_loaded_scene
                    .scene
                    .nodes()
                    .map(|node| node.parent_id.is_some())
                    .collect();
                for (i, node) in other_loaded_scene.scene.nodes_mut().enumerate() {
                    if node_has_parent[i] {
                        continue;
                    }
                    node.transform
                        .set_position(node.transform.position() + Vec3::new(0.0, 29.0, 0.0));
                }
                engine_state
                    .scene
                    .merge_scene(&mut renderer_data_guard, other_loaded_scene);

                if REMOVE_LARGE_OBJECTS_FROM_FOREST {
                    let node_ids: Vec<_> =
                        engine_state.scene.nodes().map(|node| node.id()).collect();
                    for node_id in node_ids {
                        if let Some(sphere) = engine_state
                            .scene
                            .get_node_bounding_sphere(node_id, &renderer_data_guard)
                        {
                            if sphere.radius > 10.0 {
                                engine_state.scene.remove_node(node_id);
                            }
                        }
                    }
                }
            }
        }

        if let Some(asset_id) = asset_ids_guard.legendary_robot {
            if let Entry::Occupied(entry) = loaded_assets_guard.entry(asset_id) {
                let (_, mut other_loaded_scene) = entry.remove_entry();
                if let Some(jump_up_animation) = other_loaded_scene
                    .scene
                    .animations
                    .iter_mut()
                    .find(|animation| animation.name == Some(String::from("jump_up_root_motion")))
                {
                    jump_up_animation.speed = 0.25;
                    jump_up_animation.state.is_playing = true;
                    jump_up_animation.state.loop_type = LoopType::Wrap;
                }
                engine_state
                    .scene
                    .merge_scene(&mut renderer_data_guard, other_loaded_scene);
            }
        }

        if let Some(asset_id) = asset_ids_guard.test_level {
            if let Entry::Occupied(entry) = loaded_assets_guard.entry(asset_id) {
                let (_, other_loaded_scene) = entry.remove_entry();
                let skip_nodes = engine_state.scene.node_count();
                engine_state
                    .scene
                    .merge_scene(&mut renderer_data_guard, other_loaded_scene);

                let test_level_node_ids: Vec<_> = engine_state
                    .scene
                    .nodes()
                    .skip(skip_nodes)
                    .map(|node| node.id())
                    .collect();
                for node_id in test_level_node_ids {
                    if let Some(_mesh) = engine_state
                        .scene
                        .get_node_mut(node_id)
                        .unwrap()
                        .visual
                        .as_mut()
                    {
                        // mesh.wireframe = true;
                    }
                    let transform =
                        &mut engine_state.scene.get_node_mut(node_id).unwrap().transform;
                    transform.set_position(transform.position() + Vec3::new(0.0, 25.0, 0.0));
                    add_static_box(
                        &mut engine_state.physics_state,
                        &engine_state.scene,
                        &renderer_data_guard,
                        node_id,
                    );
                }
            }
        }

        for asset_id in asset_ids_guard.anonymous_scenes.iter().copied() {
            if let Entry::Occupied(entry) = loaded_assets_guard.entry(asset_id) {
                let (_, mut other_loaded_scene) = entry.remove_entry();
                for animation in other_loaded_scene.scene.animations.iter_mut() {
                    animation.state.is_playing = true;
                    animation.state.loop_type = LoopType::Wrap;
                }
                engine_state
                    .scene
                    .merge_scene(&mut renderer_data_guard, other_loaded_scene);
            }
        }
    }

    {
        let mut loaded_audio_guard = engine_state.asset_loader.loaded_audio.lock();
        let asset_ids_guard = game_state.asset_ids.lock();

        if let Some(asset_id) = asset_ids_guard.bgm {
            if let Entry::Occupied(entry) = loaded_audio_guard.entry(asset_id) {
                let (_, bgm_sound_index) = entry.remove_entry();
                game_state.bgm_sound_index = Some(bgm_sound_index);

                let audio_manager_clone = engine_state.audio_manager.clone();
                let bgm_sound_index_clone = bgm_sound_index;
                ikari::thread::spawn(move || {
                    // #[cfg(not(target_arch = "wasm32"))]
                    // ikari::thread::sleep(ikari::time::Duration::from_secs_f32(5.0));

                    let mut audio_manager_guard = audio_manager_clone.lock();
                    audio_manager_guard.play_sound(bgm_sound_index_clone);
                });
            }
        }
        if let Some(asset_id) = asset_ids_guard.gunshot {
            if let Entry::Occupied(entry) = loaded_audio_guard.entry(asset_id) {
                let (_, gunshot_sound_index) = entry.remove_entry();
                game_state.gunshot_sound_index = Some(gunshot_sound_index);
                // audio_manager_guard.set_sound_volume(gunshot_sound_index, 0.001);
            }
        }
    }

    if game_state.character.is_none() {
        let legendary_robot_root_node_id = engine_state
            .scene
            .nodes()
            .find(|node| node.name == Some(String::from("robot")))
            .map(|legendary_robot_root_node| legendary_robot_root_node.id());

        game_state.character = legendary_robot_root_node_id.map(|legendary_robot_root_node_id| {
            engine_state
                .scene
                .get_node_mut(legendary_robot_root_node_id)
                .unwrap()
                .transform
                .set_position(Vec3::new(2.0, 0.0, 0.0));

            let legendary_robot_skin_index = 0;

            Character::new(
                &mut engine_state.scene,
                &mut engine_state.physics_state,
                &renderer.constant_data,
                legendary_robot_root_node_id,
                legendary_robot_skin_index,
            )
        });
    }

    // "src/models/gltf/free_low_poly_forest/scene.gltf"

    let time_tracker = engine_state.time();
    let global_time_seconds = time_tracker
        .global_time()
        .map(|global_time| global_time.as_secs_f32())
        .unwrap_or_default();

    // results in ~60 state changes per second
    let min_update_timestep_seconds = 1.0 / 60.0;
    // if frametime takes longer than this, we give up on trying to catch up completely
    // prevents the game from getting stuck in a spiral of death
    let max_delay_catchup_seconds = 0.25;
    let mut frame_time_seconds = time_tracker
        .last_frame_times()
        .map(|last_frame_time| last_frame_time.1.total.as_secs_f64())
        .unwrap_or_default();
    if frame_time_seconds > max_delay_catchup_seconds {
        frame_time_seconds = max_delay_catchup_seconds;
    }
    game_state.state_update_time_accumulator += frame_time_seconds;

    engine_state.physics_state.step();

    game_state
        .player_controller
        .update(&mut engine_state.physics_state);

    let new_player_transform = game_state
        .player_controller
        .transform(&engine_state.physics_state);
    if let Some(camera_node_id) = renderer_data.lock().camera_node_id {
        if let Some(player_transform) = engine_state.scene.get_node_mut(camera_node_id) {
            player_transform.transform = new_player_transform;
        }
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
        .map(|(prev_ball, next_ball)| prev_ball.lerp(next_ball, alpha as f32))
        .collect();
    game_state
        .ball_node_ids
        .iter()
        .zip(game_state.actual_balls.iter())
        .for_each(|(node_id, ball)| {
            if let Some(node) = engine_state.scene.get_node_mut(*node_id) {
                node.transform = ball.transform;
            }
        });

    if let Some(point_light_0) = engine_state.scene.point_lights.get_mut(0) {
        let t = {
            let mut t = engine_state
                .scene
                .animations
                .iter_mut()
                .find(|animation| animation.name == Some(String::from("jump_up_root_motion")))
                .map(|animation| animation.state.current_time_seconds * 2.0)
                .unwrap_or(global_time_seconds * 0.5);

            if CREATE_POINT_SHADOW_MAP_DEBUG_OBJECTS {
                // t = 8.14; // puts the shadow at the corner of the box
                t = game_state.player_controller.speed * 0.05
            }

            t
        };

        point_light_0.color = POINT_LIGHT_COLOR.lerp(POINT_LIGHT_COLOR_B, (2.0 * t).sin());

        let node_id = point_light_0.node_id;
        if let Some(node) = engine_state.scene.get_node_mut(node_id) {
            let center = if CREATE_POINT_SHADOW_MAP_DEBUG_OBJECTS {
                let cube_radius = 4.0;
                Vec3::new(20.0, cube_radius, -4.5)
            } else {
                Vec3::new(0.0, 6.5, 0.0)
            };

            node.transform.set_position(
                center
                    + Vec3::new(
                        (t * 2.0).cos() * (t * 0.25).cos(),
                        2.0 * (t * 1.0).cos(),
                        (t * 2.0).sin() * (t * 0.5).sin(),
                    ),
            );
        }
    }

    if let Some(_point_light_1) = engine_state.scene.point_lights.get_mut(1) {
        // _point_light_1.color = lerp_vec(
        //     LIGHT_COLOR_B,
        //     LIGHT_COLOR_A,
        //     (global_time_seconds * 2.0).sin(),
        // );
    }

    // sync unlit mesh config with point light component
    let point_lights_with_ids: Vec<_> = game_state
        .point_light_node_ids
        .iter()
        .cloned()
        .zip(engine_state.scene.point_lights.iter().cloned())
        .collect();
    point_lights_with_ids
        .iter()
        .for_each(|(node_id, point_light)| {
            if let Some(GameNodeVisual {
                material: Material::Unlit { ref mut color },
                ..
            }) = engine_state
                .scene
                .get_node_mut(*node_id)
                .and_then(|node| node.visual.as_mut())
            {
                *color = point_light.color * point_light.intensity * 10.0;
            }
        });

    let directional_light_0 =
        engine_state
            .scene
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

                DirectionalLight {
                    direction: Vec3::new(direction.x, direction.y + 0.00001, direction.z),
                    ..*directional_light_0
                }
            });
    if let Some(_directional_light_0) = directional_light_0 {
        // engine_state.scene.directional_lights[0] = directional_light_0;
    }

    // rotate the test object
    let rotational_displacement =
        Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), frame_time_seconds as f32 / 5.0);
    if let Some(node) = engine_state
        .scene
        .get_node_mut(game_state.test_object_node_id)
    {
        node.transform
            .set_rotation(rotational_displacement * node.transform.rotation());
    }

    // remove physics balls over time
    game_state.ball_spawner_acc += frame_time_seconds;
    let rate = 0.1; // lower value spawns balls more quickly
    let prev_ball_count = game_state.physics_balls.len();
    while game_state.ball_spawner_acc > rate {
        // let new_ball = BallComponent::rand();
        // let new_ball_transform = new_ball.transform;
        // game_state.next_balls.push(new_ball);
        // engine_state.scene.nodes.push(
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
        //     .push(engine_state.scene.nodes.len() - 1);
        // if let Some(physics_ball) = game_state.physics_balls.pop() {
        //     physics_ball.destroy(&mut engine_state.scene, &mut game_state.physics_state);
        // }
        // game_state.physics_balls.push(PhysicsBall::new_random(
        //     &mut engine_state.scene,
        //     &mut game_state.physics_state,
        //     GameNodeMesh::from_pbr_mesh_index(game_state.ball_pbr_mesh_index),
        // ));
        game_state.ball_spawner_acc -= rate;
    }
    let new_ball_count = game_state.physics_balls.len();
    if prev_ball_count != new_ball_count {
        // logger_log(&format!("Ball count: {:?}", new_ball_count));
    }

    let physics_state = &mut engine_state.physics_state;
    let ball_body = &physics_state.rigid_body_set[game_state.bouncing_ball_body_handle];
    if let Some(node) = engine_state
        .scene
        .get_node_mut(game_state.bouncing_ball_node_id)
    {
        node.transform.apply_isometry(*ball_body.position());
    }

    physics_state.integration_parameters.dt = frame_time_seconds;
    game_state
        .physics_balls
        .iter()
        .for_each(|physics_ball| physics_ball.update(&mut engine_state.scene, physics_state));

    if let Some(crosshair_node) = game_state
        .crosshair_node_id
        .and_then(|crosshair_node_id| engine_state.scene.get_node_mut(crosshair_node_id))
    {
        crosshair_node.transform = new_player_transform
            * TransformBuilder::new()
                .position(Vec3::new(0.0, 0.0, -1.0))
                .rotation(Quat::from_axis_angle(
                    Vec3::new(0.0, 1.0, 0.0),
                    90.0_f32.to_radians(),
                ))
                .scale(
                    (1080.0 / surface_data.surface_config.height as f32)
                        * 0.06
                        * Vec3::new(1.0, 1.0, 1.0),
                )
                .build();
    }

    if let Some(revolver) = game_state.revolver.as_mut() {
        revolver.update(
            game_state.player_controller.view_direction,
            &mut engine_state.scene,
        );

        if game_state.player_controller.mouse_button_pressed
            && revolver.fire(&mut engine_state.scene)
        {
            /* if let Some(bgm_sound_index) = game_state.bgm_sound_index {
                if time_tracker.global_time_seconds() > 30.0 {
                    game_state
                        .audio_manager
                        .lock()
                        .play_sound(bgm_sound_index);
                    game_state.bgm_sound_index = None;
                }
            } */

            if let Some(gunshot_sound_index) = game_state.gunshot_sound_index {
                {
                    let mut audio_manager_guard = engine_state.audio_manager.lock();
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

            let player_position = game_state
                .player_controller
                .position(&engine_state.physics_state);
            let direction_vec = game_state.player_controller.view_direction.to_vector();
            let ray = Ray::new(
                point![
                    player_position.x as f64,
                    player_position.y as f64,
                    player_position.z as f64
                ],
                vector![
                    direction_vec.x as f64,
                    direction_vec.y as f64,
                    direction_vec.z as f64
                ],
            );
            let max_distance = ARENA_SIDE_LENGTH as f64 * 10.0;
            let solid = true;
            if let Some((collider_handle, collision_point_distance)) =
                engine_state.physics_state.query_pipeline.cast_ray(
                    &engine_state.physics_state.rigid_body_set,
                    &engine_state.physics_state.collider_set,
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

                if let Some(rigid_body_handle) = engine_state
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
                        // ball.toggle_wireframe(&mut engine_state.scene);
                        ball.destroy(&mut engine_state.scene, &mut engine_state.physics_state);
                        game_state.physics_balls.remove(ball_index);
                    }
                }
                if let Some(character) = game_state.character.as_mut() {
                    character.handle_hit(&mut engine_state.scene, collider_handle);
                }
            }
        }
    }

    // step animatons
    let scene = &mut engine_state.scene;
    if game_state.is_playing_animations {
        step_animations(scene, frame_time_seconds)
    }

    if let Some(character) = game_state.character.as_mut() {
        character.update(scene, &mut engine_state.physics_state);
    }

    {
        profiling::scope!("Sync UI");

        let mut renderer_data_guard = renderer.data.lock();

        if let Some((instants, durations)) = engine_state.time().last_frame_times() {
            game_state
                .ui_overlay
                .queue_message(Message::FrameCompleted(FrameStats {
                    instants,
                    durations,
                    gpu_timer_query_results: renderer.process_profiler_frame().unwrap_or_default(),
                    culling_stats: renderer_data_guard.last_frame_culling_stats.clone(),
                }));
        }

        {
            let audio_manager_guard = engine_state.audio_manager.lock();
            for sound_index in audio_manager_guard.sound_indices() {
                let file_path = audio_manager_guard
                    .get_sound_file_path(sound_index)
                    .unwrap();
                let length_seconds = audio_manager_guard
                    .get_sound_length_seconds(sound_index)
                    .unwrap();
                let pos_seconds = audio_manager_guard
                    .get_sound_pos_seconds(sound_index)
                    .unwrap();
                let buffered_to_pos_seconds = audio_manager_guard
                    .get_sound_buffered_to_pos_seconds(sound_index)
                    .unwrap();

                game_state
                    .ui_overlay
                    .queue_message(Message::AudioSoundStatsChanged((
                        file_path.clone(),
                        AudioSoundStats {
                            length_seconds,
                            pos_seconds,
                            buffered_to_pos_seconds,
                        },
                    )));
            }
        }

        let camera_position = game_state
            .player_controller
            .position(&engine_state.physics_state);
        let camera_view_direction = game_state.player_controller.view_direction;
        game_state
            .ui_overlay
            .queue_message(Message::CameraPoseChanged((
                camera_position,
                camera_view_direction,
            )));

        let ui_state = game_state.ui_overlay.get_state();

        if ui_state.was_exit_button_pressed {
            elwt.exit();
        }

        renderer_data_guard.general_settings.enable_depth_prepass =
            ui_state.general_settings.enable_depth_prepass;

        renderer_data_guard.camera_settings.fov_x = ui_state.camera_settings.fov_x;
        renderer_data_guard.camera_settings.near_plane_distance =
            ui_state.camera_settings.near_plane_distance;
        renderer_data_guard.camera_settings.far_plane_distance =
            ui_state.camera_settings.far_plane_distance;

        renderer_data_guard.post_effect_settings.bloom_type =
            ui_state.post_effect_settings.bloom_type;
        renderer_data_guard.post_effect_settings.new_bloom_radius =
            ui_state.post_effect_settings.new_bloom_radius;
        renderer_data_guard.post_effect_settings.new_bloom_intensity =
            ui_state.post_effect_settings.new_bloom_intensity;

        renderer_data_guard.shadow_settings.enable_soft_shadows =
            ui_state.shadow_settings.enable_soft_shadows;
        renderer_data_guard
            .shadow_settings
            .shadow_small_object_culling_size_pixels = ui_state
            .shadow_settings
            .shadow_small_object_culling_size_pixels;
        renderer_data_guard.shadow_settings.soft_shadow_factor =
            ui_state.shadow_settings.soft_shadow_factor;
        renderer_data_guard
            .shadow_settings
            .soft_shadows_max_distance = ui_state.shadow_settings.soft_shadows_max_distance;
        renderer_data_guard.shadow_settings.shadow_bias = ui_state.shadow_settings.shadow_bias;
        renderer_data_guard.shadow_settings.soft_shadow_grid_dims =
            ui_state.shadow_settings.soft_shadow_grid_dims;

        renderer_data_guard.debug_settings.record_culling_stats =
            ui_state.debug_settings.record_culling_stats;
        renderer_data_guard.debug_settings.enable_shadow_debug =
            ui_state.debug_settings.enable_shadow_debug;
        renderer_data_guard.debug_settings.enable_cascade_debug =
            ui_state.debug_settings.enable_cascade_debug;
        renderer_data_guard.debug_settings.draw_culling_frustum =
            ui_state.debug_settings.draw_culling_frustum;
        renderer_data_guard
            .debug_settings
            .draw_point_light_culling_frusta =
            ui_state.debug_settings.draw_point_light_culling_frusta;
        renderer_data_guard
            .debug_settings
            .draw_directional_light_culling_frusta = ui_state
            .debug_settings
            .draw_directional_light_culling_frusta;

        renderer.set_skybox_weights([
            1.0 - ui_state.post_effect_settings.skybox_weight,
            ui_state.post_effect_settings.skybox_weight,
        ]);
        renderer.set_vsync(ui_state.general_settings.enable_vsync, surface_data);

        drop(renderer_data_guard);

        renderer.set_culling_frustum_lock(
            engine_state,
            &surface_data.surface_config,
            ui_state.debug_settings.culling_frustum_lock_mode,
        );
    }

    game_state.ui_overlay.update(window);

    {
        let viewport_dims = (
            (window.inner_size().width as f64 / window.scale_factor()) as u32,
            (window.inner_size().height as f64 / window.scale_factor()) as u32,
        );
        game_state
            .ui_overlay
            .queue_message(Message::ViewportDimsChanged(viewport_dims));

        let cursor_pos = winit::dpi::PhysicalPosition::new(
            game_state.ui_overlay.cursor_position.x / window.inner_size().width as f64,
            game_state.ui_overlay.cursor_position.y / window.inner_size().height as f64,
        );
        game_state
            .ui_overlay
            .queue_message(Message::CursorPosChanged(cursor_pos));
    }

    let is_showing_options_menu = game_state.ui_overlay.get_state().is_showing_options_menu;
    let is_showing_cursor_marker = game_state
        .ui_overlay
        .get_state()
        .debug_settings
        .is_showing_cursor_marker;
    game_state.player_controller.update_cursor_grab(
        !is_showing_options_menu && !is_showing_cursor_marker,
        window,
    );
    game_state
        .player_controller
        .set_is_controlling_game(!is_showing_options_menu);
}

fn add_static_box(
    physics_state: &mut PhysicsState,
    scene: &Scene,
    renderer_data: &RendererData,
    node_id: GameNodeId,
) {
    let collider_handles = physics_state
        .static_box_set
        .entry(node_id)
        .or_insert(vec![]);

    if let Some(node) = scene.get_node(node_id) {
        if let Some((visual, transform)) = node
            .visual
            .as_ref()
            .zip(scene.get_global_transform_for_node(node_id))
        {
            let transform_decomposed = transform.decompose();
            let bounding_box = renderer_data.binded_meshes[visual.mesh_index].bounding_box;
            let base_scale = (bounding_box.max - bounding_box.min) / 2.0;
            let base_position = (bounding_box.max + bounding_box.min) / 2.0;
            let scale = Vec3::new(
                base_scale.x * transform_decomposed.scale.x,
                base_scale.y * transform_decomposed.scale.y,
                base_scale.z * transform_decomposed.scale.z,
            );
            let position_rotated = {
                let rotated = Mat4::from_quat(transform_decomposed.rotation)
                    * Vec4::new(base_position.x, base_position.y, base_position.z, 1.0);
                Vec3::new(rotated.x, rotated.y, rotated.z)
            };
            let position = Vec3::new(
                position_rotated.x + transform_decomposed.position.x,
                position_rotated.y + transform_decomposed.position.y,
                position_rotated.z + transform_decomposed.position.z,
            );
            let rotation = transform_decomposed.rotation;
            let mut collider =
                ColliderBuilder::cuboid(scale.x as f64, scale.y as f64, scale.z as f64)
                    .collision_groups(
                        InteractionGroups::all()
                            .with_memberships(!COLLISION_GROUP_PLAYER_UNSHOOTABLE),
                    )
                    .friction(1.0)
                    .restitution(1.0)
                    .build();
            collider.set_position(Isometry::from_parts(
                nalgebra::Translation3::new(
                    position.x as f64,
                    position.y as f64,
                    position.z as f64,
                ),
                nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    rotation.w as f64,
                    rotation.x as f64,
                    rotation.y as f64,
                    rotation.z as f64,
                )),
            ));
            collider_handles.push(physics_state.collider_set.insert(collider));
        }
    }
}
