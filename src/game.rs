use super::*;

use anyhow::Result;
use cgmath::{Rad, Vector3};
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

pub const INITIAL_RENDER_SCALE: f32 = 1.0;
pub const INITIAL_TONE_MAPPING_EXPOSURE: f32 = 0.5;
pub const INITIAL_BLOOM_THRESHOLD: f32 = 0.8;
pub const INITIAL_BLOOM_RAMP_SIZE: f32 = 0.2;
pub const ARENA_SIDE_LENGTH: f32 = 50.0;
pub const LIGHT_COLOR_A: Vector3<f32> = Vector3::new(0.996, 0.973, 0.663);
pub const LIGHT_COLOR_B: Vector3<f32> = Vector3::new(0.25, 0.973, 0.663);

#[allow(clippy::let_and_return)]
fn get_gltf_path() -> &'static str {
    // let gltf_path = "/home/david/Downloads/adamHead/adamHead.gltf";
    // let gltf_path = "/home/david/Programming/glTF-Sample-Models/2.0/VC/glTF/VC.gltf";
    // let gltf_path = "./src/models/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf";
    // let gltf_path = "./src/models/gltf/SimpleMeshes/SimpleMeshes.gltf";
    // let gltf_path = "./src/models/gltf/Triangle/Triangle.gltf";
    // let gltf_path = "./src/models/gltf/TriangleWithoutIndices/TriangleWithoutIndices.gltf";
    // let gltf_path = "./src/models/gltf/Sponza/Sponza.gltf";
    // let gltf_path = "./src/models/gltf/EnvironmentTest/EnvironmentTest.gltf";
    // let gltf_path = "./src/models/gltf/Arrow/Arrow.gltf";
    // let gltf_path = "./src/models/gltf/DamagedHelmet/DamagedHelmet.gltf";
    // let gltf_path = "./src/models/gltf/VertexColorTest/VertexColorTest.gltf";
    // let gltf_path =
    //     "/home/david/Programming/glTF-Sample-Models/2.0/BoomBoxWithAxes/glTF/BoomBoxWithAxes.gltf";
    // let gltf_path =
    //     "./src/models/gltf/TextureLinearInterpolationTest/TextureLinearInterpolationTest.glb";
    // let gltf_path = "../glTF-Sample-Models/2.0/RiggedFigure/glTF/RiggedFigure.gltf";
    // let gltf_path = "../glTF-Sample-Models/2.0/RiggedSimple/glTF/RiggedSimple.gltf";
    // let gltf_path = "../glTF-Sample-Models/2.0/CesiumMan/glTF/CesiumMan.gltf";
    // let gltf_path = "../glTF-Sample-Models/2.0/Fox/glTF/Fox.gltf";
    let gltf_path = "../glTF-Sample-Models/2.0/BrainStem/glTF/BrainStem.gltf";
    // let gltf_path =
    //     "/home/david/Programming/glTF-Sample-Models/2.0/BoxAnimated/glTF/BoxAnimated.gltf";
    // let gltf_path = "/home/david/Programming/glTF-Sample-Models/2.0/InterpolationTest/glTF/InterpolationTest.gltf";
    // let gltf_path = "./src/models/gltf/VC/VC.gltf";
    // let gltf_path =
    //     "../glTF-Sample-Models-master/2.0/InterpolationTest/glTF/InterpolationTest.gltf";
    gltf_path
}

pub fn init_game_state(
    mut scene: GameScene,
    renderer_state: &mut RendererState,
) -> Result<GameState> {
    let sphere_mesh = BasicMesh::new("./src/models/sphere.obj")?;
    let plane_mesh = BasicMesh::new("./src/models/plane.obj")?;
    let _cube_mesh = BasicMesh::new("./src/models/cube.obj")?;

    let camera_controller = CameraController::new(6.0, Camera::new((0.0, 3.0, 4.0).into()));
    scene.nodes.push(GameNode::default());
    let camera_node_index = scene.nodes.len() - 1;

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
    let point_light_node_indices: Vec<usize> =
        (scene.nodes.len()..(scene.nodes.len() + point_lights.len())).collect();
    let mut point_light_components: Vec<PointLightComponent> = Vec::new();
    for (transform, color, intensity) in &point_lights {
        scene.nodes.push(
            GameNodeBuilder::new()
                .mesh(Some(GameNodeMesh::Unlit {
                    mesh_indices: vec![point_light_unlit_mesh_index],
                    color: color * *intensity,
                }))
                .transform(*transform)
                .build(),
        );
        point_light_components.push(PointLightComponent {
            node_index: scene.nodes.len() - 1,
            color: LIGHT_COLOR_A,
            intensity: *intensity,
        });
    }

    // rotate the animated character 90 deg
    // if let Some(node_0) = scene.nodes.get_mut(0) {
    //     node_0.transform.set_rotation(make_quat_from_axis_angle(
    //         Vector3::new(0.0, 1.0, 0.0),
    //         Deg(90.0).into(),
    //     ));
    // }

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
            diffuse: Some(&earth_texture),
            normal: Some(&earth_normal_map),
            metallic_roughness: Some(&test_object_metallic_roughness_map),
            ..Default::default()
        },
        Default::default(),
    )?;
    scene.nodes.push(
        GameNodeBuilder::new()
            .mesh(Some(GameNodeMesh::Pbr {
                mesh_indices: vec![test_object_pbr_mesh_index],
                material_override: None,
            }))
            .transform(
                TransformBuilder::new()
                    .position(Vector3::new(4.0, 10.0, 4.0))
                    .build(),
            )
            .build(),
    );
    let test_object_node_index = scene.nodes.len() - 1;

    // add floor to scene
    let checkerboard_texture_img = {
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
    let checkerboard_texture = Texture::from_decoded_image(
        &renderer_state.base.device,
        &renderer_state.base.queue,
        &checkerboard_texture_img,
        checkerboard_texture_img.dimensions(),
        Some("checkerboard_texture"),
        None,
        true,
        &texture::SamplerDescriptor(wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            ..texture::SamplerDescriptor::default().0
        }),
    )?;

    let floor_pbr_mesh_index = renderer_state.bind_basic_pbr_mesh(
        &plane_mesh,
        &PbrMaterial {
            diffuse: Some(&checkerboard_texture),
            ..Default::default()
        },
        Default::default(),
    )?;
    scene.nodes.push(
        GameNodeBuilder::new()
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

    let ball_count = 5;
    let balls: Vec<_> = (0..ball_count)
        .into_iter()
        .map(|_| BallComponent::rand())
        .collect();

    let ball_pbr_mesh_index = renderer_state.bind_basic_pbr_mesh(
        &sphere_mesh,
        &PbrMaterial {
            diffuse: Some(&mars_texture),
            ..Default::default()
        },
        Default::default(),
    )?;

    let ball_node_indices: Vec<usize> =
        (scene.nodes.len()..(scene.nodes.len() + ball_count)).collect();
    for ball in &balls {
        scene.nodes.push(
            GameNodeBuilder::new()
                .mesh(Some(GameNodeMesh::Pbr {
                    mesh_indices: vec![ball_pbr_mesh_index],
                    material_override: None,
                }))
                .transform(ball.transform)
                .build(),
        );
    }

    Ok(GameState {
        scene,
        time_tracker: None,
        state_update_time_accumulator: 0.0,

        camera_controller,
        camera_node_index,

        point_lights: point_light_components,
        point_light_node_indices,
        directional_lights,

        next_balls: balls.clone(),
        prev_balls: balls.clone(),
        actual_balls: balls,
        ball_node_indices,
        ball_pbr_mesh_index,

        ball_spawner_acc: 0.0,

        test_object_node_index,
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
                _ => {}
            }
        }
    }
    game_state
        .camera_controller
        .process_window_events(event, window, logger);
}

pub fn update_game_state(game_state: &mut GameState, logger: &mut Logger) {
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

    game_state.camera_controller.update(frame_time_seconds);
    game_state.scene.nodes[game_state.camera_node_index].transform = game_state
        .camera_controller
        .current_pose
        .to_transform()
        .into();

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
        .ball_node_indices
        .iter()
        .zip(game_state.actual_balls.iter())
        .for_each(|(node_index, ball)| {
            game_state.scene.nodes[*node_index].transform = ball.transform;
        });

    if let Some(point_light_0) = game_state.point_lights.get_mut(0) {
        point_light_0.color = lerp_vec(
            LIGHT_COLOR_A,
            LIGHT_COLOR_B,
            (global_time_seconds * 2.0).sin(),
        );
        let transform = &mut game_state.scene.nodes[point_light_0.node_index].transform;
        transform.set_position(Vector3::new(
            1.5 * (global_time_seconds * 0.25 + std::f32::consts::PI).cos(),
            transform.position().y - frame_time_seconds * 0.25,
            1.5 * (global_time_seconds * 0.25 + std::f32::consts::PI).sin(),
        ));
    }

    if let Some(point_light_1) = game_state.point_lights.get_mut(1) {
        point_light_1.color = lerp_vec(
            LIGHT_COLOR_B,
            LIGHT_COLOR_A,
            (global_time_seconds * 2.0).sin(),
        );
        // let transform = &mut game_state.scene.nodes[point_light_1.node_index].transform;
        // transform.set_position(Vector3::new(
        //     1.1 * (global_time_seconds * 0.25 + std::f32::consts::PI).cos(),
        //     transform.position().y,
        //     1.1 * (global_time_seconds * 0.25 + std::f32::consts::PI).sin(),
        // ));
    }

    // sync unlit mesh config with point light component
    game_state
        .point_light_node_indices
        .iter()
        .zip(game_state.point_lights.iter())
        .for_each(|(node_index, point_light)| {
            if let Some(GameNodeMesh::Unlit { ref mut color, .. }) =
                game_state.scene.nodes[*node_index].mesh
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
    let test_object_transform =
        &mut game_state.scene.nodes[game_state.test_object_node_index].transform;
    test_object_transform.set_rotation(rotational_displacement * test_object_transform.rotation());

    // logger.log(&format!("Frame time: {:?}", frame_time_seconds));
    // logger.log(&format!(
    //     "state_update_time_accumulator: {:?}",
    //     game_state.state_update_time_accumulator
    // ));

    game_state.ball_spawner_acc += frame_time_seconds;
    let rate = 0.1;
    let prev_ball_count = game_state.ball_node_indices.len();
    while game_state.ball_spawner_acc > rate {
        let new_ball = BallComponent::rand();
        let new_ball_transform = new_ball.transform;
        game_state.next_balls.push(new_ball);
        game_state.scene.nodes.push(
            GameNodeBuilder::new()
                .mesh(Some(GameNodeMesh::Pbr {
                    mesh_indices: vec![game_state.ball_pbr_mesh_index],
                    material_override: None,
                }))
                .transform(new_ball_transform)
                .build(),
        );
        game_state
            .ball_node_indices
            .push(game_state.scene.nodes.len() - 1);
        game_state.ball_spawner_acc -= rate;
    }
    let new_ball_count = game_state.ball_node_indices.len();
    if prev_ball_count != new_ball_count {
        // logger.log(&format!("Ball count: {:?}", new_ball_count));
    }
}

pub fn init_scene(
    base_renderer_state: &mut BaseRendererState,
    logger: &mut Logger,
) -> Result<(GameScene, RenderScene)> {
    let (document, buffers, images) = gltf::import(get_gltf_path())?;
    validate_animation_property_counts(&document, logger);
    build_scene(base_renderer_state, (&document, &buffers, &images))
}
