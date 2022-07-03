use super::*;

use anyhow::Result;
use cgmath::{Rad, Vector3};

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

pub fn update_game_state(
    game_state: &mut GameState,
    renderer_state: &mut RendererState,
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

    let new_point_light_0 = game_state.point_lights.get(0).map(|point_light_0| {
        let mut transform = point_light_0.transform;
        transform.set_position(Vector3::new(
            // light_1.transform.position.get().x,
            1.5 * (global_time_seconds * 0.25 + std::f32::consts::PI).cos(),
            point_light_0.transform.position().y - frame_time_seconds * 0.25,
            1.5 * (global_time_seconds * 0.25 + std::f32::consts::PI).sin(),
            // light_1.transform.position.get().z,
        ));
        let color = lerp_vec(
            LIGHT_COLOR_A,
            LIGHT_COLOR_B,
            (global_time_seconds * 2.0).sin(),
        );

        PointLightComponent {
            transform,
            color,
            intensity: point_light_0.intensity,
        }
    });
    if let Some(new_point_light_0) = new_point_light_0 {
        game_state.point_lights[0] = new_point_light_0;
    }

    let new_point_light_1 = game_state.point_lights.get(1).map(|point_light_1| {
        let transform = point_light_1.transform;
        // transform.set_position(Vector3::new(
        //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).cos(),
        //     transform.position.get().y,
        //     1.1 * (time_seconds * 0.25 + std::f32::consts::PI).sin(),
        // ));
        let color = lerp_vec(
            LIGHT_COLOR_B,
            LIGHT_COLOR_A,
            (global_time_seconds * 2.0).sin(),
        );

        PointLightComponent {
            transform,
            color,
            intensity: point_light_1.intensity,
        }
    });
    if let Some(new_point_light_1) = new_point_light_1 {
        game_state.point_lights[1] = new_point_light_1;
    }

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

    let rotational_displacement =
        make_quat_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(frame_time_seconds / 5.0));
    let curr = game_state.test_object_instances[0].transform.rotation();
    game_state.test_object_instances[0]
        .transform
        .set_rotation(rotational_displacement * curr);

    // logger.log(&format!("Frame time: {:?}", frame_time_seconds));
    // logger.log(&format!(
    //     "state_update_time_accumulator: {:?}",
    //     game_state.state_update_time_accumulator
    // ));

    if global_time_seconds > 5.0 && !game_state.actual_balls.is_empty() {
        let first_ball = game_state.actual_balls[0].clone();
        let first_ball_transform = first_ball.instance.transform;

        let sphere_mesh = renderer_state.sphere_mesh.take().unwrap();

        renderer_state
            .scene
            .buffers
            .binded_mesh_data
            .push(BindedMeshData {
                vertex_buffer: BufferAndLength {
                    buffer: sphere_mesh.vertex_buffer,
                    length: sphere_mesh._num_vertices.try_into().unwrap(),
                },
                index_buffer: Some(BufferAndLength {
                    buffer: sphere_mesh.index_buffer,
                    length: sphere_mesh.num_indices.try_into().unwrap(),
                }),
                instance_buffer: BufferAndLength {
                    buffer: sphere_mesh.instance_buffer,
                    length: 1,
                },
                dynamic_material_params: Default::default(),
                textures_bind_group: sphere_mesh.textures_bind_group,
                alpha_mode: AlphaMode::Opaque,
                primitive_mode: PrimitiveMode::Triangles,
            });

        game_state.scene.nodes.push(
            GameNodeBuilder::new()
                .transform(first_ball_transform)
                // .binded_mesh_indices(Some(vec![
                //     renderer_state.scene.buffers.binded_mesh_data.len() - 1,
                // ]))
                .build(),
        );

        // creates a binded mesh data
        // let binded_mesh_index = renderer_state.bind_basic_mesh(basic_mesh, pbr_textures);
        // registers a game node
        // game_state.scene.nodes.push(GameNode::default());
        // let new_node_index = game_state.scene.nodes.len() - 1;
        // let mut new_node = game_state.scene.nodes[new_node_index]

        game_state.actual_balls = vec![];
        game_state.prev_balls = vec![];
        game_state.next_balls = vec![];
    } else {
        let ball_node_index = game_state.scene.nodes.len() - 1;
        let ball_node = &mut game_state.scene.nodes[ball_node_index];
        ball_node.transform.set_position(Vector3::new(
            ball_node.transform.position().x,
            ball_node.transform.position().y + 0.5 * frame_time_seconds,
            ball_node.transform.position().z,
        ));
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
