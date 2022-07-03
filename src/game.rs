use super::*;

use anyhow::Result;

pub fn init_scene(
    base_renderer_state: &mut BaseRendererState,
    logger: &mut Logger,
) -> Result<(GameScene, RenderScene)> {
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

    let (document, buffers, images) = gltf::import(gltf_path)?;
    validate_animation_property_counts(&document, logger);
    build_scene(base_renderer_state, (&document, &buffers, &images))
}

pub fn update_game_state(game_state: &mut GameState, _renderer_state: &RendererState) {
    let _time_tracker = game_state.time();
}
