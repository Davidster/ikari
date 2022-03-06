// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

struct ModelTransformUniform {
    value: mat4x4<f32>;
};
[[group(2), binding(0)]]
var<uniform> model_transform:ModelTransformUniform;

struct ModelNormalRotationUniform {
    value: mat4x4<f32>;
};
[[group(3), binding(0)]]
var<uniform> normal_rotation:ModelNormalRotationUniform;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
    [[location(1)]] object_normal: vec3<f32>;
    [[location(2)]] object_tex_coords: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec3<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    vshader_input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.world_normal = vshader_input.object_normal;
    
    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let model_view_matrix = camera.view_proj * model_transform.value;
    let world_position = model_transform.value * object_position;

    out.world_position = world_position.xyz;
    out.clip_position = model_view_matrix * object_position;

    // out.world_normal = inverse(model_view_matrix) * vshader_input.object_normal;
    out.world_normal = (normal_rotation.value * vec4<f32>(vshader_input.object_normal, 0.0)).xyz;

    out.tex_coords = vshader_input.object_tex_coords;

    return out;
}

// Fragment shader

[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let to_light_vec = normalize(vec3<f32>(1.0, 1.0, 5.0) - in.world_position);
    let light_intensity = max(dot(in.world_normal, to_light_vec), 0.0);
    let albedo = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let max_light_intensity = 1.0;
    let ambient_light = 0.05;
    let final_light_intensity = ambient_light + (light_intensity * max_light_intensity);
    let final = final_light_intensity * albedo;
    let some_color = vec3<f32>(0.5, 0.5, 0.5);
    return final;
}
 