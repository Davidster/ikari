// Vertex shader
struct CameraUniform {
    proj: mat4x4<f32>;
    view: mat4x4<f32>;
    rotation_only_view: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
    // TODO: can these be removed?
    [[location(1)]] object_normal: vec3<f32>;
    [[location(2)]] object_tex_coords: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec3<f32>;
};

[[stage(vertex)]]
fn vs_main(
    vshader_input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let clip_position = camera.proj * camera.rotation_only_view * vec4<f32>(vshader_input.object_position, 1.0);
    out.clip_position = clip_position.xyww;
    out.world_position = vshader_input.object_position;
    return out;
}

[[group(0), binding(0)]]
var r_texture: texture_cube<f32>;

[[group(0), binding(1)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(r_texture, r_sampler, in.world_position);
}