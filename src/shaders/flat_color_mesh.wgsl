struct CameraUniform {
    proj: mat4x4<f32>;
    view: mat4x4<f32>;
    rotation_only_view: mat4x4<f32>;
    position: vec4<f32>;
    near_plane_distance: f32;
    far_plane_distance: f32;
};
[[group(0), binding(0)]]
var<uniform> camera: CameraUniform;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
    [[location(1)]] object_normal: vec3<f32>;
    [[location(2)]] object_tex_coords: vec2<f32>;
    [[location(3)]] object_tangent: vec3<f32>;
    [[location(4)]] object_bitangent: vec3<f32>;
    [[location(5)]] object_color: vec4<f32>;
};

struct Instance {
    [[location(6)]]  model_transform_0: vec4<f32>;
    [[location(7)]]  model_transform_1: vec4<f32>;
    [[location(8)]]  model_transform_2: vec4<f32>;
    [[location(9)]]  model_transform_3: vec4<f32>;
    [[location(10)]]  color: vec4<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] color: vec4<f32>;
    [[location(1)]] vertex_color: vec4<f32>;
};

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    vshader_input: VertexInput,
    instance: Instance,
) -> VertexOutput {
    let model_transform = mat4x4<f32>(
        instance.model_transform_0,
        instance.model_transform_1,
        instance.model_transform_2,
        instance.model_transform_3,
    );

    var out: VertexOutput;

    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let camera_view_proj = camera.proj * camera.view;
    let model_view_matrix = camera_view_proj * model_transform;
    let clip_position = model_view_matrix * object_position;

    out.clip_position = clip_position;
    out.color = instance.color;
    out.vertex_color = vshader_input.object_color;
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    // out.color = vec4<f32>(0.996078431372549, 0.9725490196078431, 0.6627450980392157, 1.0);
    out.color = in.color * in.vertex_color;
    return out;
}