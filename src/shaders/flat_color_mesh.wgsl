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

struct LightUniform {
    position: vec4<f32>;
    color: vec4<f32>;
};
[[group(0), binding(1)]]
var<uniform> light: LightUniform;

struct ModelTransformUniform {
    value: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> model_transform: ModelTransformUniform;

struct ColorUniform {
    value: vec4<f32>;
};
[[group(2), binding(0)]]
var<uniform> color: ColorUniform;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
    [[location(1)]] object_normal: vec3<f32>;
    [[location(2)]] object_tex_coords: vec2<f32>;
    [[location(3)]] object_tangent: vec3<f32>;
    [[location(4)]] object_bitangent: vec3<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
};

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    vshader_input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    let object_position   = vec4<f32>(vshader_input.object_position, 1.0);
    let camera_view_proj  = camera.proj * camera.view;
    let model_view_matrix = camera_view_proj * model_transform.value;
    let clip_position     = model_view_matrix * object_position;

    out.clip_position = clip_position;
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    // out.color = vec4<f32>(0.996078431372549, 0.9725490196078431, 0.6627450980392157, 1.0);
    out.color = color.value;
    return out;
}