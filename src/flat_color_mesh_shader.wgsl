// Vertex shader
struct CameraUniform {
    proj: mat4x4<f32>;
    view: mat4x4<f32>;
    rotation_only_view: mat4x4<f32>;
    near_plane_distance: f32;
    far_plane_distance: f32;
};
[[group(0), binding(0)]]
var<uniform> camera: CameraUniform;

struct LightPositionUniform {
    value: vec3<f32>;
};
[[group(0), binding(1)]]
var<uniform> light_position: LightPositionUniform;

struct ModelTransformUniform {
    value: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> model_transform: ModelTransformUniform;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] clip_position_nopersp: vec4<f32>; // clip position without perspective division
};

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
    [[builtin(frag_depth)]] depth: f32;
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
    out.clip_position_nopersp = clip_position;
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> FragmentOutput {
    // apply logarithmic depth
    // https://outerra.blogspot.com/2009/08/logarithmic-z-buffer.html
    // https://www.gamedev.net/blog/73/entry-2006307-tip-of-the-day-logarithmic-zbuffer-artifacts-fix/
    let c = 1.0;
    let depth_override = log(1.0 + in.clip_position_nopersp.z * c)
            / log(1.0 + camera.far_plane_distance * c);

    let color = vec4<f32>(0.996078431372549, 0.9725490196078431, 0.6627450980392157, 1.0);
    
    var out: FragmentOutput;
    out.color = color;
    out.depth = depth_override;
    return out;
}