struct MeshShaderCameraRaw {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    far_plane_distance: f32,
}

@group(0) @binding(0)
var<uniform> CAMERA: MeshShaderCameraRaw;

struct Instance {
    model_transform_0: vec4<f32>,
    model_transform_1: vec4<f32>,
    model_transform_2: vec4<f32>,
    model_transform_3: vec4<f32>,
    color: vec4<f32>,
}

struct BonesUniform {
    value: array<mat4x4<f32>>,
}
struct InstancesUniform {
    value: array<Instance>,
}

@group(1) @binding(0)
var<storage, read> bones_uniform: BonesUniform;
@group(1) @binding(1)
var<storage, read> instances_uniform: InstancesUniform;

struct VertexInput {
    @location(0) object_position: vec3<f32>,
    @location(1) bone_weights: vec4<f32>,
    @location(2) object_normal: vec3<f32>,
    @location(3) object_tangent: vec3<f32>,
    @location(4) object_tex_coords: vec2<f32>,
    @location(5) object_color: vec4<f32>,
    @location(6) bone_indices: vec4<u32>, 
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) vertex_color: vec4<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    vshader_input: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let instance = instances_uniform.value[instance_index];

    let model_transform = mat4x4<f32>(
        instance.model_transform_0,
        instance.model_transform_1,
        instance.model_transform_2,
        instance.model_transform_3,
    );

    let bone_indices = vshader_input.bone_indices;
    let bone_weights = vshader_input.bone_weights; // one f32 per weight
    let skin_transform_0 = bone_weights.x * bones_uniform.value[bone_indices.x];
    let skin_transform_1 = bone_weights.y * bones_uniform.value[bone_indices.y];
    let skin_transform_2 = bone_weights.z * bones_uniform.value[bone_indices.z];
    let skin_transform_3 = bone_weights.w * bones_uniform.value[bone_indices.w];
    let skin_transform = skin_transform_0 + skin_transform_1 + skin_transform_2 + skin_transform_3;
    let skinned_model_transform = model_transform * skin_transform;

    var out: VertexOutput;

    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let model_view_matrix = CAMERA.view_proj * skinned_model_transform;
    let clip_position = model_view_matrix * object_position;

    out.clip_position = clip_position;
    out.color = instance.color;
    out.vertex_color = vshader_input.object_color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    // out.color = vec4<f32>(0.996078431372549, 0.9725490196078431, 0.6627450980392157, 1.0);
    out.color = in.color * in.vertex_color;
    return out;
}