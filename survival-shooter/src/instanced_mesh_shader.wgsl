// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
    [[location(1)]] object_normal: vec3<f32>;
    [[location(2)]] object_tex_coords: vec2<f32>;
};

struct InstanceInput {
    [[location(5)]]  model_transform_0: vec4<f32>;
    [[location(6)]]  model_transform_1: vec4<f32>;
    [[location(7)]]  model_transform_2: vec4<f32>;
    [[location(8)]]  model_transform_3: vec4<f32>;
    [[location(9)]]  normal_rotation_transform_0: vec4<f32>;
    [[location(10)]] normal_rotation_transform_1: vec4<f32>;
    [[location(11)]] normal_rotation_transform_2: vec4<f32>;
    [[location(12)]] normal_rotation_transform_3: vec4<f32>;
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
    instance: InstanceInput,
) -> VertexOutput {
    let model_transform = mat4x4<f32>(
        instance.model_transform_0,
        instance.model_transform_1,
        instance.model_transform_2,
        instance.model_transform_3,
    );
    let normal_rotation_transform = mat4x4<f32>(
        instance.normal_rotation_transform_0,
        instance.normal_rotation_transform_1,
        instance.normal_rotation_transform_2,
        instance.normal_rotation_transform_3,
    );

    var out: VertexOutput;
    out.world_normal = vshader_input.object_normal;
    
    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let model_view_matrix = camera.view_proj * model_transform;
    let world_position = model_transform * object_position;

    out.world_position = world_position.xyz;
    out.clip_position = model_view_matrix * object_position;

    out.world_normal = (normal_rotation_transform * vec4<f32>(vshader_input.object_normal, 0.0)).xyz;
    // out.world_normal = vshader_input.object_normal;

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
    let to_light_vec = normalize(vec3<f32>(0.0, 3.0, 0.0) - in.world_position);
    let light_intensity = max(dot(in.world_normal, to_light_vec), 0.0);
    let albedo = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let max_light_intensity = 1.0;
    let ambient_light = 0.05;
    let distance_squared = dot(to_light_vec, to_light_vec) * 2.0;
    let final_light_intensity =
        ambient_light + ((light_intensity * max_light_intensity) / distance_squared);
    let final = final_light_intensity * albedo;
    // let final = final_light_intensity * vec4<f32>(0.5, 0.5, 0.5, 1.0);
    let some_color = vec3<f32>(0.5, 0.5, 0.5);
    return final;
}
 