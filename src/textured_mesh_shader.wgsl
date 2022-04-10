// Vertex shader
struct CameraUniform {
    proj: mat4x4<f32>;
    view: mat4x4<f32>;
    rotation_only_view: mat4x4<f32>;
    far_plane_distance: f32;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

// used for non-instanced renders
struct ModelTransformUniform {
    value: mat4x4<f32>;
};
[[group(2), binding(0)]]
var<uniform> model_transform: ModelTransformUniform;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
    [[location(1)]] object_normal: vec3<f32>;
    [[location(2)]] object_tex_coords: vec2<f32>;
};

// used for instanced renders
struct ModelTransformInstance {
    [[location(5)]]  model_transform_0: vec4<f32>;
    [[location(6)]]  model_transform_1: vec4<f32>;
    [[location(7)]]  model_transform_2: vec4<f32>;
    [[location(8)]]  model_transform_3: vec4<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec3<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] tex_coords: vec2<f32>;
};

fn do_vertex_shade(vshader_input: VertexInput, model_transform: mat4x4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.world_normal = vshader_input.object_normal;
    
    let camera_view_proj = camera.proj * camera.view;
    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let model_view_matrix = camera_view_proj * model_transform;
    let world_position = model_transform * object_position;
    var clip_position = model_view_matrix * object_position;

    // apply logarithmic depth
    // https://outerra.blogspot.com/2009/08/logarithmic-z-buffer.html
    let nearby_object_resolution_scalar = 1.0;
    clip_position.z = log(nearby_object_resolution_scalar * clip_position.w + 1.0)
            / log(nearby_object_resolution_scalar * camera.far_plane_distance + 1.0);
    clip_position.z = clip_position.z * clip_position.w;

    out.world_position = world_position.xyz;
    out.clip_position = clip_position;

    out.world_normal = normalize((model_transform * vec4<f32>(vshader_input.object_normal, 0.0)).xyz);

    out.tex_coords = vshader_input.object_tex_coords;

    return out;
}

// non-instanced vertex shader:

[[stage(vertex)]]
fn vs_main(
    vshader_input: VertexInput,
) -> VertexOutput {
    return do_vertex_shade(vshader_input, model_transform.value);
}

// instanced vertex shader:

[[stage(vertex)]]
fn instanced_vs_main(
    vshader_input: VertexInput,
    instance: ModelTransformInstance,
) -> VertexOutput {
    let model_transform = mat4x4<f32>(
        instance.model_transform_0,
        instance.model_transform_1,
        instance.model_transform_2,
        instance.model_transform_3,
    );
    return do_vertex_shade(vshader_input, model_transform);
}

// main fragment shader

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
    let ambient_light = 0.15;
    let distance_squared = dot(to_light_vec, to_light_vec) * 2.0;
    let final_light_intensity =
        ambient_light + ((light_intensity * max_light_intensity) / distance_squared);
    let final = final_light_intensity * albedo;
    // let final = final_light_intensity * vec4<f32>(0.5, 0.5, 0.5, 1.0);
    let some_color = vec4<f32>(0.5, 1.0, 0.5, 1.0);
    // return some_color;
    return final;
}