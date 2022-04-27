struct CameraUniform {
    proj: mat4x4<f32>;
    view: mat4x4<f32>;
    rotation_only_view: mat4x4<f32>;
    near_plane_distance: f32;
    far_plane_distance: f32;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

struct LightPositionUniform {
    value: vec4<f32>;
};
[[group(1), binding(1)]]
var<uniform> light_position: LightPositionUniform;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
    [[location(1)]] object_normal: vec3<f32>;
    [[location(2)]] object_tex_coords: vec2<f32>;
    [[location(3)]] object_tangent: vec3<f32>;
    [[location(4)]] object_bitangent: vec3<f32>;
};

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
    [[location(2)]] world_tangent: vec3<f32>;
    [[location(3)]] world_bitangent: vec3<f32>;
    [[location(4)]] tex_coords: vec2<f32>;
};

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
};

fn do_vertex_shade(vshader_input: VertexInput, model_transform: mat4x4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.world_normal = vshader_input.object_normal;
    
    let object_position   = vec4<f32>(vshader_input.object_position, 1.0);
    let camera_view_proj  = camera.proj * camera.view;
    let model_view_matrix = camera_view_proj * model_transform;
    let world_position    = model_transform * object_position;
    let clip_position     = model_view_matrix * object_position;
    let world_normal      = normalize((model_transform * vec4<f32>(vshader_input.object_normal, 0.0)).xyz);
    let world_tangent     = normalize((model_transform * vec4<f32>(vshader_input.object_tangent, 0.0)).xyz);
    let world_bitangent   = normalize((model_transform * vec4<f32>(vshader_input.object_bitangent, 0.0)).xyz);

    out.clip_position = clip_position;
    out.world_position = world_position.xyz;
    out.world_normal = world_normal;
    out.world_tangent = world_tangent;
    out.world_bitangent = world_bitangent;
    out.tex_coords = vshader_input.object_tex_coords;

    return out;
}

[[stage(vertex)]]
fn vs_main(
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

[[group(0), binding(0)]]
var diffuse_texture: texture_2d<f32>;
[[group(0), binding(1)]]
var diffuse_sampler: sampler;
[[group(0), binding(2)]]
var normal_map_texture: texture_2d<f32>;
[[group(0), binding(3)]]
var normal_map_sampler: sampler;


fn do_fragment_shade(
    world_position: vec3<f32>,
    world_normal: vec3<f32>, 
    tex_coords: vec2<f32>,
) -> FragmentOutput {    
    let albedo = textureSample(diffuse_texture, diffuse_sampler, tex_coords);

    let ambient_light_intensity = 0.05;
    let ambient_light_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    let light_position = light_position.value.xyz;
    let to_light_vec = normalize(light_position - world_position);
    let distance_squared = dot(to_light_vec, to_light_vec) * 2.0;
    let light_angle_factor = max(dot(world_normal, to_light_vec), 0.0);
    let max_light_intensity = 1.0;
    let light_intensity =
        ((light_angle_factor * max_light_intensity) / distance_squared);
    
    let light_color = vec4<f32>(0.996078431372549, 0.9725490196078431, 0.6627450980392157, 1.0);
    let final_color = (ambient_light_color * ambient_light_intensity + light_color * light_intensity) * albedo;
    
    var out: FragmentOutput;
    out.color = final_color;
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let tbn = (mat3x3<f32>(
        in.world_tangent,
        in.world_bitangent,
        in.world_normal,
    ));
    let normal_map_normal = textureSample(
        normal_map_texture, normal_map_sampler, in.tex_coords
    )  * 2.0 - 1.0;
    let tangent_space_normal = vec3<f32>(
        normal_map_normal.x, 
       -normal_map_normal.y, // I guess this is needed due to differing uv-mapping conventions
        normal_map_normal.z
    );
    let transformed_normal = normalize(tbn * tangent_space_normal);

    return do_fragment_shade(
        in.world_position, 
        transformed_normal, 
        in.tex_coords
    );
}