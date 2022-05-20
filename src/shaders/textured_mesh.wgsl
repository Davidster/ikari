struct CameraUniform {
    proj: mat4x4<f32>;
    view: mat4x4<f32>;
    rotation_only_view: mat4x4<f32>;
    position: vec4<f32>;
    near_plane_distance: f32;
    far_plane_distance: f32;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

struct LightUniform {
    position: vec4<f32>;
    color: vec4<f32>;
};
[[group(1), binding(1)]]
var<uniform> light: LightUniform;

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

    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let camera_view_proj = camera.proj * camera.view;
    let model_view_matrix = camera_view_proj * model_transform;
    let world_position = model_transform * object_position;
    let clip_position = model_view_matrix * object_position;
    let world_normal = normalize((model_transform * vec4<f32>(vshader_input.object_normal, 0.0)).xyz);
    let world_tangent = normalize((model_transform * vec4<f32>(vshader_input.object_tangent, 0.0)).xyz);
    let world_bitangent = normalize((model_transform * vec4<f32>(vshader_input.object_bitangent, 0.0)).xyz);

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

[[group(2), binding(0)]]
var skybox_texture: texture_cube<f32>;
[[group(2), binding(1)]]
var skybox_sampler: sampler;
[[group(2), binding(2)]]
var diffuse_env_map_texture: texture_cube<f32>;
[[group(2), binding(3)]]
var diffuse_env_map_sampler: sampler;
[[group(2), binding(4)]]
var specular_env_map_texture: texture_cube<f32>;
[[group(2), binding(5)]]
var specular_env_map_sampler: sampler;
[[group(2), binding(6)]]
var brdf_lut_texture: texture_2d<f32>;
[[group(2), binding(7)]]
var brdf_lut_sampler: sampler;

let pi: f32 = 3.141592653589793;
let two_pi: f32 = 6.283185307179586;
let half_pi: f32 = 1.570796326794897;
let epsilon: f32 = 0.00001;

// https://learnopengl.com/PBR/Theory
fn normal_distribution_func_tr_ggx(
    a: f32,
    n: vec3<f32>,
    h: vec3<f32>,
) -> f32 {
    let a2 = a * a;
    let n_dot_h = dot(n, h);
    let n_dot_h_2 = n_dot_h * n_dot_h;
    let denom_temp = n_dot_h_2 * (a2 - 1.0) + 1.0;
    return a2 / (pi * denom_temp * denom_temp + epsilon);
}

fn geometry_func_schlick_ggx_k_direct(
    a: f32,
) -> f32 {
    let a_plus_1 = a + 1.0;
    return (a_plus_1 * a_plus_1) / 8.0;
}

fn geometry_func_schlick_ggx_k_ibl(
    a: f32,
) -> f32 {
    return (a * a) / 2.0;
}

fn geometry_func_schlick_ggx(
    n_dot_v: f32,
    k: f32,
) -> f32 {
    return n_dot_v / (n_dot_v * (1.0 - k) + k + epsilon);
}

fn geometry_func_smith_ggx(
    k: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let ggx_1 = geometry_func_schlick_ggx(n_dot_v, k);
    let ggx_2 = geometry_func_schlick_ggx(n_dot_l, k);
    return ggx_1 * ggx_2;
}

fn fresnel_func_schlick(
    h_dot_v: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - h_dot_v, 0.0, 1.0), 5.0);
}

fn fresnel_func_schlick_with_roughness(
    h_dot_v: f32,
    f0: vec3<f32>,
    a: f32,
) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - a), f0) - f0) * pow(clamp(1.0 - h_dot_v, 0.0, 1.0), 5.0);
    // return f0 + (max(vec3<f32>(1.0 - a), f0) - f0) * pow(1.0 - h_dot_v, 5.0);
}

fn radical_inverse_vdc(
    bits: u32,
) -> f32 {
    var out = bits;
    out = (out << 16u) | (out >> 16u);
    out = ((out & 0x55555555u) << 1u) | ((out & 0xAAAAAAAAu) >> 1u);
    out = ((out & 0x33333333u) << 2u) | ((out & 0xCCCCCCCCu) >> 2u);
    out = ((out & 0x0F0F0F0Fu) << 4u) | ((out & 0xF0F0F0F0u) >> 4u);
    out = ((out & 0x00FF00FFu) << 8u) | ((out & 0xFF00FF00u) >> 8u);
    return f32(out) * 2.3283064365386963e-10; // / 0x100000000
}

fn hammersley(
    i_u: u32,
    num_samples_u: u32,
) -> vec2<f32> {
    let i = f32(i_u);
    let num_samples = f32(num_samples_u);
    return vec2<f32>(i / num_samples, radical_inverse_vdc(i_u));
}

fn do_fragment_shade(
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    tex_coords: vec2<f32>,
    camera_position: vec3<f32>,
) -> FragmentOutput {

    let roughness = 0.3;
    let metallicness = 0.0;
    let albedo = textureSample(diffuse_texture, diffuse_sampler, tex_coords).rgb;
    // let albedo = vec3<f32>(textureSample(brdf_lut_texture, brdf_lut_sampler, tex_coords).rg, 0.0);

    let to_viewer_vec = normalize(camera_position - world_position);
    let to_light_vec = light.position.xyz - world_position;
    let to_light_vec_norm = normalize(to_light_vec);
    let distance_from_light = length(to_light_vec);
    let halfway_vec = normalize(to_viewer_vec + to_light_vec_norm);
    let reflection_vec = reflect(-to_viewer_vec, normalize(world_normal));
    let surface_reflection_at_zero_incidence_dialectric = vec3<f32>(0.04);
    let surface_reflection_at_zero_incidence = mix(
        surface_reflection_at_zero_incidence_dialectric,
        albedo,
        metallicness
    );
    // let surface_reflection_at_zero_incidence = vec3<f32>(0.95, 0.93, 0.88);

    // copy variable names from the math formulas
    let n = world_normal;
    let w0 = to_viewer_vec;
    let v = w0;
    let wi = to_light_vec_norm;
    let l = wi;
    let h = halfway_vec;
    let a = roughness;
    let f0 = surface_reflection_at_zero_incidence;
    let k = geometry_func_schlick_ggx_k_direct(a); // use geometry_func_schlick_ggx_k_ibl for IBL

    // specular
    let h_dot_v = max(dot(h, v), 0.0);
    let normal_distribution = normal_distribution_func_tr_ggx(a, n, h);
    let geometry = geometry_func_smith_ggx(k, n, v, l);
    let fresnel = fresnel_func_schlick(h_dot_v, f0);
    let cook_torrance_denominator = 4.0 * max(dot(n, w0), 0.0) * max(dot(n, wi), 0.0) + epsilon;
    let specular_component = normal_distribution * geometry * fresnel / cook_torrance_denominator;
    let ks = fresnel;

    // diffuse
    let diffuse_component = albedo / pi; // lambertian
    let kd = vec3<f32>(1.0) - ks;

    // https://learnopengl.com/Lighting/Light-casters
    let light_attenuation_factor_d20 = 1.0 / (1.0 + 0.22 * distance_from_light + 0.20 * distance_from_light * distance_from_light);
    let light_attenuation_factor_d100 = 1.0 / (1.0 + 0.045 * distance_from_light + 0.0075 * distance_from_light * distance_from_light);
    let light_attenuation_factor_d600 = 1.0 / (1.0 + 0.007 * distance_from_light + 0.0002 * distance_from_light * distance_from_light);
    let light_attenuation_factor_d3250 = 1.0 / (1.0 + 0.0014 * distance_from_light + 0.000007 * distance_from_light * distance_from_light);
    let light_attenuation_factor = light_attenuation_factor_d3250;
    let incident_angle_factor = max(dot(n, wi), 0.0);      
    //                                  ks was already multiplied by fresnel so it's omitted here       
    let bdrf = kd * diffuse_component + specular_component;
    let light_irradiance = bdrf * incident_angle_factor * light_attenuation_factor * light.color.rgb;

    let n_dot_v = max(dot(n, v), 0.0);
    let fresnel_ambient = fresnel_func_schlick_with_roughness(n_dot_v, f0, a);
    let kd_ambient = vec3<f32>(1.0) - fresnel_ambient;
    let ambient_diffuse_irradiance = textureSample(diffuse_env_map_texture, diffuse_env_map_sampler, world_normal).rgb * albedo;
    let ambient_irradiance = kd_ambient * ambient_diffuse_irradiance;

    let combined_irradiance_hdr = ambient_irradiance + light_irradiance;
    // let combined_irradiance_hdr = ambient_irradiance;
    let combined_irradiance_ldr = combined_irradiance_hdr / (combined_irradiance_hdr + vec3<f32>(1.0, 1.0, 1.0));

    // mip level count - 1
    // let MAX_REFLECTION_LOD = 4.0;
    // let surface_reflection = textureSampleLevel(
    //     specular_env_map_texture,
    //     specular_env_map_sampler,
    //     reflection_vec,
    //     roughness * MAX_REFLECTION_LOD
    // );
    // let final_color = surface_reflection;

    // let final_color = vec4<f32>(fresnel * incident_angle_factor, 1.0);
    let final_color = vec4<f32>(combined_irradiance_ldr, 1.0);

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
        normal_map_texture,
        normal_map_sampler,
        in.tex_coords
    ) * 2.0 - 1.0;
    let tangent_space_normal = vec3<f32>(
        normal_map_normal.x,
        -normal_map_normal.y, // I guess this is needed due to differing uv-mapping conventions
        normal_map_normal.z
    );
    let transformed_normal = normalize(tbn * tangent_space_normal);

    return do_fragment_shade(
        in.world_position,
        transformed_normal,
        in.tex_coords,
        camera.position.xyz
    );
}