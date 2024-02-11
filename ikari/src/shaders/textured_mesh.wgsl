struct MeshShaderCameraRaw {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    far_plane_distance: f32,
}

@group(0) @binding(0)
var<uniform> CAMERA: MeshShaderCameraRaw;

const MAX_LIGHTS = 32u;
const MAX_BONES = 512u;
const POINT_LIGHT_SHOW_MAP_COUNT = 2u;
const DIRECTIONAL_LIGHT_SHOW_MAP_COUNT = 2u;
const POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE: f32 = 1000.0;
const MAX_SHADOW_CASCADES = 4u;
const MAX_TOTAL_SHADOW_CASCADES = 128u; // MAX_LIGHTS * MAX_SHADOW_CASCADES
// TODO: pass this from cpu
const SOFT_SHADOW_MAX_DISTANCE: f32 = 10000.0;

const MIN_SHADOW_MAP_BIAS: f32 = 0.00005;
const pi: f32 = 3.141592653589793;
const two_pi: f32 = 6.283185307179586;
const half_pi: f32 = 1.570796326794897;
const epsilon: f32 = 0.00001;
const DEBUG_POINT_LIGHT_SAMPLED_FACES: f32 = 0.0;

struct PointLight {
    position: vec4<f32>,
    color: vec4<f32>,
}
struct DirectionalLight {
    direction: vec4<f32>,
    color: vec4<f32>,
}
struct DirectionalLightCascade {
    world_space_to_light_space: mat4x4<f32>,
    distance_and_pixel_size: vec4<f32>,
}
struct Instance {
    model_transform_0: vec4<f32>,
    model_transform_1: vec4<f32>,
    model_transform_2: vec4<f32>,
    model_transform_3: vec4<f32>,
    base_color_factor: vec4<f32>,
    emissive_factor: vec4<f32>,
    mrno: vec4<f32>, // metallicness_factor, roughness_factor, normal scale, occlusion strength
    alpha_cutoff: vec4<f32>, // alpha_cutoff, padding
}

struct PointLightsUniform {
    lights: array<PointLight, MAX_LIGHTS>,
}
struct DirectionalLightsUniform {
    lights: array<DirectionalLight, MAX_LIGHTS>,
    cascades: array<DirectionalLightCascade, MAX_TOTAL_SHADOW_CASCADES>,
}
struct BonesUniform {
    value: array<mat4x4<f32>>,
}
struct InstancesUniform {
    value: array<Instance>,
}
// scratch board for shader options
struct PbrShaderOptionsUniform {
    options_1: vec4<f32>,
    options_2: vec4<f32>,
    options_3: vec4<f32>,
    options_4: vec4<f32>,
}

@group(0) @binding(1)
var<uniform> point_lights: PointLightsUniform;
@group(0) @binding(2)
var<uniform> directional_lights: DirectionalLightsUniform;
@group(0) @binding(3)
var<uniform> shader_options: PbrShaderOptionsUniform;

@group(2) @binding(0)
var<storage, read> bones_uniform: BonesUniform;
@group(2) @binding(1)
var<storage, read> instances_uniform: InstancesUniform;

@group(1) @binding(0)
var<storage, read> shadow_bones_uniform: BonesUniform;
@group(1) @binding(1)
var<storage, read> shadow_instances_uniform: InstancesUniform;



struct VertexInput {
    @location(0) object_position: vec3<f32>,
    @location(1) object_normal: vec3<f32>,
    @location(2) object_tex_coords: vec2<f32>,
    @location(3) object_tangent: vec3<f32>,
    @location(4) object_bitangent: vec3<f32>,
    @location(5) object_color: vec4<f32>,
    @location(6) bone_indices: vec4<u32>,
    @location(7) bone_weights: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) world_bitangent: vec3<f32>,
    @location(4) tex_coords: vec2<f32>,
    @location(5) vertex_color: vec4<f32>,
    @location(6) base_color_factor: vec4<f32>,
    @location(7) emissive_factor: vec4<f32>,
    @location(8) metallicness_factor: f32,
    @location(9) roughness_factor: f32,
    @location(10) normal_scale: f32,
    @location(11) occlusion_strength: f32,
    @location(12) alpha_cutoff: f32,
    @location(13) object_tangent: vec3<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

struct ShadowMappingVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) alpha_cutoff: f32,
}

struct ShadowMappingFragmentOutput {
    @builtin(frag_depth) depth: f32,
}

@group(2) @binding(0)
var shadow_diffuse_texture: texture_2d<f32>;
@group(2) @binding(1)
var shadow_diffuse_sampler: sampler;

@group(3) @binding(0)
var diffuse_texture: texture_2d<f32>;
@group(3) @binding(1)
var diffuse_sampler: sampler;
@group(3) @binding(2)
var normal_map_texture: texture_2d<f32>;
@group(3) @binding(3)
var normal_map_sampler: sampler;
@group(3) @binding(4)
var metallic_roughness_map_texture: texture_2d<f32>;
@group(3) @binding(5)
var metallic_roughness_map_sampler: sampler;
@group(3) @binding(6)
var emissive_map_texture: texture_2d<f32>;
@group(3) @binding(7)
var emissive_map_sampler: sampler;
@group(3) @binding(8)
var ambient_occlusion_map_texture: texture_2d<f32>;
@group(3) @binding(9)
var ambient_occlusion_map_sampler: sampler;

@group(1) @binding(0)
var skybox_texture: texture_cube<f32>;
@group(1) @binding(1)
var skybox_sampler: sampler;
@group(1) @binding(2)
var skybox_texture_2: texture_cube<f32>;
@group(1) @binding(3)
var<uniform> skybox_weights: vec4<f32>;
@group(1) @binding(4)
var diffuse_env_map_texture: texture_cube<f32>;
@group(1) @binding(5)
var diffuse_env_map_texture_2: texture_cube<f32>;
@group(1) @binding(6)
var specular_env_map_texture: texture_cube<f32>;
@group(1) @binding(7)
var specular_env_map_texture_2: texture_cube<f32>;
@group(1) @binding(8)
var brdf_lut_texture: texture_2d<f32>;
@group(1) @binding(9)
var brdf_lut_sampler: sampler;
@group(1) @binding(10)
var point_shadow_map_textures: texture_2d_array<f32>;
@group(1) @binding(11)
var directional_shadow_map_textures: texture_2d_array<f32>;
@group(1) @binding(12)
var shadow_map_sampler: sampler;

fn get_soft_shadows_are_enabled() -> bool {
    return shader_options.options_1[0] > 0.0;
}

fn get_soft_shadow_factor() -> f32 {
    return shader_options.options_1[1];
}

fn get_shadow_debug_enabled() -> bool {
    return shader_options.options_1[2] > 0.0;
}

fn get_soft_shadow_grid_dims() -> u32 {
    return u32(shader_options.options_1[3]);
}

fn get_shadow_bias() -> f32 {
     return shader_options.options_2[0];
}

fn get_cascade_debug_enabled() -> bool {
    return shader_options.options_2[1] > 0.0;
}

fn do_vertex_shade(
    vshader_input: VertexInput,
    camera_view_proj: mat4x4<f32>,
    model_transform: mat4x4<f32>,
    skin_transform: mat4x4<f32>,
    base_color_factor: vec4<f32>,
    emissive_factor: vec4<f32>,
    metallicness_factor: f32,
    roughness_factor: f32,
    normal_scale: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32
) -> VertexOutput {
    var out: VertexOutput;
    out.world_normal = vshader_input.object_normal;

    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let skinned_model_transform = model_transform * skin_transform;
    let world_position = skinned_model_transform * object_position;
    let clip_position = camera_view_proj * skinned_model_transform * object_position;
    let world_normal = normalize((skinned_model_transform * vec4<f32>(vshader_input.object_normal, 0.0)).xyz);
    let world_tangent = normalize((skinned_model_transform * vec4<f32>(vshader_input.object_tangent, 0.0)).xyz);
    let world_bitangent = normalize((skinned_model_transform * vec4<f32>(vshader_input.object_bitangent, 0.0)).xyz);

    out.clip_position = clip_position;
    out.world_position = world_position.xyz;
    out.world_normal = world_normal;
    out.world_tangent = world_tangent;
    out.object_tangent = vshader_input.object_tangent;
    out.world_bitangent = world_bitangent;
    out.tex_coords = vshader_input.object_tex_coords;
    out.vertex_color = vshader_input.object_color;
    out.base_color_factor = base_color_factor;
    out.emissive_factor = emissive_factor;
    out.metallicness_factor = metallicness_factor;
    out.roughness_factor = roughness_factor;
    out.normal_scale = normal_scale;
    out.occlusion_strength = occlusion_strength;
    out.alpha_cutoff = alpha_cutoff;

    return out;
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

    return do_vertex_shade(
        vshader_input,
        CAMERA.view_proj,
        model_transform,
        skin_transform,
        instance.base_color_factor,
        instance.emissive_factor,
        instance.mrno[0],
        instance.mrno[1],
        instance.mrno[2],
        instance.mrno[3],
        instance.alpha_cutoff[0],
    );
}

@vertex
fn shadow_map_vs_main(
    vshader_input: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> ShadowMappingVertexOutput {
    let instance = shadow_instances_uniform.value[instance_index];

    let model_transform = mat4x4<f32>(
        instance.model_transform_0,
        instance.model_transform_1,
        instance.model_transform_2,
        instance.model_transform_3,
    );

    let bone_indices = vshader_input.bone_indices;
    let bone_weights = vshader_input.bone_weights; // one f32 per weight
    let skin_transform_0 = bone_weights.x * shadow_bones_uniform.value[bone_indices.x];
    let skin_transform_1 = bone_weights.y * shadow_bones_uniform.value[bone_indices.y];
    let skin_transform_2 = bone_weights.z * shadow_bones_uniform.value[bone_indices.z];
    let skin_transform_3 = bone_weights.w * shadow_bones_uniform.value[bone_indices.w];
    let skin_transform = skin_transform_0 + skin_transform_1 + skin_transform_2 + skin_transform_3;

    let object_position = vec4<f32>(vshader_input.object_position, 1.0);
    let skinned_model_transform = model_transform * skin_transform;
    // let skinned_model_transform = model_transform/* * skin_transform */;
    let world_position = skinned_model_transform * object_position;
    let clip_position = CAMERA.view_proj * skinned_model_transform * object_position;

    var out: ShadowMappingVertexOutput;
    out.clip_position = clip_position;
    out.world_position = world_position.xyz;
    out.tex_coords = vshader_input.object_tex_coords;
    out.alpha_cutoff = instance.alpha_cutoff[0];
    return out;
}

@fragment
fn point_shadow_map_fs_main(
    in: ShadowMappingVertexOutput
) -> ShadowMappingFragmentOutput {

    let base_color_alpha = textureSample(
        shadow_diffuse_texture,
        shadow_diffuse_sampler,
        in.tex_coords
    ).a;

    if base_color_alpha <= in.alpha_cutoff {
        discard;
    }

    var out: ShadowMappingFragmentOutput;
    let light_distance = length(in.world_position - CAMERA.position.xyz);
    out.depth = light_distance / CAMERA.far_plane_distance;
    return out;
}

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
    cos_theta: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_func_schlick_with_roughness(
    cos_theta: f32,
    f0: vec3<f32>,
    a: f32,
) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - a), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
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

fn world_normal_to_cubemap_vec(world_pos: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(-world_pos.x, world_pos.y, world_pos.z);
}

// noise functions from:
// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
fn mod289(x: vec4<f32>) -> vec4<f32> { return x - floor(x * (1. / 289.)) * 289.; }
fn perm4(x: vec4<f32>) -> vec4<f32> { return mod289(((x * 34.) + 1.) * x); }

fn noise3(p: vec3<f32>) -> f32 {
  let a = floor(p);
  var d: vec3<f32> = p - a;
  d = d * d * (3. - 2. * d);

  let b = a.xxyy + vec4<f32>(0., 1., 0., 1.);
  let k1 = perm4(b.xyxy);
  let k2 = perm4(k1.xyxy + b.zzww);

  let c = k2 + a.zzzz;
  let k3 = perm4(c);
  let k4 = perm4(c + 1.);

  let o1 = fract(k3 * (1. / 41.));
  let o2 = fract(k4 * (1. / 41.));

  let o3 = o2 * d.z + o1 * (1. - d.z);
  let o4 = o3.yw * d.x + o3.xz * (1. - d.x);

  return o4.y * d.y + o4.x * (1. - d.y);
}

fn rand(n: f32) -> f32 { return fract(n * n * 43758.5453123); }
fn noise(p: f32) -> f32 {
  let fl = floor(p);
  let fc = fract(p);
  return mix(rand(fl), rand(fl + 1.), fc);
}

fn disk_warp(coord: vec2<f32>) -> vec2<f32> {
    let warped = vec2<f32>(
        sqrt(coord.y) * cos(2.0 * pi * coord.x),
        sqrt(coord.y) * sin(2.0 * pi * coord.x)
    );
    return vec2<f32>(0.5, 0.5) * warped + vec2<f32>(0.5, 0.5);
}

// see https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-17-efficient-soft-edged-shadows-using
// and https://www.shadertoy.com/view/dtd3Dr
// jitter_amount values have domain of (-1, 1)
fn get_soft_shadow_sample_jitter(grid_coord: vec2<u32>, jitter_amount: vec2<f32>, grid_dims: u32) -> vec2<f32> {
    // (0, 1) domain
    let cell_size = 1.0 / f32(grid_dims);
    let cell_top_left = vec2<f32>(f32(grid_coord.x), f32(grid_coord.y)) * vec2<f32>(cell_size, cell_size);
    let cell_center = cell_top_left + vec2<f32>(0.5 * cell_size);

    // jitter from the default center positions
    let max_jitter = 0.5 * cell_size;
    let jittered_cell_center = cell_center + vec2<f32>(max_jitter, max_jitter) * jitter_amount;
    
    // convert to (-1, 1) domain
    return vec2(2.0, 2.0) * disk_warp(jittered_cell_center) - vec2(1.0, 1.0);
}

fn compute_direct_lighting(
    world_normal: vec3<f32>,
    to_viewer_vec: vec3<f32>,
    to_light_vec: vec3<f32>,
    light_color_scaled: vec3<f32>,
    light_attenuation_factor: f32,
    base_color: vec3<f32>,
    roughness: f32,
    metallicness: f32,
    f0: vec3<f32>
) -> vec3<f32> {
    // copy variable names from the math formulas
    let n = world_normal;
    let w0 = to_viewer_vec;
    let v = w0;
    let a = roughness;

    let halfway_vec = normalize(to_viewer_vec + to_light_vec);
    
    // let surface_reflection_at_zero_incidence = vec3<f32>(0.95, 0.93, 0.88);

    // copy variable names from the math formulas
    let wi = to_light_vec;
    let l = wi;
    let h = halfway_vec;

    // specular
    let h_dot_v = max(dot(h, v), 0.0);
    let normal_distribution = normal_distribution_func_tr_ggx(a, n, h);
    let k = geometry_func_schlick_ggx_k_direct(a);
    let geometry = geometry_func_smith_ggx(k, n, v, l);
    let fresnel = fresnel_func_schlick(h_dot_v, f0);
    let cook_torrance_denominator = 4.0 * max(dot(n, w0), 0.0) * max(dot(n, wi), 0.0) + epsilon;
    let specular_component = normal_distribution * geometry * fresnel / cook_torrance_denominator;
    let ks = fresnel;

    // diffuse
    let diffuse_component = base_color / pi; // lambertian
    let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallicness);

    let incident_angle_factor = max(dot(n, wi), 0.0);      
    //                                  ks was already multiplied by fresnel so it's omitted here       
    let bdrf = kd * diffuse_component + specular_component;
    return bdrf * incident_angle_factor * light_attenuation_factor * light_color_scaled;
}

// cursed magic, adapted from these links:
// https://kosmonautblog.wordpress.com/2017/03/25/shadow-filtering-for-pointlights/
// https://github.com/Kosmonaut3d/DeferredEngine/blob/f772b53e7e09dde6d0dd0682f4c4c1f1f6957b69/EngineTest/Content/Shaders/Common/helper.fx#L367
fn vector_to_cubemap_uv(in_vec_preflip: vec3<f32>) -> vec3<f32> {
    var uv: vec2<f32>;
    var slice: f32;
    let in_vec = vec3(-in_vec_preflip.x, in_vec_preflip.y, -in_vec_preflip.z);
    let in_vec_abs = abs(in_vec);
    
    // positive and negative x
    if in_vec_abs.x >= in_vec_abs.y && in_vec_abs.x >= in_vec_abs.z {
        let in_vec_div = in_vec / in_vec.x;
        uv.x = -in_vec_div.z;
        if in_vec.x > 0.0 {
            slice = 1.0;
            uv.y = -in_vec_div.y;
        } else {
            slice = 0.0;
            uv.y = in_vec_div.y;
        }
    // positive and negative y
    } else if in_vec_abs.y >= in_vec_abs.x && in_vec_abs.y >= in_vec_abs.z {
        let in_vec_div = in_vec / in_vec.y;
        uv.y = -in_vec_div.z;
        if in_vec.y > 0.0 {
            slice = 2.0;
            uv.x = -in_vec_div.x;
        } else {
            slice = 3.0; 
            uv.x = in_vec_div.x;
        }
    // positive and negative z
    } else {
        let in_vec_div = in_vec / in_vec.z;
        uv.x = in_vec_div.x;
        if in_vec.z < 0.0 {
            slice = 4.0;
            uv.y = in_vec_div.y;
        } else {
            slice = 5.0;
            uv.y = -in_vec_div.y;
        }
    }

    let one_sixth = 1.0 / 6.0;

    // now we are in [-1,1]x[-1,1] space, so transform to texCoords
    uv = (uv + vec2(1.0, 1.0)) * 0.5;

    // now transform to slice position
    uv.x = uv.x * one_sixth + slice * one_sixth;

    return vec3(uv, slice);
}

fn clamp_jittered_cubemap_uv(uv: vec2<f32>, face_slice: f32) -> vec2<f32> {
    let min_max_u = vec2(0.00001, -0.00001) + vec2(face_slice, 1.0 + face_slice) / vec2(6.0, 6.0);
    return vec2<f32>(
        clamp(uv.x, min_max_u.x, min_max_u.y),
        uv.y
    );
}

fn do_fragment_shade(
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    tex_coords: vec2<f32>,
    vertex_color: vec4<f32>,
    camera_position: vec3<f32>,
    base_color_factor: vec4<f32>,
    emissive_factor: vec4<f32>,
    metallicness_factor: f32,
    roughness_factor: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32
) -> FragmentOutput {

    // let roughness = 0.12;
    // let metallicness = 0.8;
    let base_color_t = textureSample(
        diffuse_texture,
        diffuse_sampler,
        tex_coords
    );
    
    let base_color = base_color_t.rgb * base_color_factor.rgb * vertex_color.rgb;
    let metallic_roughness = textureSample(
        metallic_roughness_map_texture,
        metallic_roughness_map_sampler,
        tex_coords
    ).rgb;
    let metallicness = metallic_roughness.z * metallicness_factor;
    let roughness = metallic_roughness.y * roughness_factor;
    let ambient_occlusion = textureSample(
        ambient_occlusion_map_texture,
        ambient_occlusion_map_sampler,
        tex_coords
    ).r;
    let emissive = textureSample(
        emissive_map_texture,
        emissive_map_sampler,
        tex_coords
    ).rgb * emissive_factor.rgb;

    let to_viewer_vec_length = length(camera_position - world_position);
    let to_viewer_vec = (camera_position - world_position) / to_viewer_vec_length;
    let reflection_vec = reflect(-to_viewer_vec, normalize(world_normal));
    let surface_reflection_at_zero_incidence_dialectric = vec3<f32>(0.04);
    let surface_reflection_at_zero_incidence = mix(
        surface_reflection_at_zero_incidence_dialectric,
        base_color,
        metallicness
    );

    let MAX_REFLECTION_LOD = 4.0;
    let pre_filtered_color_1 = textureSampleLevel(
        specular_env_map_texture,
        skybox_sampler,
        world_normal_to_cubemap_vec(reflection_vec),
        roughness * MAX_REFLECTION_LOD
    ).rgb;
    let pre_filtered_color_2 = textureSampleLevel(
        specular_env_map_texture_2,
        skybox_sampler,
        world_normal_to_cubemap_vec(reflection_vec),
        roughness * MAX_REFLECTION_LOD
    ).rgb;

    let pre_filtered_color = (skybox_weights.x * pre_filtered_color_1) + (skybox_weights.y * pre_filtered_color_2);

    // copy variable names from the math formulas
    let n = world_normal;
    let w0 = to_viewer_vec;
    let v = w0;
    let a = roughness;
    let f0 = surface_reflection_at_zero_incidence;

    let n_dot_v = max(dot(n, v), 0.0);
    let brdf_lut_res = textureSample(brdf_lut_texture, brdf_lut_sampler, vec2<f32>(n_dot_v, roughness));
    let env_map_diffuse_irradiance_1 = textureSample(
        diffuse_env_map_texture, 
        skybox_sampler, 
        world_normal_to_cubemap_vec(world_normal)
    ).rgb;
    let env_map_diffuse_irradiance_2 = textureSample(
        diffuse_env_map_texture_2, 
        skybox_sampler, 
        world_normal_to_cubemap_vec(world_normal)
    ).rgb;

    let env_map_diffuse_irradiance = (skybox_weights.x * env_map_diffuse_irradiance_1) + (skybox_weights.y * env_map_diffuse_irradiance_2);

    if base_color_t.a <= alpha_cutoff {
        discard;
    }

    var random_seed_x = 1000.0 * vec3<f32>(
        world_position.x,
        world_position.y,
        world_position.z,
    );
    var random_seed_y = random_seed_x + 1000.0;
    var random_jitter = vec2(2.0, 2.0) * vec2(noise3(random_seed_x), noise3(random_seed_y)) - vec2(1.0, 1.0);

    let shadow_bias = get_shadow_bias();
    let soft_shadow_grid_dims = get_soft_shadow_grid_dims();

    var total_shadow_occlusion_acc = 0.0;
    var total_light_count = 0u;

    var total_light_irradiance = vec3<f32>(0.0);
    for (var light_index = 0u; light_index < MAX_LIGHTS; light_index = light_index + 1u) {
        let light = point_lights.lights[light_index];

        if light.color.w < epsilon {
            // break seems to be decently faster than continue here
            break;
        }

        let light_color_scaled = light.color.xyz * light.color.w;

        let from_shadow_vec = world_position - light.position.xyz;
        let to_light_vec = -from_shadow_vec;
        let to_light_vec_norm = normalize(to_light_vec);
        let n_dot_l = max(dot(n, to_light_vec_norm), 0.0);
        let light_space_position_uv_and_face_slice = vector_to_cubemap_uv(world_normal_to_cubemap_vec(from_shadow_vec));
        let light_space_position_uv = light_space_position_uv_and_face_slice.xy;
        let light_space_position_face_slice = light_space_position_uv_and_face_slice.z;
        let current_depth = length(from_shadow_vec) / POINT_LIGHT_SHADOW_MAP_FRUSTUM_FAR_PLANE; // domain is (0, 1), lower means closer to the light
        let bias = mix(shadow_bias, MIN_SHADOW_MAP_BIAS, n_dot_l);

        var shadow_occlusion_acc = 0.0;

        if n_dot_l > 0.0 && light_index < POINT_LIGHT_SHOW_MAP_COUNT {
            if get_soft_shadows_are_enabled() {
                // soft shadows code path
                // TODO: dedupe with directional lights

                var early_test_coords = array<vec2<u32>, 4>(
                    vec2<u32>(0u, 3u),
                    vec2<u32>(1u, 3u),
                    vec2<u32>(2u, 3u),
                    vec2<u32>(3u, 3u)
                );

                let max_sample_jitter = get_soft_shadow_factor() * 30.0;

                for (var i = 0; i < 4; i++) {
                    let base_sample_jitter = get_soft_shadow_sample_jitter(early_test_coords[i], random_jitter, 4u);
                    // TODO: multiply by current_depth to get softer shadows at a distance?
                    var sample_jitter = base_sample_jitter * max_sample_jitter;
                    sample_jitter.y = sample_jitter.y * 6.0;
                    let closest_depth = textureSampleLevel(
                        point_shadow_map_textures,
                        shadow_map_sampler,
                        clamp_jittered_cubemap_uv(
                            light_space_position_uv + sample_jitter, 
                            light_space_position_face_slice
                        ),
                        i32(light_index),
                        0.0
                    ).r;

                    if current_depth - bias < closest_depth {
                        shadow_occlusion_acc = shadow_occlusion_acc + 0.25;
                    }
                }

                // if the early test finds the fragment to be completely in shadow, 
                // completely in light, or its surface isn't facing the light (n_dot_l =< 0)
                // then skip the extra work that we do to soften the penumbra
                if (shadow_occlusion_acc - 1.0) * shadow_occlusion_acc * n_dot_l != 0.0 {
                    
                    if soft_shadow_grid_dims > 0u {
                        // TODO: don't clear shadow_occlusion_acc, we can perserve it and perform fewer samples here
                        // (skip the samples already done by early test) for a theoretically equivalent level of quality
                        // and decent performance boost if soft_shadow_grid_dims isn't too high
                        shadow_occlusion_acc = 0.0;
                    }

                    for (var i = 0u; i < soft_shadow_grid_dims; i++) {
                        for (var j = 0u; j < soft_shadow_grid_dims; j++) {
                            let coord = vec2<u32>(i, j);
                            let base_sample_jitter = get_soft_shadow_sample_jitter(coord, random_jitter, soft_shadow_grid_dims);
                            // TODO: multiply by current_depth to get softer shadows at a distance?
                            var sample_jitter = base_sample_jitter * max_sample_jitter;
                            sample_jitter.y = sample_jitter.y * 6.0;
                            let closest_depth = textureSampleLevel(
                                point_shadow_map_textures,
                                shadow_map_sampler,
                                clamp_jittered_cubemap_uv(
                                    light_space_position_uv + sample_jitter, 
                                    light_space_position_face_slice
                                ),
                                i32(light_index),
                                0.0
                            ).r;
                            
                            if current_depth - bias < closest_depth {
                                shadow_occlusion_acc = shadow_occlusion_acc + (1.0 / f32(soft_shadow_grid_dims * soft_shadow_grid_dims));
                            }
                        }
                    }
                }
            } else {
                // hard shadows
                let closest_depth = textureSampleLevel(
                    point_shadow_map_textures,
                    shadow_map_sampler,
                    light_space_position_uv,
                    i32(light_index),
                    0.0
                ).r;
                if (current_depth - bias < closest_depth) {
                    shadow_occlusion_acc = 1.0;
                }
            }
        }

        var shadow_occlusion_factor = shadow_occlusion_acc;
        total_shadow_occlusion_acc = total_shadow_occlusion_acc + shadow_occlusion_acc;
        total_light_count = total_light_count + 1u;

        if shadow_occlusion_factor < epsilon {
                continue;
        }

        let distance_from_light = length(to_light_vec);
        // https://learnopengl.com/Lighting/Light-casters
        // let light_attenuation_factor_d20 = 1.0 / (1.0 + 0.22 * distance_from_light + 0.20 * distance_from_light * distance_from_light);
        // let light_attenuation_factor_d100 = 1.0 / (1.0 + 0.045 * distance_from_light + 0.0075 * distance_from_light * distance_from_light);
        let light_attenuation_factor_d600 = 1.0 / (1.0 + 0.007 * distance_from_light + 0.0002 * distance_from_light * distance_from_light);
        // let light_attenuation_factor_d3250 = 1.0 / (1.0 + 0.0014 * distance_from_light + 0.000007 * distance_from_light * distance_from_light);
        let light_attenuation_factor = light_attenuation_factor_d600;

        var light_irradiance = compute_direct_lighting(
            world_normal,
            to_viewer_vec,
            to_light_vec_norm,
            light_color_scaled,
            light_attenuation_factor,
            base_color,
            roughness,
            metallicness,
            f0
        );

        if DEBUG_POINT_LIGHT_SAMPLED_FACES > 0.0 {
            light_irradiance = (vec3(light_space_position_face_slice, light_space_position_face_slice + 1.0, light_space_position_face_slice + 2.0) % 6.0) / 6.0;
        }

        total_light_irradiance = total_light_irradiance + light_irradiance * shadow_occlusion_factor;
    }

    for (var light_index = 0u; light_index < MAX_LIGHTS; light_index = light_index + 1u) {
        let light = directional_lights.lights[light_index];

        if light.color.w < epsilon {
            // break seems to be decently faster than continue here
            break;
        }

        let light_color_scaled = light.color.xyz * light.color.w;

        let to_light_vec = -light.direction.xyz;
        let to_light_vec_norm = normalize(to_light_vec);
        let n_dot_l = max(dot(n, to_light_vec_norm), 0.0);
       
        let bias = shadow_bias;

        // this reduces peter-panning but causes nasty artifacts with current soft shadow solution
        // let bias = mix(shadow_bias, MIN_SHADOW_MAP_BIAS, n_dot_l);

        var shadow_occlusion_acc = 0.0;

        var debug_cascade_index = -1;

        if n_dot_l > 0.0 {
            if light_index < DIRECTIONAL_LIGHT_SHOW_MAP_COUNT {

                var shadow_cascade_index = light_index;
                var shadow_cascade_dist = 0.0;

                // dist = directional_lights.cascades[light_index * MAX_SHADOW_CASCADES].distance.x;
                for (var cascade_index = 0u; cascade_index < MAX_SHADOW_CASCADES; cascade_index = cascade_index + 1u) {
                    shadow_cascade_dist = directional_lights.cascades[light_index * MAX_SHADOW_CASCADES + cascade_index].distance_and_pixel_size.x;
                    debug_cascade_index = i32(shadow_cascade_index);
                    if shadow_cascade_dist == 0.0 || 
                        to_viewer_vec_length < shadow_cascade_dist {

                        if (shadow_cascade_dist == 0.0) {
                            debug_cascade_index = -1;
                        }

                        break;
                    }
                    shadow_cascade_index = shadow_cascade_index + 1u;
                }

                let shadow_cascade_pixel_size = directional_lights.cascades[shadow_cascade_index].distance_and_pixel_size.y;
                let light_space_position_nopersp = directional_lights.cascades[shadow_cascade_index].world_space_to_light_space * vec4<f32>(world_position, 1.0);
                let light_space_position = light_space_position_nopersp / light_space_position_nopersp.w;
                let light_space_position_uv = vec2<f32>(
                    light_space_position.x * 0.5 + 0.5,
                    1.0 - (light_space_position.y * 0.5 + 0.5),
                );
                let current_depth = light_space_position.z; // domain is (0, 1), lower means closer to the light

                // assume we're not in shadow if we're outside the shadow's viewproj area
                if shadow_cascade_dist != 0.0 && to_viewer_vec_length < shadow_cascade_dist && light_space_position.x >= -1.0 && light_space_position.x <= 1.0 && light_space_position.y >= -1.0 && light_space_position.y <= 1.0 && light_space_position.z >= 0.0 && light_space_position.z <= 1.0 {
                    if get_soft_shadows_are_enabled() && to_viewer_vec_length < SOFT_SHADOW_MAX_DISTANCE {
                        // soft shadows code path. costs about 0.15ms extra (per shadow map?) per frame
                        // on an RTX 3060 when compared to hard shadows

                        // these coordinates will distribute the early samples
                        // around the edges of the soft shadow poisson sampling disc
                        var early_test_coords = array<vec2<u32>, 4>(
                            vec2<u32>(0u, 3u),
                            vec2<u32>(1u, 3u),
                            vec2<u32>(2u, 3u),
                            vec2<u32>(3u, 3u)
                        );

                        let max_sample_jitter = get_soft_shadow_factor() / shadow_cascade_pixel_size;

                        for (var i = 0; i < 4; i++) {
                            let base_sample_jitter = get_soft_shadow_sample_jitter(early_test_coords[i], random_jitter, 4u);
                            // TODO: multiply by current_depth to get softer shadows at a distance?
                            let sample_jitter = base_sample_jitter * max_sample_jitter;
                            let closest_depth = textureSampleLevel(
                                directional_shadow_map_textures,
                                shadow_map_sampler,
                                light_space_position_uv + sample_jitter,
                                i32(light_index + shadow_cascade_index),
                                0.0
                            ).r;

                            if current_depth - bias < closest_depth {
                                shadow_occlusion_acc = shadow_occlusion_acc + 0.25;
                            }
                        }

                        // if the early test finds the fragment to be completely in shadow, 
                        // completely in light, or its surface isn't facing the light (n_dot_l =< 0)
                        // then skip the extra work that we do to soften the penumbra
                        if (shadow_occlusion_acc - 1.0) * shadow_occlusion_acc * n_dot_l != 0.0 {
                            
                            if soft_shadow_grid_dims > 0u {
                                // TODO: don't clear shadow_occlusion_acc, we can perserve it and perform fewer samples here
                                // (skip the samples already done by early test) for a theoretically equivalent level of quality
                                // and decent performance boost if soft_shadow_grid_dims isn't too high
                                shadow_occlusion_acc = 0.0;
                            }

                            for (var i = 0u; i < soft_shadow_grid_dims; i++) {
                                for (var j = 0u; j < soft_shadow_grid_dims; j++) {
                                    let coord = vec2<u32>(i, j);
                                    let base_sample_jitter = get_soft_shadow_sample_jitter(coord, random_jitter, soft_shadow_grid_dims);
                                    // TODO: multiply by current_depth to get softer shadows at a distance?
                                    let sample_jitter = base_sample_jitter * max_sample_jitter;
                                    let closest_depth = textureSampleLevel(
                                        directional_shadow_map_textures,
                                        shadow_map_sampler,
                                        light_space_position_uv + sample_jitter,
                                        i32(light_index + shadow_cascade_index),
                                        0.0
                                    ).r;
                                    
                                    if current_depth - bias < closest_depth {
                                        shadow_occlusion_acc = shadow_occlusion_acc + (1.0 / f32(soft_shadow_grid_dims * soft_shadow_grid_dims));
                                    }
                                }
                            }
                        }
                    } else {
                        // hard shadows
                        let closest_depth = textureSampleLevel(
                            directional_shadow_map_textures,
                            shadow_map_sampler,
                            light_space_position_uv,
                            i32(light_index + shadow_cascade_index),
                            0.0
                        ).r;
                        if current_depth - bias < closest_depth {
                            shadow_occlusion_acc = 1.0;
                        }
                    }
                } else {
                    shadow_occlusion_acc = 1.0;
                }
            }
        }
        
        var shadow_occlusion_factor = shadow_occlusion_acc;
        total_shadow_occlusion_acc = total_shadow_occlusion_acc + shadow_occlusion_acc;
        total_light_count = total_light_count + 1u;        

        if shadow_occlusion_factor < epsilon {
                continue;
        }

        let light_attenuation_factor = 1.0;

        var light_irradiance = compute_direct_lighting(
            world_normal,
            to_viewer_vec,
            to_light_vec_norm,
            light_color_scaled,
            light_attenuation_factor,
            base_color,
            roughness,
            metallicness,
            f0
        );

        if get_cascade_debug_enabled() {
            // let light_irradiance = vec3(f32(debug_cascade_index % 1u), f32(debug_cascade_index % 2u), f32(debug_cascade_index % 3u));
            // let light_irradiance = vec3(dist / 100.0, dist / 100.0, dist / 100.0);
            light_irradiance = vec3(0.0, 0.0, 0.0);
            if debug_cascade_index == 0 {
                light_irradiance = vec3(0.0, 1.0, 0.0);
            }
            if debug_cascade_index == 1 {
                light_irradiance = vec3(0.0, 0.0, 1.0);
            }
            if debug_cascade_index == 2 {
                light_irradiance = vec3(1.0, 0.0, 1.0);
            }
            if debug_cascade_index == 3 {
                light_irradiance = vec3(1.0, 0.0, 0.0);
            }
        }

        total_light_irradiance = total_light_irradiance + light_irradiance * shadow_occlusion_factor;
    }

    if get_shadow_debug_enabled() {
        total_shadow_occlusion_acc = total_shadow_occlusion_acc / f32(total_light_count);

        // https://en.wikipedia.org/wiki/Luma_(video)
        let light_brightness = dot(total_light_irradiance, vec3<f32>(0.2126, 0.7152, 0.0722));
        let color = light_brightness * total_shadow_occlusion_acc;

        var out: FragmentOutput;
        out.color = vec4<f32>(color, color, color, 1.0);
        return out;
    }

    let fresnel_ambient = fresnel_func_schlick_with_roughness(n_dot_v, f0, a);
    // mip level count - 1
    
    // let pre_filtered_color = textureSample(
    //     specular_env_map_texture,
    //     specular_env_map_sampler,
    //     world_normal_to_cubemap_vec(reflection_vec)
    // ).rgb;

    let ambient_specular_irradiance = pre_filtered_color * (fresnel_ambient * brdf_lut_res.r + brdf_lut_res.g);

    let kd_ambient = (vec3<f32>(1.0) - fresnel_ambient) * (1.0 - metallicness);

    let ambient_diffuse_irradiance = env_map_diffuse_irradiance * base_color;

    let ambient_irradiance_pre_ao = (kd_ambient * ambient_diffuse_irradiance + ambient_specular_irradiance);
    let ambient_irradiance = mix(
        ambient_irradiance_pre_ao,
        ambient_irradiance_pre_ao * ambient_occlusion,
        occlusion_strength
    );
    // let ambient_irradiance = ambient_irradiance_pre_ao;

    let combined_irradiance_hdr = ambient_irradiance + total_light_irradiance + emissive;
    // let combined_irradiance_hdr = total_light_irradiance;
    // let combined_irradiance_ldr = (combined_irradiance_hdr / (combined_irradiance_hdr + vec3<f32>(1.0, 1.0, 1.0))) + emissive;

    // let hi = textureSample(shadow_map_texture, shadow_map_sampler, vec2<f32>(0.1, 0.1));

    // let final_color = vec4<f32>(combined_irradiance_ldr, 1.0);
    let final_color = vec4<f32>(combined_irradiance_hdr, 1.0);

    var out: FragmentOutput;
    out.color = final_color;
    return out;
}

@fragment
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
        sqrt(1.0 - clamp(normal_map_normal.x * normal_map_normal.x - normal_map_normal.y * normal_map_normal.y, 0.0, 1.0))
    );
    // normal scale helpful comment:
    // https://github.com/KhronosGroup/glTF/issues/885#issuecomment-288320363
    let transformed_normal = normalize(
        tbn * normalize(
            tangent_space_normal * vec3<f32>(in.normal_scale, in.normal_scale, 1.0)
        )
    );

    //  var out: FragmentOutput;
    // out.color = vec4<f32>(in.object_tangent, 1.0);;
    // return out;

    return do_fragment_shade(
        in.world_position,
        transformed_normal,
        in.tex_coords,
        in.vertex_color,
        CAMERA.position.xyz,
        in.base_color_factor,
        in.emissive_factor,
        in.metallicness_factor,
        in.roughness_factor,
        in.occlusion_strength,
        in.alpha_cutoff
    );
}

@fragment
fn depth_prepass_fs_main(in: VertexOutput) -> FragmentOutput {
    let base_color_t = textureSample(
        diffuse_texture,
        diffuse_sampler,
        in.tex_coords
    );

    if base_color_t.a <= in.alpha_cutoff {
        discard;
    }

    var out: FragmentOutput;
    out.color = vec4(0.0, 0.0, 0.0, 0.0);
    return out;
}
