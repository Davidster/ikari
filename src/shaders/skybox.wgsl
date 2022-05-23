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

struct RougnessInput {
    value: f32;
};
[[group(1), binding(1)]]
var<uniform> roughness_input: RougnessInput;

struct VertexInput {
    [[location(0)]] object_position: vec3<f32>;
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
    out.clip_position = vec4<f32>(clip_position.x, clip_position.y, 0.0, clip_position.w);
    out.world_position = vshader_input.object_position;
    return out;
}

// cubemap version

fn world_normal_to_cubemap_vec(world_pos: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(-world_pos.x, world_pos.y, world_pos.z);
}

[[group(0), binding(0)]]
var cubemap_texture: texture_cube<f32>;

[[group(0), binding(1)]]
var cubemap_sampler: sampler;

[[stage(fragment)]]
fn cubemap_fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // let col = textureSampleLevel(cubemap_texture, cubemap_sampler, world_normal_to_cubemap_vec(in.world_position), 0.0);
    let col = textureSample(cubemap_texture, cubemap_sampler, world_normal_to_cubemap_vec(in.world_position));
    return vec4<f32>(col.x % 1.01, col.y % 1.01, col.z % 1.01, 1.0);
}

// for mapping equirectangular to cubemap

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

// https://learnopengl.com/PBR/IBL/Specular-IBL
fn importance_sampled_ggx(x_i: vec2<f32>, n: vec3<f32>, a: f32) -> vec3<f32> {
    let a2 = a * a;
    let phi = two_pi * x_i.x;
    let cos_theta = sqrt((1.0 - x_i.y) / (1.0 + (a2 * a2 - 1.0) * x_i.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let h = vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta,
    );

    var up: vec3<f32>;
    if (abs(n.z) < 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        up = vec3<f32>(1.0, 0.0, 0.0);
    };
    let tangent = normalize(cross(up, n));
    let bitangent = normalize(cross(n, tangent));

    let sample_vec = tangent * h.x + bitangent * h.y + n * h.z;
    return normalize(sample_vec);
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

// in radians
fn angle_modulo(angle: f32) -> f32 {
    return (angle + two_pi) % two_pi;
}

fn latlng_to_uv(latitude: f32, longitude: f32) -> vec2<f32> {
    return vec2<f32>(
        angle_modulo(longitude) / two_pi,
        -(latitude - (pi / 2.0)) / pi,
    );
}

[[group(0), binding(0)]]
var equirectangular_texture: texture_2d<f32>;

[[group(0), binding(1)]]
var equirectangular_sampler: sampler;

[[stage(fragment)]]
fn equirectangular_to_cubemap_fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let latitude = atan2(
        in.world_position.y,
        sqrt(in.world_position.x * in.world_position.x + in.world_position.z * in.world_position.z)
    );
    let longitude = atan2(in.world_position.z, in.world_position.x);

    let col = textureSample(equirectangular_texture, equirectangular_sampler, latlng_to_uv(latitude, longitude));
    // let col_gamma_corrected = vec4<f32>(pow(col.rgb, vec3<f32>(1.0 / 2.2)), 1.0);
    return col;
    // return vec4<f32>(col.x % 1.01, col.y % 1.01, col.z % 1.01, 1.0);
}

[[stage(fragment)]]
fn diffuse_env_map_gen_fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let normal = normalize(in.world_position);
    var irradiance = vec3<f32>(0.0, 0.0, 0.0);

    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
    let up = normalize(cross(normal, right));

    let phi_sample_count = 256.0;
    let theta_sample_count = 72.0;
    let phi_sample_delta = two_pi / phi_sample_count;
    let theta_sample_delta = half_pi / theta_sample_count;
    for (var phi = 0.0; phi < two_pi; phi = phi + phi_sample_delta) {
        for (var theta = 0.0; theta < half_pi; theta = theta + theta_sample_delta) {
            let sample_dir = normalize(
                right * sin(theta) * cos(phi) + up * sin(theta) * sin(phi) + normal * cos(theta)
            );
            irradiance = irradiance + textureSample(
                cubemap_texture,
                cubemap_sampler,
                world_normal_to_cubemap_vec(sample_dir)
                    // vec3<f32>(-sample_dir.x, sample_dir.y, sample_dir.z)
            ).rgb * cos(theta) * // cos(theta);
                sin(theta);
        }
    }

    let total_samples = phi_sample_count * theta_sample_count;
    irradiance = pi * irradiance * (1.0 / total_samples);

    return vec4<f32>(irradiance, 1.0);
}

[[stage(fragment)]]
fn specular_env_map_gen_fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let normal = normalize(in.world_position);
    let reflect_dir = normal;
    let view_dir = reflect_dir;

    let roughness = roughness_input.value;

    var total_pre_filtered_color = vec3<f32>(0.0, 0.0, 0.0);
    var total_weight = 0.0;
    // let sample_count = 1024u;
    let sample_count = 4096u;
    for (var i = 0u; i < sample_count; i = i + 1u) {
        let x_i = hammersley(i, sample_count);
        let h = importance_sampled_ggx(x_i, normal, roughness);
        let l = normalize(2.0 * dot(view_dir, h) * h - view_dir);

        let n_dot_l = max(dot(normal, l), 0.0);
        total_pre_filtered_color = total_pre_filtered_color + textureSample(
            cubemap_texture,
            cubemap_sampler,
            world_normal_to_cubemap_vec(l)
        ).rgb * n_dot_l;
        total_weight = total_weight + n_dot_l;
    }
    let pre_filtered_color = total_pre_filtered_color / total_weight;

    return vec4<f32>(pre_filtered_color.rgb, 1.0);
}