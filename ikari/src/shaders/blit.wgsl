struct BloomDownscaleConfig {
    src_texture_resolution: vec4<f32>,
}

struct BloomUpscaleConfig {
    filter_radius: vec4<f32>,
}

struct ToneMappingConfigUniform {
    exposure_bloom_factor: vec4<f32>,
}

@group(1) @binding(0)
var<uniform> BLOOM_DOWNSCALE_CONFIG: BloomDownscaleConfig;

@group(1) @binding(0)
var<uniform> BLOOM_UPSCALE_CONFIG: BloomUpscaleConfig;

@group(1) @binding(0)
var<uniform> TONE_MAPPING_CONFIG: ToneMappingConfigUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

// should be called with 3 vertex indices: 0, 1, 2
// draws one large triangle over the clip space like this:
// (the asterisks represent the clip space bounds)
//-1,1           1,1
// ---------------------------------
// |              *              .
// |              *           .
// |              *        .
// |              *      .
// |              *    . 
// |              * .
// |***************
// |            . 1,-1 
// |          .
// |       .
// |     .
// |   .
// |.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    out.position = vec4<f32>(
        tc.x * 2.0 - 1.0,
        1.0 - tc.y * 2.0,
        0.0,
        1.0
    );
    out.tex_coords = tc;
    return out;
}

@group(0) @binding(0)
var texture_1: texture_2d<f32>;
@group(0) @binding(1)
var sampler_1: sampler;
@group(0) @binding(2)
var texture_2: texture_2d<f32>;
@group(0) @binding(3)
var sampler_2: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(texture_1, sampler_1, in.tex_coords);
}

@fragment
fn surface_blit_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(texture_1, sampler_1, in.tex_coords);
}

@fragment
fn tone_mapping_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let exposure = TONE_MAPPING_CONFIG.exposure_bloom_factor.x;
    let bloom_factor = TONE_MAPPING_CONFIG.exposure_bloom_factor.y;
    let shaded_color = textureSample(texture_1, sampler_1, in.tex_coords).rgb;
    let bloom_color = textureSample(texture_2, sampler_2, in.tex_coords).rgb;

    var final_color_hdr: vec3<f32>;
    if bloom_factor == -1.0 {
        final_color_hdr = shaded_color + bloom_color;
    } else {
        final_color_hdr = mix(shaded_color, bloom_color, bloom_factor);
    }
    return vec4<f32>(1.0 - exp(-final_color_hdr * exposure), 1.0);
}

// https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
@fragment 
fn bloom_downscale_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let src_texel_size = 1.0 / BLOOM_DOWNSCALE_CONFIG.src_texture_resolution.xy;
    let x = src_texel_size.x;
    let y = src_texel_size.y;

    // Take 13 samples around current texel:
    // a - b - c
    // - j - k -
    // d - e - f
    // - l - m -
    // g - h - i
    // === ('e' is the current texel) ===
    let a = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x - 2.0 * x, in.tex_coords.y + 2.0 * y)).rgb;
    let b = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x,           in.tex_coords.y + 2.0 * y)).rgb;
    let c = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x + 2.0 * x, in.tex_coords.y + 2.0 * y)).rgb;

    let d = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x - 2.0 * x, in.tex_coords.y)).rgb;
    let e = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x,           in.tex_coords.y)).rgb;
    let f = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x + 2.0 * x, in.tex_coords.y)).rgb;

    let g = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x - 2.0 * x, in.tex_coords.y - 2.0 * y)).rgb;
    let h = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x,           in.tex_coords.y - 2.0 * y)).rgb;
    let i = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x + 2.0 * x, in.tex_coords.y - 2.0 * y)).rgb;

    let j = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x - x, in.tex_coords.y + y)).rgb;
    let k = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x + x, in.tex_coords.y + y)).rgb;
    let l = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x - x, in.tex_coords.y - y)).rgb;
    let m = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x + x, in.tex_coords.y - y)).rgb;

    // Apply weighted distribution:
    // 0.5 + 0.125 + 0.125 + 0.125 + 0.125 = 1
    // a,b,d,e * 0.125
    // b,c,e,f * 0.125
    // d,e,g,h * 0.125
    // e,f,h,i * 0.125
    // j,k,l,m * 0.5
    // This shows 5 square areas that are being sampled. But some of them overlap,
    // so to have an energy preserving downsample we need to make some adjustments.
    // The weights are the distributed, so that the sum of j,k,l,m (e.g.)
    // contribute 0.5 to the final color output. The code below is written
    // to effectively yield this sum. We get:
    // 0.125*5 + 0.03125*4 + 0.0625*4 = 1
    let result = 0.125*e + 0.03125*(a+c+g+i) + 0.0625*(b+d+f+h) + 0.125*(j+k+l+m);
    return vec4<f32>(result, 1.0);
}

// https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
@fragment 
fn bloom_upscale_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // The filter kernel is applied with a radius, specified in texture
    // coordinates, so that the radius will vary across mip resolutions.
    let x = BLOOM_UPSCALE_CONFIG.filter_radius.x;
    let y = BLOOM_UPSCALE_CONFIG.filter_radius.x;

    // Take 9 samples around current texel:
    // a - b - c
    // d - e - f
    // g - h - i
    // === ('e' is the current texel) ===     
    let a = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x - x, in.tex_coords.y + y)).rgb;
    let b = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x,     in.tex_coords.y + y)).rgb;
    let c = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x + x, in.tex_coords.y + y)).rgb;

    let d = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x - x, in.tex_coords.y)).rgb;
    let e = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x,     in.tex_coords.y)).rgb;
    let f = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x + x, in.tex_coords.y)).rgb;

    let g = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x - x, in.tex_coords.y - y)).rgb;
    let h = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x,     in.tex_coords.y - y)).rgb;
    let i = textureSample(texture_1, sampler_1, vec2<f32>(in.tex_coords.x + x, in.tex_coords.y - y)).rgb;

    // Apply weighted distribution, by using a 3x3 tent filter:
    //  1   | 1 2 1 |
    // -- * | 2 4 2 |
    // 16   | 1 2 1 |
    let result = (1.0 / 16.0) * (4.0*e + 2.0*(b+d+f+h) + 1.0*(a+c+g+i));
    return vec4<f32>(result, 1.0);
}

// BRDF LUT:

const pi: f32 = 3.141592653589793;
const two_pi: f32 = 6.283185307179586;
const half_pi: f32 = 1.570796326794897;
const epsilon: f32 = 0.00001;

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
    if abs(n.z) < 0.999 {
        up = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        up = vec3<f32>(1.0, 0.0, 0.0);
    };
    let tangent = normalize(cross(up, n));
    let bitangent = normalize(cross(n, tangent));

    let sample_vec = tangent * h.x + bitangent * h.y + n * h.z;
    return normalize(sample_vec);
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

// fn van_der_corput(n_in: u32, base: u32) -> f32 {
//     var n = n_in;
//     var inv_base = 1.0 / f32(base);
//     var denom = 1.0;
//     var result = 0.0;

//     for (var i = 0u; i < 32u; i = i + 1u) {
//         if (n > 0u) {
//             denom = f32(n) % 2.0;
//             result = result + denom * inv_base;
//             inv_base = inv_base / 2.0;
//             n = u32(f32(n) / 2.0);
//         }
//     }

//     return result;
// }

fn hammersley(
    i_u: u32,
    num_samples_u: u32,
) -> vec2<f32> {
    let i = f32(i_u);
    let num_samples = f32(num_samples_u);
    return vec2<f32>(i / num_samples, radical_inverse_vdc(i_u));
    // return vec2<f32>(i / num_samples, van_der_corput(i_u, 2u));
}

                //0.9487        0.9162
fn integrate_brdf(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    var a = 0.0;
    var b = 0.0;
    let sample_count = 1024u;
    for (var i = 0u; i < sample_count; i = i + 1u) {
        let x_i = hammersley(i, sample_count);
        let h = importance_sampled_ggx(x_i, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);

        let n_dot_l = max(l.z, 0.0);
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);

        if n_dot_l > 0.0 {
            let k = geometry_func_schlick_ggx_k_ibl(roughness); // 0.41
            let g = geometry_func_smith_ggx(k, n, v, l); // 0.95
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v); // 0.956
            let f_c = pow(1.0 - v_dot_h, 5.0); // 3.5e-7

            a = a + (1.0 - f_c) * g_vis;
            b = b + f_c * g_vis;
        }
    }

    a = a / f32(sample_count);
    b = b / f32(sample_count);
    return vec2<f32>(a, b);
}

@fragment
fn brdf_lut_gen_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(integrate_brdf(in.tex_coords.x, in.tex_coords.y), 0.0, 1.0);
}
