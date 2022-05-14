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

fn world_pos_to_cubemap_vec(world_pos: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(-world_pos.x, world_pos.y, world_pos.z);
}

[[group(0), binding(0)]]
var cubemap_texture: texture_cube<f32>;

[[group(0), binding(1)]]
var cubemap_sampler: sampler;

[[stage(fragment)]]
fn cubemap_fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let col = textureSample(cubemap_texture, cubemap_sampler, world_pos_to_cubemap_vec(in.world_position));
    return vec4<f32>(col.x % 1.01, col.y % 1.01, col.z % 1.01, 1.0);
}

// for mapping equirectangular to cubemap

let pi: f32 = 3.141592653589793;
let two_pi: f32 = 6.283185307179586;
let half_pi: f32 = 1.570796326794897;

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
    let col_gamma_corrected = vec4<f32>(pow(col.xyz, vec3<f32>(1.0 / 2.2)), 1.0);
    return col;
    // return vec4<f32>(col.x % 1.01, col.y % 1.01, col.z % 1.01, 1.0);
}

[[stage(fragment)]]
fn env_map_gen_fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let normal = normalize(in.world_position);
    var irradiance: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
    let up = normalize(cross(normal, right));

    let phi_sample_count = 256.0;
    let theta_sample_count = 72.0;
    let phi_sample_delta = two_pi / phi_sample_count;
    let theta_sample_delta = half_pi / theta_sample_count;
    for (var phi = 0.0; phi < two_pi; phi = phi + phi_sample_delta) {
        for (var theta = 0.0; theta < half_pi; theta = theta + theta_sample_delta) {
            let sample_dir = normalize(
                right * sin(theta) * cos(phi) +
                up * sin(theta) * sin(phi) +
                normal * cos(theta)
            );
            irradiance = irradiance +
                textureSample(
                    cubemap_texture, cubemap_sampler, 
                    world_pos_to_cubemap_vec(sample_dir)
                    // vec3<f32>(-sample_dir.x, sample_dir.y, sample_dir.z)
                ).xyz * 
                cos(theta) * 
                // cos(theta);
                sin(theta);
        }
    }

    let total_samples = phi_sample_count * theta_sample_count;
    irradiance = pi * irradiance * (1.0 / total_samples);

    return vec4<f32>(irradiance, 1.0);
}