// Vertex shader
// TODO: pass camera vars as separate variables to be able to convert from screen coordinates
// (object_position) into world space coordinates
// struct CameraUniform {
//     // from camera to screen
//     proj: mat4x4<f32>,
//     // from screen to camera
//     proj_inv: mat4x4<f32>,
//     // from world to camera
//     view: mat4x4<f32>,
// };
struct CameraUniform {
    view_proj: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec3<f32>;
};

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
[[stage(vertex)]]
fn vs_main(
    [[builtin(vertex_index)]] vertex_index: u32,
) -> VertexOutput {
    let tmp1 = i32(vertex_index) / 2;
    let tmp2 = i32(vertex_index) & 1;
    let object_position = vec4<f32>(
        f32(tmp1) * 4.0 - 1.0,
        f32(tmp2) * 4.0 - 1.0,
        1.0,
        1.0
    );

    let inv_view_proj = transpose(mat3x3<f32>(camera.view_proj.x.xyz, camera.view_proj.y.xyz, camera.view_proj.z.xyz));
    let world_position = inv_view_proj * object_position.xyz;
    // let world_position = camera.view_proj * object_position;

    var out: VertexOutput;
    out.clip_position = object_position;
    out.world_position = world_position;
    return out;
}

[[group(0), binding(0)]]
var r_texture: texture_cube<f32>;

[[group(0), binding(1)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(r_texture, r_sampler, in.world_position);
}
