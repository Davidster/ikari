use anyhow::Result;
use gltf::Gltf;

pub fn load_gltf(path: &str) -> Result<Vec<[f32; 3]>> {
    let (document, buffers, images) = gltf::import(path)?;

    // let first_buffer = &buffers[0];
    // let buffer_arr: &[u8] = first_buffer;
    // let buffer_data: &[[f32; 3]] = bytemuck::cast_slice(buffer_arr);
    // dbg!(buffer_data);

    let first_mesh = document
        .scenes()
        .next()
        .unwrap()
        .nodes()
        .next()
        .unwrap()
        .mesh()
        .unwrap();

    let triangles: Vec<_> = first_mesh
        .primitives()
        .filter(|prim| prim.mode() == gltf::mesh::Mode::Triangles)
        .collect();
    let first_triangle = &triangles[0];
    let attribute = first_triangle.attributes().next().unwrap();
    let accessor = attribute.1;
    let data_type = accessor.data_type();
    if data_type != gltf::accessor::DataType::F32 {
        panic!("I only understand f32 data");
    }
    let buffer_slice: &[u8] = &buffers[0][accessor.offset()..(accessor.count() * accessor.size())];
    let triangle_data: &[[f32; 3]] = bytemuck::cast_slice(buffer_slice);

    Ok(triangle_data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_gltf() {
        let positions = load_gltf("./src/models/gltf/TriangleWithoutIndices.gltf").unwrap();
        assert_eq!(
            positions,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        );
    }
}
