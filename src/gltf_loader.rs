use super::*;
use anyhow::Result;
use cgmath::Matrix4;
use gltf::Gltf;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GltfVertex {
    position: [f32; 3],
    normal: Option<[f32; 3]>,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct GltfMesh {
    vertices: Vec<GltfVertex>,
    indices: Option<Vec<u16>>,
    instance_transforms: Vec<Matrix4<f32>>,
}

fn gltf_transform_to_mat4(gltf_transform: gltf::scene::Transform) -> Matrix4<f32> {
    match gltf_transform {
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            let transform = transform::Transform::new();
            transform.set_position(translation.into());
            transform.set_rotation(rotation.into());
            transform.set_scale(scale.into());
            transform.matrix.get()
        }
        gltf::scene::Transform::Matrix { matrix } => Matrix4::from(matrix),
    }
}

pub fn load_gltf(path: &str) -> Result<Vec<GltfMesh>> {
    let (document, buffers, images) = gltf::import(path)?;

    // dbg!(&document);

    // let first_buffer = &buffers[0];
    // let buffer_arr: &[u8] = first_buffer;
    // let buffer_data: &[[f32; 3]] = bytemuck::cast_slice(buffer_arr);
    // dbg!(buffer_data);

    Ok(document
        .meshes()
        .map(|mesh| {
            let triangles: Vec<_> = mesh
                .primitives()
                .filter(|prim| prim.mode() == gltf::mesh::Mode::Triangles)
                .collect();
            let first_triangle = &triangles[0];

            let get_buffer_slice_from_accessor = |accessor: gltf::Accessor| {
                let buffer_view = accessor.view().unwrap();
                let buffer = &buffers[buffer_view.buffer().index()];
                let byte_range_start = buffer_view.offset() + accessor.offset();
                let byte_range_end = byte_range_start + (accessor.size() * accessor.count());
                let byte_range = byte_range_start..byte_range_end;
                &buffer[byte_range]
            };

            let vertices: &[[f32; 3]] = first_triangle
                .attributes()
                .find(|(semantic, _)| *semantic == gltf::Semantic::Positions)
                .map(|(_, accessor)| {
                    let data_type = accessor.data_type();
                    if data_type != gltf::accessor::DataType::F32 {
                        panic!("Expected f32 data but found: {:?}", data_type);
                    }
                    bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor))
                })
                .unwrap();

            let normals: Option<&[[f32; 3]]> = first_triangle
                .attributes()
                .find(|(semantic, _)| *semantic == gltf::Semantic::Normals)
                .map(|(_, accessor)| {
                    let data_type = accessor.data_type();
                    if data_type != gltf::accessor::DataType::F32 {
                        panic!("Expected f32 data but found: {:?}", data_type);
                    }
                    bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor))
                });

            let mut indices: Vec<u16> = Vec::new(); // lifetime bs
            let indices: Option<&[u16]> = first_triangle.indices().map(|accessor| {
                // let buffer = &buffers[accessor.view().unwrap().index()];
                // let buffer_slice: &[u8] =
                //     &buffer[accessor.offset()..(accessor.count() * accessor.size())];
                let data_type = accessor.data_type();
                let buffer_slice = get_buffer_slice_from_accessor(accessor);

                match data_type {
                    gltf::accessor::DataType::U16 => bytemuck::cast_slice(buffer_slice),
                    gltf::accessor::DataType::U8 => {
                        indices = buffer_slice.iter().map(|&x| x as u16).collect::<Vec<_>>();
                        bytemuck::cast_slice(&indices)
                    }
                    data_type => panic!("Expected u16 or u8 indices but found: {:?}", data_type),
                }
            });

            GltfMesh {
                vertices: vertices
                    .iter()
                    .enumerate()
                    .map(|(i, position)| GltfVertex {
                        position: *position,
                        normal: normals.and_then(|normals| normals.get(i)).copied(),
                    })
                    .collect(),
                indices: indices.map(|indices| indices.to_vec()),
                instance_transforms: document
                    .nodes()
                    .filter(|node| {
                        node.mesh().is_some() && node.mesh().unwrap().index() == mesh.index()
                    })
                    .map(|node| gltf_transform_to_mat4(node.transform()))
                    .collect(),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use cgmath::Vector3;

    use super::*;

    #[test]
    fn test_triangle_without_indices() -> Result<()> {
        let mesh =
            load_gltf("./src/models/gltf/TriangleWithoutIndices/TriangleWithoutIndices.gltf")?;
        assert_eq!(
            mesh,
            vec![GltfMesh {
                vertices: vec![
                    GltfVertex {
                        position: [0.0, 0.0, 0.0],
                        normal: None,
                    },
                    GltfVertex {
                        position: [1.0, 0.0, 0.0],
                        normal: None,
                    },
                    GltfVertex {
                        position: [0.0, 1.0, 0.0],
                        normal: None,
                    },
                ],
                indices: None,
                instance_transforms: vec![Matrix4::identity()],
            }]
        );
        Ok(())
    }

    #[test]
    fn test_triangle_with_indices() -> Result<()> {
        let mesh = load_gltf("./src/models/gltf/Triangle/Triangle.gltf")?;
        assert_eq!(
            mesh,
            vec![GltfMesh {
                vertices: vec![
                    GltfVertex {
                        position: [0.0, 0.0, 0.0],
                        normal: None,
                    },
                    GltfVertex {
                        position: [1.0, 0.0, 0.0],
                        normal: None,
                    },
                    GltfVertex {
                        position: [0.0, 1.0, 0.0],
                        normal: None,
                    },
                ],
                indices: Some(vec![0, 1, 2]),
                instance_transforms: vec![Matrix4::identity()],
            }]
        );
        Ok(())
    }

    #[test]
    fn test_simple_meshes() -> Result<()> {
        // there's two meshes in this file
        let mesh = load_gltf("./src/models/gltf/SimpleMeshes/SimpleMeshes.gltf")?;
        assert_eq!(
            mesh,
            vec![GltfMesh {
                vertices: vec![
                    GltfVertex {
                        position: [0.0, 0.0, 0.0],
                        normal: Some([0.0, 0.0, 1.0]),
                    },
                    GltfVertex {
                        position: [1.0, 0.0, 0.0],
                        normal: Some([0.0, 0.0, 1.0]),
                    },
                    GltfVertex {
                        position: [0.0, 1.0, 0.0],
                        normal: Some([0.0, 0.0, 1.0]),
                    },
                ],
                indices: Some(vec![0, 1, 2]),
                instance_transforms: vec![
                    Matrix4::identity(),
                    make_translation_matrix(Vector3::new(1.0, 0.0, 0.0))
                ],
            }]
        );
        Ok(())
    }

    #[test]
    fn test_texture_coordinate() -> Result<()> {
        // there's two meshes in this file
        let mesh = load_gltf("./src/models/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf")?;
        assert_eq!(dbg!(mesh), vec![]);
        Ok(())
    }
}
