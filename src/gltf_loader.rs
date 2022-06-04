use super::*;
use anyhow::Result;
use cgmath::Matrix4;
use gltf::Gltf;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GltfVertex {
    position: [f32; 3],
    normal: Option<[f32; 3]>,
    tex_coord: Option<[f32; 2]>,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct GltfMesh {
    name: Option<String>,
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

// TODO: return result instead of panic
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

            let positions: &[[f32; 3]] = first_triangle
                .attributes()
                .find(|(semantic, _)| *semantic == gltf::Semantic::Positions)
                .map(|(_, accessor)| {
                    let data_type = accessor.data_type();
                    let dimensions = accessor.dimensions();
                    if dimensions != gltf::accessor::Dimensions::Vec3 {
                        panic!("Expected vec3 data but found: {:?}", dimensions);
                    }
                    if data_type != gltf::accessor::DataType::F32 {
                        panic!("Expected f32 data but found: {:?}", data_type);
                    }
                    bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor))
                })
                .unwrap_or_else(|| panic!("No positions found"));

            let normals: Option<&[[f32; 3]]> = first_triangle
                .attributes()
                .find(|(semantic, _)| *semantic == gltf::Semantic::Normals)
                .map(|(_, accessor)| {
                    let data_type = accessor.data_type();
                    let dimensions = accessor.dimensions();
                    if dimensions != gltf::accessor::Dimensions::Vec3 {
                        panic!("Expected vec3 data but found: {:?}", dimensions);
                    }
                    if data_type != gltf::accessor::DataType::F32 {
                        panic!("Expected f32 data but found: {:?}", data_type);
                    }
                    bytemuck::cast_slice(get_buffer_slice_from_accessor(accessor))
                });

            let tex_coords: Option<&[[f32; 2]]> = first_triangle
                .attributes()
                .find(|(semantic, _)| *semantic == gltf::Semantic::TexCoords(0))
                .map(|(_, accessor)| {
                    let data_type = accessor.data_type();
                    let dimensions = accessor.dimensions();
                    if dimensions != gltf::accessor::Dimensions::Vec2 {
                        panic!("Expected vec2 data but found: {:?}", dimensions);
                    }
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
                name: mesh.name().map(String::from),
                // TODO: use izip! macro from itertools instead of indexing
                vertices: positions
                    .iter()
                    .enumerate()
                    .map(|(i, position)| GltfVertex {
                        position: *position,
                        normal: normals.and_then(|normals| normals.get(i)).copied(),
                        tex_coord: tex_coords.and_then(|tex_coords| tex_coords.get(i)).copied(),
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
        let meshes =
            load_gltf("./src/models/gltf/TriangleWithoutIndices/TriangleWithoutIndices.gltf")?;
        assert_eq!(
            meshes,
            vec![GltfMesh {
                name: None,
                vertices: vec![
                    GltfVertex {
                        position: [0.0, 0.0, 0.0],
                        normal: None,
                        tex_coord: None,
                    },
                    GltfVertex {
                        position: [1.0, 0.0, 0.0],
                        normal: None,
                        tex_coord: None,
                    },
                    GltfVertex {
                        position: [0.0, 1.0, 0.0],
                        normal: None,
                        tex_coord: None,
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
        let meshes = load_gltf("./src/models/gltf/Triangle/Triangle.gltf")?;
        assert_eq!(
            meshes,
            vec![GltfMesh {
                name: None,
                vertices: vec![
                    GltfVertex {
                        position: [0.0, 0.0, 0.0],
                        normal: None,
                        tex_coord: None,
                    },
                    GltfVertex {
                        position: [1.0, 0.0, 0.0],
                        normal: None,
                        tex_coord: None,
                    },
                    GltfVertex {
                        position: [0.0, 1.0, 0.0],
                        normal: None,
                        tex_coord: None,
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
        // there's two nodes in this file
        let meshes = load_gltf("./src/models/gltf/SimpleMeshes/SimpleMeshes.gltf")?;
        assert_eq!(
            meshes,
            vec![GltfMesh {
                name: None,
                vertices: vec![
                    GltfVertex {
                        position: [0.0, 0.0, 0.0],
                        normal: Some([0.0, 0.0, 1.0]),
                        tex_coord: None,
                    },
                    GltfVertex {
                        position: [1.0, 0.0, 0.0],
                        normal: Some([0.0, 0.0, 1.0]),
                        tex_coord: None,
                    },
                    GltfVertex {
                        position: [0.0, 1.0, 0.0],
                        normal: Some([0.0, 0.0, 1.0]),
                        tex_coord: None,
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
        let meshes =
            load_gltf("./src/models/gltf/TextureCoordinateTest/TextureCoordinateTest.gltf")?;
        let first_mesh = &meshes[0];
        // dbg!(first_mesh);
        assert_eq!(
            *first_mesh,
            GltfMesh {
                name: Some(String::from("TopRightMesh")),
                vertices: vec![
                    GltfVertex {
                        position: [1.2, 0.2000002, -5.2054858e-8,],
                        normal: Some([-2.1316282e-14, 1.07284414e-7, 1.0,],),
                        tex_coord: Some([1.0, 0.39999998,],),
                    },
                    GltfVertex {
                        position: [0.19999981, 1.1999999, -1.5933927e-7,],
                        normal: Some([-2.1316282e-14, 1.07284414e-7, 1.0,],),
                        tex_coord: Some([0.6, 0.0,],),
                    },
                    GltfVertex {
                        position: [0.19999996, 0.20000005, -5.2054858e-8,],
                        normal: Some([-2.1316282e-14, 1.07284414e-7, 1.0,],),
                        tex_coord: Some([0.6, 0.39999998,],),
                    },
                    GltfVertex {
                        position: [1.1999998, 1.2000002, -1.5933927e-7,],
                        normal: Some([-2.1316282e-14, 1.07284414e-7, 1.0,],),
                        tex_coord: Some([1.0, 0.0,],),
                    },
                ],
                indices: Some(vec![0, 1, 2, 3, 1, 0,],),
                instance_transforms: vec![Matrix4::identity()],
            }
        );
        Ok(())
    }
}
