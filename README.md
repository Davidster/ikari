## Latest screenshot:

![screenshot](https://user-images.githubusercontent.com/2389735/169865297-34800766-ea06-4f17-b3fe-0f81943a39ef.jpg)

## TODO List
- [x] basic phong shading 
- [x] floor plane with diffuse texture
- [x] wasd movement
- [x] add a debug log and fps counter
- [x] spawn as many planets as I can, make them bounce around the boundaries
- [x] add mipmaps
- [x] add super sampling anti-aliasing
- [x] add a skybox
  - [x] try to use one of my photospheres or panorama from comp-425 - it worked!
- [x] see if rust 1.60 improves the stability of the simulation with the std::Instant changes
  - It did not
- [x] use clippy with this project
- [x] fix simulation stability (dt seems to be wrong!)
- [x] do logarithmic depth buffer to prevent z fighting
- [x] try reverse depth instead of / in addition to logarithmic depth
- [x] support normal maps
- [x] refactor the proj!
- [x] pass color to flat color shader with uniform
- [x] merge photosphere and cubemap skybox shaders into one?
- [x] try to implement PBR shaders based on [article](https://learnopengl.com/PBR/Theory)
- [x] pass roughness, metallicness and ao via textures
- [ ] support gltf format
  - [x] support multiple lights
  - [x] support tangent vector map
  - [x] support srgb
  - [x] support non-texture material params (pass through instance buffer?)
  - [x] why is the lantern mis-aligned?
  - [x] why doesn't the photosphere skybox work on mac?
  - [x] support vertex colors
  - [x] support ambient occlusion strength and normal map strength
  - [ ] support Texture Linear Interpolation Test
  - [ ] load in some [sample models](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0):
    - [x] [Triangle Without Indices](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TriangleWithoutIndices)
    - [x] [Triangle](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/Triangle)
    - [x] [Simple Meshes](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/SimpleMeshes)
    - [x] [Texture Coordinate Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TextureCoordinateTest)
    - [ ] [Texture Linear Interpolation Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TextureLinearInterpolationTest)
    - [x] [Vertex Color Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/VertexColorTest)
    - [x] [Environment Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/EnvironmentTest)
- [ ] improve macos support: 
  - [ ] make sure the skybox rad texture resolution is capped at a sane level; it causes the renderer to break on my m1.
- [ ] support srgb conversions for all unsupported texture types
- [ ] add adaptive exposure based on histogram
- [ ] add bloom effect for emissive materials
- [ ] remove duped shader functions?
