## Latest screenshot:

![image](https://user-images.githubusercontent.com/2389735/162113906-6e92e045-5057-41d0-9fec-258a742522fd.png)

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
- [ ] pass color to flat color shader with uniform
- [ ] merge photosphere and cubemap skybox shaders into one?
- [ ] support gltf format
- [ ] try to implement PBR shaders based on [article](https://learnopengl.com/PBR/Theory), implement some [sample models](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0):
  - [ ] [Triangle Without Indices](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TriangleWithoutIndices)
  - [ ] [Triangle](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/Triangle)
  - [ ] [Simple Meshes](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/SimpleMeshes)
  - [ ] [Texture Coordinate Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TextureCoordinateTest)
  - [ ] [Texture Linear Interpolation Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TextureLinearInterpolationTest)
  - [ ] [Vertex Color Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/VertexColorTest)
  - [ ] [Environment Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/EnvironmentTest)
