## Latest screenshots:

![screenshot_1](https://user-images.githubusercontent.com/2389735/174690197-1761b4ca-3c93-43c2-ba0f-a17470802613.jpg)
![screenshot_2](https://user-images.githubusercontent.com/2389735/174689921-9aad3283-171a-48ee-9d3a-c544aed2314e.jpg)

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
- [x] support gltf format
  - [x] support multiple lights
  - [x] support tangent vector map
  - [x] support srgb
  - [x] support non-texture material params (pass through instance buffer?)
  - [x] why is the lantern mis-aligned?
  - [x] why doesn't the photosphere skybox work on mac?
  - [x] support vertex colors
  - [x] support ambient occlusion strength and normal map strength
  - [x] support Texture Linear Interpolation Test
  - [x] load in some [sample models](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0):
    - [x] [Triangle Without Indices](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TriangleWithoutIndices)
    - [x] [Triangle](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/Triangle)
    - [x] [Simple Meshes](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/SimpleMeshes)
    - [x] [Texture Coordinate Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TextureCoordinateTest)
    - [x] [Texture Linear Interpolation Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/TextureLinearInterpolationTest)
    - [x] [Vertex Color Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/VertexColorTest)
    - [x] [Environment Test](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/EnvironmentTest)
- [x] why doesn't MetalRoughSpheresNoTextures work?
- [x] add shadows
- [ ] get windows support back by removing the srgb thing
- [ ] add bloom
  - [x] move tone mapping to later pass
  - [x] support light intensity value
- [ ] add adaptive exposure based on histogram

## Low priority
- [ ] Asset loading
  - [ ] glTF pbrSpecularGlossiness (adamHead model)
  - [ ] glTF doubleSided
  - [ ] make sure normal mapping is working (make the NormalTangentTest work????)
  - [ ] support srgb conversions for all unsupported texture types
- [ ] Shadows
  - [ ] Try out PCSS: [link 1](https://download.nvidia.com/developer/SDK/Individual_Samples/MEDIA/docPix/docs/PCSS.pdf), [link 2](https://developer.download.nvidia.com/whitepapers/2008/PCSS_Integration.pdf). Judging by [this link](https://developer.download.nvidia.com/presentations/2008/GDC/GDC08_SoftShadowMapping.pdf), PCSS with 9x9 PCF seems to be a good tradeoff
  - [ ] add shadow bias material property to instances
- [ ] make sure the skybox rad texture resolution is capped at a sane level; it causes the renderer to break on my m1.