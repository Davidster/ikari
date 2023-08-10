# Ikari

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

Ikari is a game engine project written in pure rust for learning about rust and game tech. It is developed to have decent visuals and good performance, ideally supporting frame rates of 144 or higher.

Most of the rendering features were implemented by following the awesome tutorials on [learnopengl](https://learnopengl.com/) and [learnwgpu](https://sotrh.github.io/learn-wgpu/).

Hopefully one day it will be used in a real game 😃

## Features:

- Physically-Based Rendering materials (PBR)
- Image-Based Lighting
- Point and directional lights
- Soft shadow edges with randomized Percentage Closer Filtering
- Adjustable Bloom
- Adjustable camera exposure
- Skeletal animations
- Auto-generated box collision meshes, including support for skinned characters
- [Rapier](https://rapier.rs/) integration for physics
- [Oddio](https://github.com/Ralith/oddio) integration for audio
- [Iced](https://github.com/iced-rs/iced) integration for UI
- [wgpu-profiler](https://github.com/Wumpf/wgpu-profiler)
- Dynamic render scale
  - Can be used to do supersampling anti-aliasing
- glTF asset import
  - Supports scene graph, meshes, materials, animations, and skinning
- Linux, Windows & MacOS support
- Wireframe mode
- Unlit materials
- Equirectangular and cubemap skybox support
- Scene graph
- CPU-side frustum culling
- Mipmapping
- Reverse-Z for better depth accuracy / less z-fighting

## TODO

- stop using release build, switch to dev but with optimizations enabled, but leave debug=true for release builds.
- fix resolve_path function in file_loader so it can be used outside the context of the engine!
- path in error messages in file loader module (see TODO: there)
- renderer should not need to know about the game state!
- lift state from the ui overlay into the game state to not need to pass state around
- convert all paths to be relative to the bin
- pre-process skyboxes
- support time-sliced skybox binder?
- allow to blend between two skyboxes to allow transition
- find all occurences of dynamic image and try to remove them, should be faster?
- take as much stuff out of baserenderer as possible

## Try it out

```sh
# native
cargo run --release --features="tracy" --bin example_game
# web
cargo build_web --release --bin example_game
```

See console logs for list of available controls

## Screenshots / Videos:

### PBR glTF assets

![screenshot_1](https://user-images.githubusercontent.com/2389735/174690197-1761b4ca-3c93-43c2-ba0f-a17470802613.jpg)

### Point-lights + shadows

![screenshot_2](https://user-images.githubusercontent.com/2389735/174689921-9aad3283-171a-48ee-9d3a-c544aed2314e.jpg)

### Mesh skinning + animations

https://user-images.githubusercontent.com/2389735/176325053-18c47d31-71b3-4aa4-a1d9-ec3a6356d2c7.mp4

### Rapier physics

https://user-images.githubusercontent.com/2389735/178186964-c42f44c7-8e3e-475c-8104-48a98be7709f.mp4

### Auto-generated character collision meshes

https://user-images.githubusercontent.com/2389735/180101651-86ba2084-4196-494b-9a36-3b6847161af1.mp4

### Iced UI + wgpu-profiler

![screenshot_3](https://user-images.githubusercontent.com/2389735/229004532-8c2b21c5-1473-4243-b1f0-821c7abc5fca.png)

## Profiling
You can profile ikari with [tracy](https://github.com/wolfpld/tracy) via [profiling](https://github.com/aclysma/profiling) crate.
To do that you have to:
- Download [tracy 0.9](https://github.com/wolfpld/tracy/releases/tag/v0.9)
- Build ikari by adding --features="tracy" to the cargo command
- Run ikari and tracy (in any order), when the game is loading tracy will start profiling

If something does not work it is possible that the crate profiling has been updated and is no longer aligned with the tracy version and a more recent one must be used.

### Tracy linux install:

Based on instructions from here: https://github.com/wolfpld/tracy/issues/484

- Install glfw, freetype2
  - (e.g. aur packages `glfw-x11`, `freetype2`). maybe different on Ubuntu
- Install libcapstone
  - Clone https://github.com/libcapstone/libcapstone
  - cd libcapstone
  - PATH=/usr/local/bin:/usr/bin:/bin CC=clang CXX=clang++ make -j12
  - sudo make install
- Install tracy 0.9
  - Clone https://github.com/wolfpld/tracy
  - cd tracy/profiler/build/unix
  - git checkout v0.9
  - PATH=/usr/local/bin:/usr/bin:/bin CC=clang CXX=clang++ make -j12
- Run tracy
  - ./Tracy_release
- I made a desktop entry to make it easier to start from the system menus:
  - Not sure if it works in other distros, tested on Arch/KDE
  - Create file ~/.local/share/applications/tracy.desktop with contents:
    ```
    [Desktop Entry]
    Name=Tracy
    Exec=/home/david/Programming/tracy/profiler/build/unix/Tracy-release
    Type=Application
    ```

## Running clippy for wasm target

```sh
# this will run clippy on the example game as well as ikari by dependency
RUSTFLAGS=--cfg=web_sys_unstable_apis cargo clippy --package example_game --target wasm32-unknown-unknown
```
