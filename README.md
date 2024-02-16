# Ikari

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

Ikari is a game engine project written in pure rust for learning about rust and game tech. It is developed to have decent visuals and good performance, ideally supporting frame rates of 144 or higher.

Most of the rendering features were implemented by following the awesome tutorials on [learnopengl](https://learnopengl.com/) and [learnwgpu](https://sotrh.github.io/learn-wgpu/).

Hopefully one day it will be used in a real game 😃

## Features:

- Linux, Windows & MacOS support
- Web support via WASM/WebGPU
- Rendering
  - Forward rendered with optional depth prepass
  - PBR + IBL
  - Soft shadows via PCF + poisson disk random sample
  - Cascaded shadow mapping
  - Bloom
  - Mesh skinning
  - Dynamic render scale / SSAA
  - GLTF
  - Skybox/environment map blending
  - BCN texture compression
  - Orthographic camera
  - Unlit & transparent materials
- Integrations
  - [Rapier](https://rapier.rs/) physics
  - [Oddio](https://github.com/Ralith/oddio) audio
  - [Iced](https://github.com/iced-rs/iced) UI
  - [Tracy profiler](https://github.com/wolfpld/tracy) CPU profiling + dumps
  - [wgpu-profiler](https://github.com/Wumpf/wgpu-profiler) GPU profiling

## Try it out

```sh
# native
cargo run --features="tracy" --bin example_game
# web
rustup target add wasm32-unknown-unknown
cargo build_web --release --bin example_game
```

See console logs for list of available controls

## clikari

The ikari CLI has the following capabilities:
 
- compress jpg/png textures into GPU-compressed BCN format with baked mips
- pre-process skybox + HDR env maps to be loaded much more efficiently into the game at runtime (500ms vs 10ms)

For example:
```sh
cargo run --bin clikari -- --command process_skybox --background_path ikari/src/textures/milkyway/background.jpg --environment_hdr_path ikari/src/textures/milkyway/radiance.hdr --out_folder ikari/src/skyboxes/milkyway
```

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

### Tracy macos install:

- Install deps:

```sh
brew install freetype2 glfw
# version 4.0.2, haven't tested with v5, maybe it works.
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/442f9cc511ce6dfe75b96b2c83749d90dde914d2/Formula/c/capstone.rb
brew install ./capstone.rb
brew pin capstone
rm capstone.rb
```

- Install tracy
```sh
cd tracy/profiler/build/unix
git checkout v0.9
# you might need to mess around with clang for this to work, not sure if the system default clang is able to perform the compilation
make -I/opt/homebrew/Cellar/capstone/4.0.2/include/capstone -j12
```

- Run tracy

```sh
# the TRACY_DPI_SCALE env var might be needed for the high dpi monitor of your macbook
# you can export it in ~/.zprofile (add `export TRACY_DPI_SCALE=1.0`) to make it work when running from spotlight (cmd+space search thingy)
TRACY_DPI_SCALE=1.0 ./Tracy-release
```

## Running clippy for wasm target

```sh
# this will run clippy on the example game as well as ikari by dependency
RUSTFLAGS=--cfg=web_sys_unstable_apis cargo clippy --package example_game --target wasm32-unknown-unknown
```
