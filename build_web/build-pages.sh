#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
site_dir="$repo_root/target/pages"

cd "$repo_root"
cargo build_web --release --bin example_game --build-only --coi-serviceworker

rm -rf "$repo_root/target/pages"
mkdir -p "$site_dir/target/wasm-examples"

cp "$repo_root/ikari-web.html" "$site_dir/index.html"
cp "$repo_root/build_web/coi-serviceworker.js" "$site_dir/coi-serviceworker.js"
cp -R "$repo_root/target/wasm-examples/example_game" "$site_dir/target/wasm-examples/example_game"
touch "$site_dir/.nojekyll"

while IFS= read -r asset_path; do
    destination="$site_dir/$asset_path"
    mkdir -p "$(dirname "$destination")"
    cp "$repo_root/$asset_path" "$destination"
done <<'ASSETS'
ikari/src/models/gltf/ColtPython/colt_python.glb
ikari/src/models/gltf/free_low_poly_forest/scene.gltf
ikari/src/models/gltf/free_low_poly_forest/scene.bin
ikari/src/models/gltf/free_low_poly_forest/textures/PP_Standard_Material_baseColor.png
ikari/src/models/gltf/free_low_poly_forest/textures/PP_Standard_Material_metallicRoughness.png
ikari/src/models/gltf/LegendaryRobot/Legendary_Robot.gltf
ikari/src/models/gltf/LegendaryRobot/Legendary_Robot.bin
ikari/src/models/gltf/LegendaryRobot/robot_emission.png
ikari/src/models/gltf/LegendaryRobot/robot_bump.png
ikari/src/models/gltf/LegendaryRobot/robot_albedo.png
ikari/src/models/gltf/LegendaryRobot/Metalness-robot_roughness.png
ikari/src/models/gltf/DamagedHelmet/DamagedHelmet.gltf
ikari/src/models/gltf/DamagedHelmet/DamagedHelmet.bin
ikari/src/models/gltf/DamagedHelmet/Default_albedo.jpg
ikari/src/models/gltf/DamagedHelmet/Default_metalRoughness.jpg
ikari/src/models/gltf/DamagedHelmet/Default_emissive.jpg
ikari/src/models/gltf/DamagedHelmet/Default_AO.jpg
ikari/src/models/gltf/DamagedHelmet/Default_normal.jpg
ikari/src/skyboxes/milkyway/background/pos_x.jpg
ikari/src/skyboxes/milkyway/background/neg_x.jpg
ikari/src/skyboxes/milkyway/background/pos_y.jpg
ikari/src/skyboxes/milkyway/background/neg_y.jpg
ikari/src/skyboxes/milkyway/background/pos_z.jpg
ikari/src/skyboxes/milkyway/background/neg_z.jpg
ikari/src/skyboxes/milkyway/diffuse_environment_map_compressed.bin
ikari/src/skyboxes/milkyway/specular_environment_map_compressed.bin
ikari/src/sounds/gunshot.wav
ikari/src/sounds/bgm.mp3
ikari/src/textures/rainbow_gradient_vertical.jpg
ikari/src/textures/brick_normal_map.jpg
ASSETS

echo "GitHub Pages artifact assembled at $site_dir"
du -sh "$site_dir"
