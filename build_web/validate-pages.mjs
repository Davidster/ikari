import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const componentByteLengths = new Map([
    [5120, 1],
    [5121, 1],
    [5122, 2],
    [5123, 2],
    [5125, 4],
    [5126, 4],
]);
const typeComponentCounts = new Map([
    ["SCALAR", 1],
    ["VEC2", 2],
    ["VEC3", 3],
    ["VEC4", 4],
    ["MAT2", 4],
    ["MAT3", 9],
    ["MAT4", 16],
]);

function findFiles(directory, extensions) {
    return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const entryPath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            return findFiles(entryPath, extensions);
        }
        return extensions.some((extension) => entryPath.endsWith(extension))
            ? [entryPath]
            : [];
    });
}

function readGltfDocument(gltfPath) {
    if (gltfPath.endsWith(".gltf")) {
        return JSON.parse(fs.readFileSync(gltfPath, "utf8"));
    }

    const glb = fs.readFileSync(gltfPath);
    const jsonChunkLength = glb.readUInt32LE(12);
    const jsonChunkType = glb.readUInt32LE(16);
    if (jsonChunkType !== 0x4e4f534a) {
        throw new Error(`${gltfPath} does not start with a glTF JSON chunk`);
    }
    return JSON.parse(
        glb.subarray(20, 20 + jsonChunkLength).toString("utf8").trimEnd(),
    );
}

function validateGltf(gltfPath) {
    const document = readGltfDocument(gltfPath);
    const errors = [];

    for (const collection of [document.buffers ?? [], document.images ?? []]) {
        for (const item of collection) {
            if (!item.uri || item.uri.startsWith("data:")) {
                continue;
            }

            const dependencyPath = path.resolve(
                path.dirname(gltfPath),
                decodeURIComponent(item.uri),
            );
            if (!fs.existsSync(dependencyPath)) {
                errors.push(`missing dependency "${item.uri}"`);
            }
        }
    }

    for (const [index, accessor] of (document.accessors ?? []).entries()) {
        const bufferView = document.bufferViews?.[accessor.bufferView];
        const stride = bufferView?.byteStride;
        if (stride == null) {
            continue;
        }

        const componentByteLength = componentByteLengths.get(accessor.componentType);
        const componentCount = typeComponentCounts.get(accessor.type);
        const elementByteLength = componentByteLength * componentCount;
        if (stride !== elementByteLength) {
            errors.push(
                `accessor ${index} is interleaved (stride ${stride}, element ${elementByteLength}); ikari's loader requires tightly packed accessors`,
            );
        }
    }

    return errors;
}

const siteDirectory = path.resolve(process.argv[2] ?? "target/pages");
const failures = findFiles(siteDirectory, [".gltf", ".glb"]).flatMap((gltfPath) =>
    validateGltf(gltfPath).map(
        (error) => `${path.relative(siteDirectory, gltfPath)}: ${error}`,
    ),
);

const indexHtml = fs.readFileSync(path.join(siteDirectory, "index.html"), "utf8");
if (indexHtml.includes("coi-serviceworker.js")) {
    if (!indexHtml.includes("if (!window.crossOriginIsolated)")) {
        failures.push("index.html: threaded WASM start is not gated on cross-origin isolation");
    }
    if (!indexHtml.includes("startButton.disabled = true")) {
        failures.push("index.html: start button remains enabled before service-worker activation");
    }
}

const serviceWorkerPath = path.join(siteDirectory, "coi-serviceworker.js");
if (fs.existsSync(serviceWorkerPath)) {
    const serviceWorker = fs.readFileSync(serviceWorkerPath, "utf8");
    if (/caches\.(open|match|put|delete|keys)/.test(serviceWorker)) {
        failures.push("coi-serviceworker.js: CacheStorage use is not allowed");
    }
    if (/Cache-Control|no-store/.test(serviceWorker)) {
        failures.push("coi-serviceworker.js: cache-control overrides are not allowed");
    }
}

if (failures.length > 0) {
    console.error(failures.join("\n"));
    process.exit(1);
}

console.log("Pages assets, isolation gate, and service-worker cache policy are valid.");
