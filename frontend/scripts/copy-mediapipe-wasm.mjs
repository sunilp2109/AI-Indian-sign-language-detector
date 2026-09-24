import { cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const destination = join(root, "public", "mediapipe", "wasm");

mkdirSync(destination, { recursive: true });
cpSync(source, destination, { recursive: true });
console.log(`Copied MediaPipe WASM to ${destination}`);
