/**
 * Renders the social preview (Open Graph) image: an animated GIF and a still JPEG,
 * 1200 × 630. Run `bun run og` and open the printed URL in a browser; the page
 * renders every frame, encodes the GIF and posts both files back to the dev
 * server, which writes them into public/. The figure is an original creature
 * built from primitives (no third-party or trademarked model).
 */
import { GIFEncoder, applyPalette, quantize } from "gifenc";
import * as THREE from "three";

const W = 1200;
const H = 630;
const FRAMES = 40;
const DELAY = 70;           // ms per frame: a 2.8 s loop

// ------------------------------------------------------------------ 3D scene

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, W / H, 1, 1000);
camera.position.set(0, 150, 275);
camera.lookAt(0, 8, 0);
// Shift the picture left so the ball sits in the left half and the text has the right.
camera.setViewOffset(W, H, 290, -28, W, H);

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x1a1f2a, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 2.4);
key.position.set(60, 140, 120);
scene.add(key);
const rim = new THREE.DirectionalLight(0xff8a8e, 1.4);
rim.position.set(-160, 40, -120);
scene.add(rim);

const mat = (color: number, rough = 0.45, metal = 0.0): THREE.MeshStandardMaterial =>
  new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: metal, side: THREE.DoubleSide });

const R = 38;              // outer radius of the ball
const r = 35.5;            // inner radius
const rig = new THREE.Group();
rig.position.set(0, 0, 0);
scene.add(rig);

/** A hemispherical shell (outer + inner surface + rim), opening facing +y when `up` is false. */
function shell(color: number, up: boolean): THREE.Group {
  const g = new THREE.Group();
  const theta = up ? [0, Math.PI / 2] : [Math.PI / 2, Math.PI / 2];
  const outer = new THREE.Mesh(new THREE.SphereGeometry(R, 96, 48, 0, Math.PI * 2, theta[0], theta[1]), mat(color, 0.32));
  const inner = new THREE.Mesh(new THREE.SphereGeometry(r, 96, 48, 0, Math.PI * 2, theta[0], theta[1]), mat(0x5a6475, 0.75));
  const lip = new THREE.Mesh(new THREE.RingGeometry(r, R, 96), mat(0x12151c, 0.6));
  lip.rotation.x = -Math.PI / 2;
  g.add(outer, inner, lip);
  return g;
}

const bottom = shell(0xf2f4f8, false);
rig.add(bottom);
// The black band and the button, on the bottom half's rim.
const band = new THREE.Mesh(new THREE.CylinderGeometry(R + 0.4, R + 0.4, 3.2, 96, 1, true), mat(0x12151c, 0.5));
band.position.y = -1.6;
rig.add(band);
const button = new THREE.Mesh(new THREE.CylinderGeometry(7, 7, 3, 48), mat(0xf2f4f8, 0.3));
button.rotation.x = Math.PI / 2;
button.position.set(0, -1.6, R + 0.6);
const buttonRing = new THREE.Mesh(new THREE.TorusGeometry(7.6, 1.4, 16, 48), mat(0x12151c, 0.5));
buttonRing.position.copy(button.position);
rig.add(button, buttonRing);

// The lid hinges at the back of the rim.
const hinge = new THREE.Group();
hinge.position.set(0, 0, -R);
rig.add(hinge);
const lid = shell(0xe5484d, true);
lid.position.set(0, 0, R);
hinge.add(lid);

// Steel washer on the inner floor, and the figure on it.
const floorY = -Math.sqrt(r * r - 9 * 9) + 0.2;
const washer = new THREE.Mesh(new THREE.CylinderGeometry(9, 9, 1.4, 64), mat(0xb9c2cc, 0.25, 0.9));
washer.position.y = floorY + 0.7;
rig.add(washer);

/** An original round critter: body, ears, feet, eyes, cheeks. */
function critter(): THREE.Group {
  const g = new THREE.Group();
  const green = mat(0x4cd38a, 0.5);
  const body = new THREE.Mesh(new THREE.SphereGeometry(13, 64, 32), green);
  body.scale.set(1, 0.92, 0.95);
  body.position.y = 13;
  const head = new THREE.Mesh(new THREE.SphereGeometry(10.5, 64, 32), green);
  head.position.set(0, 29, 2);
  const earGeo = new THREE.ConeGeometry(3.6, 11, 32);
  const earL = new THREE.Mesh(earGeo, green);
  earL.position.set(-6, 39, 0);
  earL.rotation.z = 0.35;
  const earR = earL.clone();
  earR.position.x = 6;
  earR.rotation.z = -0.35;
  const footGeo = new THREE.SphereGeometry(4.6, 32, 16);
  const footL = new THREE.Mesh(footGeo, mat(0x36b673, 0.55));
  footL.scale.set(1, 0.55, 1.3);
  footL.position.set(-6, 2.2, 4);
  const footR = footL.clone();
  footR.position.x = 6;
  const eyeGeo = new THREE.SphereGeometry(1.7, 24, 12);
  const eyeMat = mat(0x12151c, 0.2);
  const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
  eyeL.position.set(-3.8, 30.5, 11.6);
  const eyeR = eyeL.clone();
  eyeR.position.x = 3.8;
  const cheekGeo = new THREE.SphereGeometry(1.9, 24, 12);
  const cheekMat = mat(0xff6b70, 0.6);
  const cheekL = new THREE.Mesh(cheekGeo, cheekMat);
  cheekL.scale.set(1, 0.6, 0.4);
  cheekL.position.set(-7, 27, 9.8);
  const cheekR = cheekL.clone();
  cheekR.position.x = 7;
  const belly = new THREE.Mesh(new THREE.SphereGeometry(8, 48, 24), mat(0xd9f7e6, 0.6));
  belly.scale.set(1, 1, 0.5);
  belly.position.set(0, 12, 8.5);
  g.add(body, head, earL, earR, footL, footR, eyeL, eyeR, cheekL, cheekR, belly);
  return g;
}

const fig = critter();
fig.scale.setScalar(0.92);
fig.position.y = floorY + 1.4;
// Turned inside the ball so it faces the camera in the still (t = 0.25), button off to the side.
fig.rotation.y = -(Math.PI / 2 - 0.5);
rig.add(fig);

// Two magnets in the feet (figure frame, so they turn with it; glowing so they read through).
for (const x of [-6, 6]) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(2.3, 2.3, 2.4, 32),
    new THREE.MeshStandardMaterial({ color: 0xff3355, emissive: 0x8a0f22, roughness: 0.4 }));
  m.position.set(x, 1.3, 4);
  fig.add(m);
}

// Soft contact shadow on the "table".
const shadow = new THREE.Mesh(new THREE.CircleGeometry(36, 64),
  new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.28 }));
shadow.rotation.x = -Math.PI / 2;
shadow.position.set(0, -R - 0.5, 0);
scene.add(shadow);

function pose(t: number): void {
  // t ∈ [0, 1): one full turn; the lid breathes open and shut twice.
  rig.rotation.y = -0.5 + Math.PI * 2 * t;
  hinge.rotation.x = -(1.35 + 0.25 * Math.sin(Math.PI * 4 * t));
  fig.position.y = floorY + 1.4 + 0.8 * Math.abs(Math.sin(Math.PI * 4 * t));
}

// ------------------------------------------------------------------ 2D composition

const out = document.createElement("canvas");
out.width = W;
out.height = H;
const ctx = out.getContext("2d")!;

function background(): void {
  const g = ctx.createLinearGradient(0, 0, W, H);
  g.addColorStop(0, "#0d1117");
  g.addColorStop(1, "#1a2230");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  const glow = ctx.createRadialGradient(300, 340, 20, 300, 340, 340);
  glow.addColorStop(0, "rgba(229,72,77,0.12)");
  glow.addColorStop(1, "rgba(229,72,77,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, W, H);
}

function text(): void {
  const x = 660;
  ctx.fillStyle = "#e6ebf2";
  ctx.font = "700 92px 'Segoe UI', system-ui, sans-serif";
  ctx.fillText("pokefit", x, 222);
  ctx.fillStyle = "#e5484d";
  ctx.beginPath();
  ctx.arc(x + 338, 196, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#c9d2de";
  ctx.font = "600 38px 'Segoe UI', system-ui, sans-serif";
  ctx.fillText("Fit any figure inside", x, 292);
  ctx.fillText("a Pokéball — with magnets", x, 340);
  ctx.fillStyle = "#8b97a8";
  ctx.font = "400 26px 'Segoe UI', system-ui, sans-serif";
  const lines = ["Largest size that fits · magnet pockets", "Printability checks · Bambu-ready 3MF", "Free · runs in your browser"];
  lines.forEach((l, i) => ctx.fillText(l, x, 412 + i * 38));
  ctx.fillStyle = "#4cd38a";
  ctx.font = "600 30px 'Segoe UI', system-ui, sans-serif";
  ctx.fillText("pokefit.party", x, 560);
}

function frame(t: number): ImageData {
  pose(t);
  renderer.render(scene, camera);
  background();
  ctx.drawImage(renderer.domElement, 0, 0);
  text();
  return ctx.getImageData(0, 0, W, H);
}

async function save(name: string, data: Blob | Uint8Array): Promise<void> {
  const res = await fetch(`/save?name=${name}`, { method: "POST", body: data });
  if (!res.ok) throw new Error(`${name}: ${await res.text()}`);
}

async function main(): Promise<void> {
  const status = document.getElementById("status")!;
  document.body.appendChild(out);
  // Still: a quarter turn in, so the button is to the side and the creature faces the viewer.
  frame(0.25);
  const jpg = await new Promise<Blob>((res) => out.toBlob((b) => res(b!), "image/jpeg", 0.9));
  await save("og.jpg", jpg);
  const gif = GIFEncoder();
  for (let i = 0; i < FRAMES; i++) {
    const { data } = frame(i / FRAMES);
    const palette = quantize(data, 256, { format: "rgb565" });
    gif.writeFrame(applyPalette(data, palette, "rgb565"), W, H, { palette, delay: DELAY });
    status.textContent = `frame ${i + 1}/${FRAMES}`;
    await new Promise((r) => setTimeout(r, 0));
  }
  gif.finish();
  const bytes = gif.bytes();
  await save("og.gif", bytes);
  status.textContent = `done: og.jpg, og.gif (${(bytes.length / 1024).toFixed(0)} KB)`;
}

main().catch((e) => {
  document.getElementById("status")!.textContent = `error: ${(e as Error).message}`;
});
