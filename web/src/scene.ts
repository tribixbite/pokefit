/**
 * Three.js view of the ball cavity and the figure.
 *
 * World = ball frame, z up (z = 0 at the ball centre). The figure GLB is the
 * figure at scale 1 in the figure frame; it sits in `figureFrame`, a group at
 * (offset_x, offset_y, floor_z), scaled by the figure scale. Pockets and check
 * highlights are already in final mm, so they go into `markers`, which shares
 * the position but not the scale.
 */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { allowedProfile, innerWallProfile, type RZ } from "./geometry";
import type { Check, Report } from "./types";

const STATUS_COLOUR: Record<string, number> = { fail: 0xff4d5e, warn: 0xffb347, pass: 0x4cd38a, info: 0x7aa2ff };

/** Lathe around the z axis (three's LatheGeometry spins around y). */
function lathe(profile: RZ[], segments = 96): THREE.LatheGeometry {
  const pts = profile.map((p) => new THREE.Vector2(Math.max(p.r, 1e-4), p.z));
  const g = new THREE.LatheGeometry(pts, segments);
  g.rotateX(Math.PI / 2);
  return g;
}

export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private ballGroup = new THREE.Group();
  private figureFrame = new THREE.Group();
  private figureScaled = new THREE.Group();
  private markers = new THREE.Group();
  private figure: THREE.Object3D | null = null;
  // Keeps y >= 0: removes the half facing the default (front, -y) camera.
  private clip = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private clipped = false;
  private showAllowed = true;
  private loader = new GLTFLoader();
  private figureUrl = "";
  private frameRequested = false;
  /** Outer radius of the ball (final mm), for framing the camera. */
  private ballRadius = 25;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    this.renderer.setClearColor(0x0d1117);
    host.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(35, 1, 0.5, 5000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(70, -95, 45);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.addEventListener("change", () => this.requestFrame());

    this.scene.add(new THREE.HemisphereLight(0xdde6ff, 0x20242c, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(60, -80, 120);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.6);
    rim.position.set(-80, 60, -40);
    this.scene.add(rim);

    this.figureFrame.add(this.figureScaled, this.markers);
    this.scene.add(this.ballGroup, this.figureFrame);
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
  }

  private resize(): void {
    const w = Math.max(this.host.clientWidth, 1);
    const h = Math.max(this.host.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestFrame();
  }

  requestFrame(): void {
    if (this.frameRequested) return;
    this.frameRequested = true;
    requestAnimationFrame(() => {
      this.frameRequested = false;
      // Damping needs a few more frames after the last input.
      if (this.controls.update()) this.requestFrame();
      this.renderer.render(this.scene, this.camera);
    });
  }

  /** Load (or reuse) the figure GLB. A new figure stays hidden until the next `update` places it. */
  async setFigure(url: string): Promise<void> {
    if (url === this.figureUrl && this.figure) return;
    this.figureFrame.visible = false;
    const gltf = await this.loader.loadAsync(url);
    if (this.figure) {
      this.figureScaled.remove(this.figure);
      disposeTree(this.figure);
    }
    this.figure = gltf.scene;
    this.figure.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        const mat = m.material as THREE.MeshStandardMaterial;
        mat.side = THREE.DoubleSide;
        mat.flatShading = false;
        m.geometry.computeVertexNormals();
      }
    });
    this.figureScaled.add(this.figure);
    this.figureUrl = url;
    this.requestFrame();
  }

  /** Rebuild the ball and markers from a report; place and scale the figure. */
  update(rep: Report, focus: Check | null = null): void {
    this.buildBall(rep);
    this.ballRadius = rep.ball.r_outer;
    this.figureFrame.visible = true;
    const s = rep.fit.scale || rep.fit.max_scale;
    this.figureFrame.position.set(rep.fit.offset[0], rep.fit.offset[1], rep.ball.floor_z);
    this.figureScaled.scale.setScalar(s);
    this.buildMarkers(rep, focus);
    this.requestFrame();
  }

  /** Scale preview while the slider moves (before the server answers). */
  previewScale(s: number): void {
    this.figureScaled.scale.setScalar(s);
    this.markers.visible = false;
    this.requestFrame();
  }

  setClip(on: boolean): void {
    this.clipped = on;
    this.ballGroup.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.Material | undefined;
      if (m) {
        m.clippingPlanes = on ? [this.clip] : [];
        m.needsUpdate = true;
      }
    });
    this.requestFrame();
  }

  setAllowedVisible(on: boolean): void {
    this.showAllowed = on;
    const a = this.ballGroup.getObjectByName("allowed");
    if (a) a.visible = on;
    this.requestFrame();
  }

  view(kind: "front" | "side" | "top" | "bottom" | "iso"): void {
    // Frame the ball to ~75 % of the shorter screen side: visible height at
    // distance d is 2·d·tan(fov/2); narrow (portrait) views are limited by width.
    const visible = 2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const d = (2 * this.ballRadius) / (0.75 * visible * Math.min(1, this.camera.aspect));
    const pos: Record<typeof kind, [number, number, number]> = {
      front: [0, -d, 0],
      side: [d, 0, 0],
      top: [0, -0.01, d],
      bottom: [0, -0.01, -d],
      iso: [d * 0.55, -d * 0.75, d * 0.35],
    };
    this.camera.position.set(...pos[kind]);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.requestFrame();
  }

  snapshot(): string {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }

  private buildBall(rep: Report): void {
    disposeTree(this.ballGroup);
    this.ballGroup.clear();
    const b = rep.ball;
    const clipping = this.clipped ? [this.clip] : [];

    const wall = new THREE.Mesh(
      lathe(innerWallProfile(b)),
      new THREE.MeshStandardMaterial({
        color: 0xe8ecf4, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false,
        roughness: 0.6, clippingPlanes: clipping,
      }),
    );
    wall.name = "wall";
    wall.renderOrder = 2;

    // The ring band, drawn darker, so it reads as the black connector.
    const ringProfile = innerWallProfile(b).filter((p) => Math.abs(p.z) <= b.ring_span);
    const ring = new THREE.Mesh(
      lathe(ringProfile),
      new THREE.MeshStandardMaterial({
        color: 0x1b1f27, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false,
        clippingPlanes: clipping,
      }),
    );
    ring.renderOrder = 3;

    const allowed = new THREE.Mesh(
      lathe(allowedProfile(b)),
      new THREE.MeshBasicMaterial({
        color: 0x37c2ff, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false,
        clippingPlanes: clipping,
      }),
    );
    allowed.name = "allowed";
    allowed.visible = this.showAllowed;
    allowed.renderOrder = 1;
    const allowedEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(lathe(allowedProfile(b), 24), 30),
      new THREE.LineBasicMaterial({ color: 0x37c2ff, transparent: true, opacity: 0.25, clippingPlanes: clipping }),
    );
    allowed.add(allowedEdges);

    // Washer resting on its seat; the seat itself as a dark disc.
    const w = b.washer;
    const washer = new THREE.Mesh(
      lathe([
        { r: w.id / 2, z: b.seat_z },
        { r: w.od / 2, z: b.seat_z },
        { r: w.od / 2, z: b.floor_z },
        { r: w.id / 2, z: b.floor_z },
        { r: w.id / 2, z: b.seat_z },
      ], 96),
      new THREE.MeshStandardMaterial({
        color: 0x9aa3ad, metalness: 0.85, roughness: 0.35, transparent: true, opacity: 0.6, depthWrite: false,
        clippingPlanes: clipping,
      }),
    );
    const seat = new THREE.Mesh(
      new THREE.CircleGeometry(w.od / 2 + 0.2, 64),
      new THREE.MeshStandardMaterial({
        color: 0xc8303a, side: THREE.DoubleSide, transparent: true, opacity: 0.35, depthWrite: false,
        clippingPlanes: clipping,
      }),
    );
    seat.position.z = b.seat_z - 0.01;

    // Equator line and the outer shell silhouette for scale.
    const eq = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(
        Array.from({ length: 129 }, (_, i) => {
          const a = (i / 128) * Math.PI * 2;
          return new THREE.Vector3(Math.cos(a) * b.r_outer, Math.sin(a) * b.r_outer, 0);
        }),
      ),
      new THREE.LineBasicMaterial({ color: 0x5a6578 }),
    );
    this.ballGroup.add(allowed, wall, ring, washer, seat, eq);
  }

  private buildMarkers(rep: Report, focus: Check | null): void {
    disposeTree(this.markers);
    this.markers.clear();
    this.markers.visible = true;
    const d = rep.params.pocket_diameter;
    const depth = rep.params.pocket_depth;
    for (const p of rep.pockets) {
      const cyl = new THREE.Mesh(
        new THREE.CylinderGeometry(d / 2, d / 2, depth, 32),
        // Drawn over everything: pockets sit inside the base and would be hidden.
        new THREE.MeshStandardMaterial({ color: 0xff3355, emissive: 0x550011, depthTest: false, transparent: true, opacity: 0.9 }),
      );
      cyl.renderOrder = 9;
      cyl.rotation.x = Math.PI / 2;
      cyl.position.set(p.x, p.y, (p.z ?? 0) + depth / 2 - 0.05);   // z: the pocket's opening above the bed
      this.markers.add(cyl);
    }
    const lim = rep.fit.limit_point;
    if (lim) this.markers.add(dot(lim, 0xffe14d, 0.9));
    // Without a selected check only failures are marked; warnings (thin tips,
    // small overhangs) would cover the figure in dots. Click a check to see its points.
    const checks = focus ? [focus] : rep.checks.filter((c) => c.status === "fail");
    for (const c of checks) {
      const colour = STATUS_COLOUR[c.status] ?? 0xffffff;
      const pts = c.highlight.filter((p) => p.length === 3);
      if (pts.length === 0) continue;
      const geom = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
      this.markers.add(
        new THREE.Points(geom, new THREE.PointsMaterial({ color: colour, size: focus ? 7 : 4, sizeAttenuation: false, depthTest: false })),
      );
    }
  }
}

function dot(p: number[], colour: number, r: number): THREE.Mesh {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(r, 16, 12),
    new THREE.MeshBasicMaterial({ color: colour, depthTest: false }),
  );
  m.position.set(p[0], p[1], p[2]);
  m.renderOrder = 10;
  return m;
}

function disposeTree(root: THREE.Object3D): void {
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
    else mat?.dispose();
  });
}
