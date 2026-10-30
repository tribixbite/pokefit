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
import { base64Bytes, groupFaces } from "./faces";
import { allowedProfile, innerWallProfile, type RZ } from "./geometry";
import type { Check, PreviewFaces, Report } from "./types";

/** A tap on the figure: surface point and outward normal in the figure frame at scale 1. */
export interface FigureHit {
  point: number[];
  normal: number[];
}

/** A brush stroke being drawn: its samples and the line that shows it. */
interface Stroke {
  id: number;
  hits: FigureHit[];
  line: THREE.Line;
  last: { x: number; y: number };
  lastAt: number;               // performance.now() of the last raycast
}

const STROKE_PX = 4;          // screen distance between stroke samples (the engine joins the gaps)
const STROKE_MS = 12;         // and time between them
const MAX_STROKE = 2000;      // samples per stroke (the engine takes up to 4000)

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
  /** Engine signature of the shown figure's geometry ("" = none): equal = recolour in place. */
  figureSignature = "";
  /** Each figure mesh's face order as loaded (colour edits regroup the index buffer from it). */
  private faceOrder = new Map<THREE.BufferGeometry, THREE.TypedArray>();
  private faceMaterials: THREE.MeshStandardMaterial[] = [];
  private frameRequested = false;
  /** Pixels covered at the bottom by a panel (`setBottomInset`). */
  private bottomInset = 0;
  /** Outer radius of the ball (final mm), for framing the camera. */
  private ballRadius = 25;
  /** Colorize: called with a tap on the figure (figure frame, scale 1); null = taps orbit only. */
  private pickHandler: ((hit: FigureHit) => void) | null = null;
  /** Colorize brush: called with a stroke's samples when a drag on the figure ends; null = drags orbit. */
  private strokeHandler: ((hits: FigureHit[]) => void) | null = null;
  private stroke: Stroke | null = null;
  /** Colorize stamp: called with the figure point under a dragged box's centre and the ellipse's semi-axes. */
  private stampHandler: ((hit: FigureHit, axes: number[]) => void) | null = null;
  private stampDrag: { id: number; x: number; y: number; box: HTMLDivElement } | null = null;
  private raycaster = new THREE.Raycaster();

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

    // A tap (press and release within a few pixels) picks; a drag still orbits.
    let down: { x: number; y: number; id: number } | null = null;
    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, id: e.pointerId }; });
    canvas.addEventListener("pointerup", (e) => {
      const d = down;
      down = null;
      if (!d || d.id !== e.pointerId || !this.pickHandler) return;
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 6) return;
      const hit = this.hitFigure(e.clientX, e.clientY);
      if (hit) this.pickHandler(hit);
    });

    // Brush strokes: a press ON the figure paints along the drag instead of orbiting. Captured on the host, before
    // OrbitControls (on the canvas) sees it; presses beside the figure, a second finger or the right button still
    // turn the view.
    host.addEventListener("pointerdown", (e) => {
      if (this.stampHandler && e.target === canvas && e.isPrimary && e.button === 0 && this.hitFigure(e.clientX, e.clientY)) {
        // Stamp: a drag that starts on the figure draws the ellipse's box instead of turning the view.
        e.stopPropagation();
        try { canvas.setPointerCapture(e.pointerId); } catch { /* the drag still ends on its pointerup */ }
        const box = document.createElement("div");
        box.className = "stamp-box";
        host.appendChild(box);
        this.stampDrag = { id: e.pointerId, x: e.clientX, y: e.clientY, box };
        this.placeStampBox(e.clientX, e.clientY);
        return;
      }
      if (!this.strokeHandler || e.target !== canvas || !e.isPrimary || e.button !== 0) {
        if (this.stroke && !e.isPrimary) this.endStroke(false);      // a pinch started: no stroke
        return;
      }
      const hit = this.hitFigure(e.clientX, e.clientY);
      if (!hit) return;
      e.stopPropagation();
      try {
        canvas.setPointerCapture(e.pointerId);      // moves outside the canvas still reach the stroke
      } catch {
        // a pointer the browser no longer tracks: the stroke ends on its own pointerup anyway
      }
      this.startStroke(e.pointerId, e.clientX, e.clientY, hit);
    }, { capture: true });
    canvas.addEventListener("pointermove", (e) => {
      if (this.stampDrag && e.pointerId === this.stampDrag.id) this.placeStampBox(e.clientX, e.clientY);
      const s = this.stroke;
      if (!s || e.pointerId !== s.id) return;
      // At most one raycast per STROKE_MS (a 300k-triangle figure takes several ms each); by time, not by animation
      // frame, which a background tab hardly fires. The engine joins the gaps between samples.
      if (performance.now() - s.lastAt < STROKE_MS) return;
      this.sampleStroke(s, e.clientX, e.clientY);
    });
    const finish = (e: PointerEvent) => {
      if (this.stampDrag && e.pointerId === this.stampDrag.id) this.endStamp(e);
      const s = this.stroke;
      if (!s || e.pointerId !== s.id) return;
      if (e.type === "pointerup") this.sampleStroke(s, e.clientX, e.clientY);     // where the drag ended
      this.endStroke(e.type === "pointerup");
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);
  }

  /** Brush mode: a drag that starts on the figure is a stroke, reported on release (null = drags orbit). */
  setStrokeHandler(fn: ((hits: FigureHit[]) => void) | null): void {
    this.strokeHandler = fn;
    if (!fn && this.stroke) this.endStroke(false);
    // While strokes take one-finger / left drags on the figure, the right button turns the view too.
    this.controls.mouseButtons.RIGHT = fn ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
  }

  /** Stamp mode: a drag on the figure draws a box; on release the ellipse in it is stamped (null = drags orbit). */
  setStampHandler(fn: ((hit: FigureHit, axes: number[]) => void) | null): void {
    this.stampHandler = fn;
    if (!fn && this.stampDrag) { this.stampDrag.box.remove(); this.stampDrag = null; }
    this.controls.mouseButtons.RIGHT = fn || this.strokeHandler ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
  }

  /** The stamp box from the press to (x, y), in the host's coordinates. */
  private placeStampBox(x: number, y: number): void {
    const d = this.stampDrag;
    if (!d) return;
    const r = this.host.getBoundingClientRect();
    Object.assign(d.box.style, {
      left: `${Math.min(d.x, x) - r.left}px`, top: `${Math.min(d.y, y) - r.top}px`,
      width: `${Math.abs(x - d.x)}px`, height: `${Math.abs(y - d.y)}px`,
    });
  }

  /** Release: the ellipse inscribed in the box, centred where its centre meets the figure, as semi-axes along the
   * screen's right and up at that depth (figure frame, scale 1). */
  private endStamp(e: PointerEvent): void {
    const d = this.stampDrag;
    this.stampDrag = null;
    if (!d) return;
    d.box.remove();
    const w = Math.abs(e.clientX - d.x), h = Math.abs(e.clientY - d.y);
    if (e.type !== "pointerup" || w < 6 || h < 6 || !this.stampHandler || !this.figure) return;
    const hit = this.hitFigure((e.clientX + d.x) / 2, (e.clientY + d.y) / 2);
    if (!hit) return;
    const world = new THREE.Vector3(...hit.point).applyMatrix4(this.figure.matrixWorld);
    const cam = this.camera;
    const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0).normalize();
    const up = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1).normalize();
    const forward = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 2).negate().normalize();
    const depth = world.clone().sub(cam.position).dot(forward);
    const perPx = (2 * depth * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2)) / this.renderer.domElement.clientHeight;
    const inv = new THREE.Matrix4().copy(this.figure.matrixWorld).invert();
    const p = new THREE.Vector3(...hit.point);
    const axis = (dir: THREE.Vector3, px: number) =>
      world.clone().addScaledVector(dir, (px / 2) * perPx).applyMatrix4(inv).sub(p);
    const u = axis(right, w), v = axis(up, h);
    this.stampHandler(hit, [u.x, u.y, u.z, v.x, v.y, v.z]);
  }

  /** Add the figure point under (x, y) to the stroke, if it moved far enough on screen and hits the figure. */
  private sampleStroke(s: Stroke, x: number, y: number): void {
    if (Math.hypot(x - s.last.x, y - s.last.y) < STROKE_PX) return;
    s.lastAt = performance.now();
    const hit = this.hitFigure(x, y);
    if (!hit) return;
    s.last = { x, y };
    this.addStrokePoint(s, hit);
  }

  private startStroke(id: number, x: number, y: number, hit: FigureHit): void {
    if (!this.figure) return;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAX_STROKE * 3), 3));
    geom.setDrawRange(0, 0);
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
    line.renderOrder = 10;
    this.figure.add(line);
    this.stroke = { id, hits: [], line, last: { x, y }, lastAt: performance.now() };
    this.addStrokePoint(this.stroke, hit);
  }

  private addStrokePoint(s: Stroke, hit: FigureHit): void {
    if (s.hits.length >= MAX_STROKE) return;
    s.hits.push(hit);
    const pos = s.line.geometry.getAttribute("position") as THREE.BufferAttribute;
    pos.setXYZ(s.hits.length - 1, hit.point[0], hit.point[1], hit.point[2]);
    pos.needsUpdate = true;
    s.line.geometry.setDrawRange(0, s.hits.length);
    this.requestFrame();
  }

  /** Stop the current stroke; `report` hands its samples to the stroke handler. */
  private endStroke(report: boolean): void {
    const s = this.stroke;
    if (!s) return;
    this.stroke = null;
    s.line.removeFromParent();
    s.line.geometry.dispose();
    (s.line.material as THREE.Material).dispose();
    this.requestFrame();
    if (report && this.strokeHandler) this.strokeHandler(s.hits);
  }

  /** Turn tap-to-pick on (with a handler) or off. */
  setPickHandler(fn: ((hit: FigureHit) => void) | null): void {
    this.pickHandler = fn;
    this.renderer.domElement.classList.toggle("picking", fn !== null);
  }

  /** The figure surface under a screen point, in the figure GLB's own frame (figure frame, scale 1). */
  private hitFigure(clientX: number, clientY: number): FigureHit | null {
    if (!this.figure || !this.figureFrame.visible) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObject(this.figure, true);
    // The cut view hides the front half of the ball only; the figure is whole, so the first hit is the one seen.
    const h0 = hits.find((h) => (h.object as THREE.Mesh).isMesh && h.face);
    if (!h0 || !h0.face) return null;
    const inv = new THREE.Matrix4().copy(this.figure.matrixWorld).invert();
    const point = h0.point.clone().applyMatrix4(inv);
    // Face normal: mesh local → world → figure frame; facing the camera (the GLB is drawn double-sided).
    const n = h0.face.normal.clone().transformDirection(h0.object.matrixWorld);
    if (n.dot(this.raycaster.ray.direction) > 0) n.negate();
    n.transformDirection(inv);
    return { point: [point.x, point.y, point.z], normal: [n.x, n.y, n.z] };
  }

  private resize(): void {
    const w = Math.max(this.host.clientWidth, 1);
    const h = Math.max(this.host.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    // With a panel over the bottom of the view (the paint dock), the camera frames the part above it: the
    // projection of a view `free` pixels tall, extended down over the panel (picking unprojects through it too).
    const free = Math.max(h - this.bottomInset, Math.min(h, 120));
    this.camera.aspect = w / free;
    if (free < h) this.camera.setViewOffset(w, free, 0, 0, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.requestFrame();
  }

  /** Pixels at the bottom of the view covered by a panel (0: none): the scene is framed above them. */
  setBottomInset(px: number): void {
    const v = Math.max(0, Math.round(px));
    if (v === this.bottomInset) return;
    this.bottomInset = v;
    this.resize();
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

  /**
   * Load (or reuse) the figure GLB: one uncoloured mesh per part (`p<i>`), coloured by `setFaceColours`.
   * A new figure stays hidden until the next `update` places it.
   */
  async setFigure(url: string, signature: string): Promise<void> {
    if (url === this.figureUrl && this.figure) return;
    this.figureFrame.visible = false;
    const gltf = await this.loader.loadAsync(url);
    if (this.figure) {
      this.figureScaled.remove(this.figure);
      disposeTree(this.figure);
    }
    this.faceOrder.clear();
    this.figure = gltf.scene;
    this.figure.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        const mat = m.material as THREE.MeshStandardMaterial;
        mat.side = THREE.DoubleSide;
        mat.flatShading = false;
        m.geometry.computeVertexNormals();
        if (m.geometry.index) this.faceOrder.set(m.geometry, m.geometry.index.array.slice());
      }
    });
    this.figureScaled.add(this.figure);
    this.figureUrl = url;
    this.figureSignature = signature;
    this.requestFrame();
  }

  /**
   * Colour the figure's faces in place (a paint or palette edit): each mesh's triangles are
   * regrouped by colour, one draw call per colour, without reloading its geometry.
   */
  setFaceColours(faces: PreviewFaces): void {
    if (!this.figure) return;
    for (const m of this.faceMaterials) m.dispose();
    this.faceMaterials = faces.colours.map((c) => new THREE.MeshStandardMaterial({
      color: new THREE.Color(c), side: THREE.DoubleSide, metalness: 0, roughness: 0.75,
    }));
    if (faces.shape) this.faceMaterials[0] = shapeMaterial(this.faceMaterials[0]);
    faces.parts.forEach((b64, i) => {
      const mesh = this.figure?.getObjectByName(`p${i}`) as THREE.Mesh | undefined;
      const order = mesh?.isMesh ? this.faceOrder.get(mesh.geometry) : undefined;
      if (!mesh || !order) return;
      const shape = faces.shape?.[i] ? base64Bytes(faces.shape[i]) : null;
      if (shape && shape.length === mesh.geometry.getAttribute("position").count * 2) {
        mesh.geometry.setAttribute("shape", new THREE.BufferAttribute(shape, 2, true));
      } else if (mesh.geometry.getAttribute("shape")) {
        mesh.geometry.deleteAttribute("shape");
      }
      const colour = base64Bytes(b64);
      if (colour.length * 3 !== order.length) return;   // another geometry: the next full load fixes it
      const { index, groups } = groupFaces(order, colour, this.faceMaterials.length);
      const geo = mesh.geometry;
      geo.setIndex(new THREE.BufferAttribute(index, 1));
      geo.clearGroups();
      for (const g of groups) geo.addGroup(g.start, g.count, g.materialIndex);
      const old = mesh.material;
      if (!Array.isArray(old) && !this.faceMaterials.includes(old as THREE.MeshStandardMaterial)) old.dispose();
      mesh.material = this.faceMaterials;
    });
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

/**
 * The shape map's material: `base` shaded per vertex from the engine's `shape` attribute (crease, relief): creases
 * darken, raised spots turn warm and sunk ones cool, and lines of equal relief are drawn like a topographic map's
 * contours (anti-aliased by screen-space derivatives, so they stay thin at any zoom).
 */
function shapeMaterial(base: THREE.MeshStandardMaterial): THREE.MeshStandardMaterial {
  base.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute vec2 shape;\nvarying vec2 vShape;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvShape = shape;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying vec2 vShape;")
      .replace("#include <color_fragment>", `#include <color_fragment>
      {
        float crease = vShape.x;
        float relief = vShape.y * 2.0 - 1.0;
        vec3 c = relief > 0.0 ? mix(diffuseColor.rgb, vec3(0.86, 0.36, 0.26), min(relief, 1.0) * 0.6)
                              : mix(diffuseColor.rgb, vec3(0.25, 0.45, 0.90), min(-relief, 1.0) * 0.6);
        float lv = relief * 6.0;
        float line = 1.0 - smoothstep(0.0, 1.5 * fwidth(lv), abs(fract(lv - 0.5) - 0.5));
        c = mix(c, vec3(0.1), line * 0.5);
        c *= 1.0 - 0.8 * smoothstep(0.15, 0.9, crease);
        diffuseColor.rgb = c;
      }`);
  };
  base.customProgramCacheKey = () => "shape-map";
  base.needsUpdate = true;
  return base;
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
