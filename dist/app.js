import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/+esm";
import * as THREE from "three";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import JSZip from "https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm";

const ui = Object.fromEntries([
  "video", "overlay", "idleVisual", "cameraBadge", "directionHint", "directionIcon", "directionTitle", "directionText",
  "countdown", "startButton", "captureButton", "resetButton", "cameraNote", "progressLabel", "progressRing", "progressPercent",
  "steps", "lightDot", "lightStatus", "faceDot", "faceStatus", "resultCard", "downloadButton", "downloadGlbButton", "captureCanvas"
].map(id => [id, document.getElementById(id)]));

const poses = [
  { title: "Olhe para a câmera", text: "Expressão neutra e rosto centralizado", icon: "●" },
  { title: "Gire levemente à esquerda", text: "Pare perto de 25 graus", icon: "←" },
  { title: "Mostre o perfil esquerdo", text: "Pare perto de 45 graus", icon: "←" },
  { title: "Gire levemente à direita", text: "Pare perto de 25 graus", icon: "→" },
  { title: "Mostre o perfil direito", text: "Pare perto de 45 graus", icon: "→" }
];

const state = {
  stream: null,
  landmarker: null,
  landmarks: null,
  captures: [],
  step: 0,
  detecting: false,
  lightOkay: false,
  glbBlob: null,
  objText: "",
  mtlText: "",
  textureBlob: null
};

const overlayCtx = ui.overlay.getContext("2d");
const captureCtx = ui.captureCanvas.getContext("2d", { willReadFrequently: true });

async function createLandmarker() {
  if (state.landmarker) return state.landmarker;
  ui.cameraNote.textContent = "Preparando o detector facial…";
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm");
  state.landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.6,
    minFacePresenceConfidence: 0.6,
    minTrackingConfidence: 0.55,
    outputFacialTransformationMatrixes: true
  });
  return state.landmarker;
}

async function startScan() {
  try {
    ui.startButton.disabled = true;
    const [landmarker, stream] = await Promise.all([
      createLandmarker(),
      navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
    ]);
    state.landmarker = landmarker;
    state.stream = stream;
    ui.video.srcObject = stream;
    await ui.video.play();
    ui.idleVisual.hidden = true;
    ui.cameraBadge.textContent = "Câmera ativa";
    ui.directionHint.hidden = false;
    ui.captureButton.disabled = false;
    ui.resetButton.disabled = false;
    ui.cameraNote.textContent = "Nada é enviado: as imagens permanecem neste dispositivo.";
    state.detecting = true;
    setPose(0);
    requestAnimationFrame(detectLoop);
  } catch (error) {
    console.error(error);
    ui.startButton.disabled = false;
    ui.cameraNote.textContent = error?.name === "NotAllowedError"
      ? "Permissão da câmera negada. Libere a câmera no navegador e tente novamente."
      : "Não foi possível iniciar a câmera. Tente outro navegador ou dispositivo.";
  }
}

function detectLoop() {
  if (!state.detecting || !state.landmarker || ui.video.readyState < 2) return requestAnimationFrame(detectLoop);
  ui.overlay.width = ui.video.videoWidth;
  ui.overlay.height = ui.video.videoHeight;
  const result = state.landmarker.detectForVideo(ui.video, performance.now());
  overlayCtx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  state.landmarks = result.faceLandmarks?.[0] || null;

  if (state.landmarks) {
    const drawing = new DrawingUtils(overlayCtx);
    drawing.drawConnectors(state.landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#79f2c0", lineWidth: 2 });
    drawing.drawConnectors(state.landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "rgba(201,255,96,.7)", lineWidth: 1 });
    drawing.drawConnectors(state.landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "rgba(201,255,96,.7)", lineWidth: 1 });
    drawing.drawConnectors(state.landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "rgba(121,242,192,.7)", lineWidth: 1 });
    ui.faceDot.className = "good";
    ui.faceStatus.textContent = "Rosto detectado";
    state.lightOkay = estimateLight();
    ui.lightDot.className = state.lightOkay ? "good" : "warn";
    ui.lightStatus.textContent = state.lightOkay ? "Iluminação adequada" : "Procure mais luz";
  } else {
    ui.faceDot.className = "warn";
    ui.faceStatus.textContent = "Centralize o rosto";
    ui.lightDot.className = "";
    ui.lightStatus.textContent = "Aguardando rosto";
  }
  requestAnimationFrame(detectLoop);
}

function estimateLight() {
  const sample = document.createElement("canvas");
  sample.width = 64; sample.height = 48;
  const ctx = sample.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(ui.video, 0, 0, 64, 48);
  const data = ctx.getImageData(0, 0, 64, 48).data;
  let total = 0;
  for (let i = 0; i < data.length; i += 16) total += (data[i] + data[i + 1] + data[i + 2]) / 3;
  const avg = total / (data.length / 16);
  return avg > 55 && avg < 235;
}

function setPose(index) {
  const pose = poses[index];
  if (!pose) return;
  ui.directionTitle.textContent = pose.title;
  ui.directionText.textContent = pose.text;
  ui.directionIcon.textContent = pose.icon;
  [...ui.steps.children].forEach((item, i) => {
    item.classList.toggle("active", i === index);
    item.classList.toggle("done", i < index);
  });
}

async function capturePose() {
  if (!state.landmarks) {
    ui.cameraNote.textContent = "Não encontrei seu rosto. Centralize-se no contorno e tente novamente.";
    return;
  }
  ui.captureButton.disabled = true;
  await runCountdown();
  const size = 1024;
  const vw = ui.video.videoWidth, vh = ui.video.videoHeight;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2, sy = (vh - side) / 2;
  captureCtx.clearRect(0, 0, size, size);
  captureCtx.drawImage(ui.video, sx, sy, side, side, 0, 0, size, size);
  const blob = await new Promise(resolve => ui.captureCanvas.toBlob(resolve, "image/png", 0.96));
  state.captures.push({ blob, landmarks: structuredClone(state.landmarks), pose: poses[state.step].title });
  state.step += 1;
  updateProgress();

  if (state.step >= poses.length) {
    state.detecting = false;
    ui.directionHint.hidden = true;
    ui.captureButton.disabled = true;
    ui.cameraNote.textContent = "Gerando a malha 3D no seu dispositivo…";
    await generateModel();
    ui.resultCard.hidden = false;
    ui.cameraNote.textContent = "Pronto. Revise o pacote no outro Codex antes de adaptar ao PES 2021.";
    ui.resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
  } else {
    setPose(state.step);
    ui.captureButton.disabled = false;
  }
}

async function runCountdown() {
  ui.countdown.hidden = false;
  for (const n of [3, 2, 1]) {
    ui.countdown.textContent = n;
    await new Promise(resolve => setTimeout(resolve, 540));
  }
  ui.countdown.hidden = true;
}

function updateProgress() {
  const percent = Math.round((state.step / poses.length) * 100);
  ui.progressLabel.textContent = `${state.step} de ${poses.length}`;
  ui.progressPercent.textContent = `${percent}%`;
  ui.progressRing.style.setProperty("--progress", percent);
  [...ui.steps.children].forEach((item, i) => {
    item.classList.toggle("done", i < state.step);
    item.classList.toggle("active", i === state.step && state.step < poses.length);
  });
}

function getTriangles() {
  const c = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
  const triangles = [];
  for (let i = 0; i + 2 < c.length; i += 3) {
    const a = c[i], b = c[i + 1], d = c[i + 2];
    if (a.end === b.start && b.end === d.start && d.end === a.start) triangles.push([a.start, a.end, b.end]);
  }
  return triangles;
}

function normalizedVertices(landmarks) {
  const left = landmarks[234], right = landmarks[454];
  const scale = 1.65 / Math.max(0.05, Math.hypot(right.x - left.x, right.y - left.y));
  const center = landmarks[1];
  return landmarks.map(p => ({
    x: (p.x - center.x) * scale,
    y: -(p.y - center.y) * scale,
    z: -p.z * scale * 1.18,
    u: p.x,
    v: 1 - p.y
  }));
}

async function generateModel() {
  const base = state.captures[0];
  const vertices = normalizedVertices(base.landmarks);
  const triangles = getTriangles();
  state.textureBlob = base.blob;

  const obj = ["# FaceScan 3D — malha facial aproximada", "mtllib rosto.mtl", "o rosto", "usemtl pele"];
  vertices.forEach(v => obj.push(`v ${v.x.toFixed(6)} ${v.y.toFixed(6)} ${v.z.toFixed(6)}`));
  vertices.forEach(v => obj.push(`vt ${v.u.toFixed(6)} ${v.v.toFixed(6)}`));
  triangles.forEach(t => obj.push(`f ${t[0] + 1}/${t[0] + 1} ${t[1] + 1}/${t[1] + 1} ${t[2] + 1}/${t[2] + 1}`));
  state.objText = obj.join("\n");
  state.mtlText = ["newmtl pele", "Ka 1.000 1.000 1.000", "Kd 1.000 1.000 1.000", "Ks 0.000 0.000 0.000", "d 1.0", "illum 1", "map_Kd textura_frontal.png"].join("\n");

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices.flatMap(v => [v.x, v.y, v.z]), 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(vertices.flatMap(v => [v.u, v.v]), 2));
  geometry.setIndex(triangles.flat());
  geometry.computeVertexNormals();
  const imageBitmap = await createImageBitmap(base.blob);
  const texture = new THREE.CanvasTexture(imageBitmap);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.flipY = false;
  const material = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.86, metalness: 0, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "Rosto_Escaneado";
  const scene = new THREE.Scene();
  scene.add(mesh);
  const exporter = new GLTFExporter();
  const glb = await exporter.parseAsync(scene, { binary: true, trs: false, onlyVisible: true });
  state.glbBlob = new Blob([glb], { type: "model/gltf-binary" });
}

async function downloadPackage() {
  const zip = new JSZip();
  zip.file("modelo/rosto.glb", state.glbBlob);
  zip.file("modelo/rosto.obj", state.objText);
  zip.file("modelo/rosto.mtl", state.mtlText);
  zip.file("modelo/textura_frontal.png", state.textureBlob);
  const refs = zip.folder("referencias");
  state.captures.forEach((capture, i) => refs.file(`${String(i + 1).padStart(2, "0")}_${slug(capture.pose)}.png`, capture.blob));
  zip.file("dados/pontos_faciais.json", JSON.stringify({
    generator: "FaceScan 3D",
    note: "Malha aproximada baseada em pontos faciais; use as imagens de referência para refinamento.",
    captures: state.captures.map(c => ({ pose: c.pose, landmarks: c.landmarks }))
  }, null, 2));
  zip.file("LEIA-ME.txt", "Pacote FaceScan 3D\n\nEnvie este ZIP ao Codex responsável pela adaptação ao PES 2021.\nA malha é uma aproximação baseada em pontos faciais e deve ser refinada usando as cinco imagens da pasta referencias.\n");
  const blob = await zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } });
  saveBlob(blob, "facescan-3d.zip");
}

function slug(value) { return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, ""); }
function saveBlob(blob, name) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }

function resetScan() {
  state.captures = []; state.step = 0; state.landmarks = null; state.glbBlob = null;
  state.detecting = Boolean(state.stream);
  ui.resultCard.hidden = true;
  ui.captureButton.disabled = !state.stream;
  ui.directionHint.hidden = !state.stream;
  setPose(0); updateProgress();
  if (state.stream) requestAnimationFrame(detectLoop);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function registerWebMCP() {
  const context = document.modelContext;
  if (!context?.registerTool) return;
  try {
    context.registerTool({
      name: "get_face_scan_status",
      title: "Consultar escaneamento facial",
      description: "Informa o progresso visível do escaneamento sem acessar imagens ou dados biométricos.",
      inputSchema: { type: "object", properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: true, untrustedContentHint: false },
      execute: async () => ({ capturesCompleted: state.step, capturesRequired: poses.length, readyForDownload: Boolean(state.glbBlob) })
    });
  } catch (error) { console.debug("WebMCP indisponível", error); }
}

ui.startButton.addEventListener("click", startScan);
ui.captureButton.addEventListener("click", capturePose);
ui.resetButton.addEventListener("click", resetScan);
ui.downloadButton.addEventListener("click", downloadPackage);
ui.downloadGlbButton.addEventListener("click", () => saveBlob(state.glbBlob, "rosto.glb"));
window.addEventListener("beforeunload", () => state.stream?.getTracks().forEach(track => track.stop()));
registerWebMCP();
