import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/+esm";

const ui = Object.fromEntries([
  "serverForm", "serverUrl", "accessKey", "connectButton", "serverDot", "serverStatus", "video", "overlay",
  "idleVisual", "cameraBadge", "directionHint", "directionIcon", "directionTitle", "directionText", "scanSweep",
  "startButton", "scanButton", "resetButton", "cameraNote", "progressLabel", "progressRing", "progressPercent",
  "angleMap", "yawValue", "frameCount", "engineStatus", "lightDot", "lightStatus", "faceDot", "faceStatus",
  "poseDot", "poseStatus", "processingCard", "processingTitle", "processingText", "resultCard", "downloadButton",
  "captureCanvas"
].map(id => [id, document.getElementById(id)]));

const TARGETS = [-45, -34, -23, -12, 0, 12, 23, 34, 45];
const FRAMES_PER_TARGET = 2;
const TARGET_TOLERANCE = 7;
const MIN_SEND_INTERVAL = 420;

const state = {
  stream: null,
  landmarker: null,
  landmarks: null,
  socket: null,
  socketReady: false,
  engineReady: false,
  scanning: false,
  processing: false,
  targetCounts: new Map(TARGETS.map(angle => [angle, 0])),
  acceptedFrames: 0,
  yaw: 0,
  lightOkay: false,
  sending: false,
  lastSentAt: 0,
  sessionId: null
};

const overlayCtx = ui.overlay.getContext("2d");
const captureCtx = ui.captureCanvas.getContext("2d", { willReadFrequently: true });

restoreConnectionFields();
updateProgress();

async function createLandmarker() {
  if (state.landmarker) return state.landmarker;
  ui.cameraNote.textContent = "Preparando o rastreamento facial…";
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm");
  state.landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.65,
    minFacePresenceConfidence: 0.65,
    minTrackingConfidence: 0.6,
    outputFacialTransformationMatrixes: true
  });
  return state.landmarker;
}

async function startCamera() {
  try {
    ui.startButton.disabled = true;
    const [landmarker, stream] = await Promise.all([
      createLandmarker(),
      navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
        audio: false
      })
    ]);
    state.landmarker = landmarker;
    state.stream = stream;
    ui.video.srcObject = stream;
    await ui.video.play();
    ui.idleVisual.hidden = true;
    ui.directionHint.hidden = false;
    ui.cameraBadge.textContent = "Câmera ativa";
    ui.resetButton.disabled = false;
    ui.cameraNote.textContent = "Rastreamento ativo. Conecte o processador privado para iniciar o scan.";
    refreshScanButton();
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
  if (!state.stream || !state.landmarker || ui.video.readyState < 2) {
    if (state.stream) requestAnimationFrame(detectLoop);
    return;
  }
  ui.overlay.width = ui.video.videoWidth;
  ui.overlay.height = ui.video.videoHeight;
  const result = state.landmarker.detectForVideo(ui.video, performance.now());
  overlayCtx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  state.landmarks = result.faceLandmarks?.[0] || null;

  if (state.landmarks) {
    const drawing = new DrawingUtils(overlayCtx);
    drawing.drawConnectors(state.landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#79f2c0", lineWidth: 2 });
    drawing.drawConnectors(state.landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "rgba(201,255,96,.72)", lineWidth: 1 });
    drawing.drawConnectors(state.landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "rgba(201,255,96,.72)", lineWidth: 1 });
    drawing.drawConnectors(state.landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "rgba(121,242,192,.72)", lineWidth: 1 });
    state.yaw = estimateYaw(state.landmarks);
    state.lightOkay = estimateLight();
    updateLiveQuality();
    if (state.scanning) maybeStreamFrame();
  } else {
    ui.faceDot.className = "warn";
    ui.faceStatus.textContent = "Centralize o rosto";
    ui.lightDot.className = "";
    ui.lightStatus.textContent = "Aguardando rosto";
    ui.poseDot.className = "";
    ui.poseStatus.textContent = "Rosto fora do quadro";
    ui.yawValue.textContent = "—";
  }
  requestAnimationFrame(detectLoop);
}

function estimateYaw(landmarks) {
  const left = landmarks[234];
  const right = landmarks[454];
  const nose = landmarks[1];
  const width = Math.max(.01, Math.abs(right.x - left.x));
  const midpoint = (left.x + right.x) / 2;
  return clamp(((nose.x - midpoint) / width) * 115, -55, 55);
}

function estimateLight() {
  const sample = document.createElement("canvas");
  sample.width = 64;
  sample.height = 48;
  const context = sample.getContext("2d", { willReadFrequently: true });
  context.drawImage(ui.video, 0, 0, 64, 48);
  const data = context.getImageData(0, 0, 64, 48).data;
  let total = 0;
  for (let index = 0; index < data.length; index += 16) total += (data[index] + data[index + 1] + data[index + 2]) / 3;
  const average = total / (data.length / 16);
  return average > 58 && average < 232;
}

function updateLiveQuality() {
  ui.faceDot.className = "good";
  ui.faceStatus.textContent = "Rosto detectado";
  ui.lightDot.className = state.lightOkay ? "good" : "warn";
  ui.lightStatus.textContent = state.lightOkay ? "Iluminação adequada" : "Ajuste a iluminação";
  ui.yawValue.textContent = `${Math.round(state.yaw)}°`;

  if (!state.scanning) {
    ui.poseDot.className = "";
    ui.poseStatus.textContent = "Pronto para iniciar";
    ui.directionTitle.textContent = "Mantenha expressão neutra";
    ui.directionText.textContent = "O guia indicará cada ângulo durante o scan";
    ui.directionIcon.textContent = "●";
    return;
  }

  const target = getRecommendedTarget();
  if (target === null) return;
  const difference = Math.abs(state.yaw - target);
  ui.poseDot.className = difference <= TARGET_TOLERANCE ? "ready" : "warn";
  ui.poseStatus.textContent = difference <= TARGET_TOLERANCE ? "Posição correta — capturando" : `Buscando ${formatAngle(target)}`;
  updateDirection(target, difference);
}

function updateDirection(target, difference) {
  if (difference <= TARGET_TOLERANCE) {
    ui.directionIcon.textContent = "✓";
    ui.directionTitle.textContent = "Segure a posição";
    ui.directionText.textContent = "Capturando os melhores quadros na memória";
    return;
  }
  const turnRight = state.yaw < target;
  ui.directionIcon.textContent = turnRight ? "→" : "←";
  ui.directionTitle.textContent = turnRight ? "Gire devagar para a direita" : "Gire devagar para a esquerda";
  ui.directionText.textContent = `Alvo ${formatAngle(target)} · atual ${Math.round(state.yaw)}°`;
}

async function maybeStreamFrame() {
  const now = performance.now();
  if (state.sending || !state.lightOkay || !state.landmarks || now - state.lastSentAt < MIN_SEND_INTERVAL) return;
  const target = getRecommendedTarget();
  if (target === null) return;
  if (Math.abs(state.yaw - target) > TARGET_TOLERANCE) return;
  state.sending = true;
  state.lastSentAt = now;
  try {
    const blob = await captureFace();
    if (!blob || !socketIsOpen()) throw new Error("Conexão com o processador perdida.");
    state.socket.send(JSON.stringify({
      type: "frame",
      targetAngle: target,
      measuredYaw: Number(state.yaw.toFixed(2)),
      width: 512,
      height: 512,
      mime: "image/jpeg",
      timestamp: Date.now()
    }));
    state.socket.send(await blob.arrayBuffer());
  } catch (error) {
    state.sending = false;
    failScan(error.message);
  }
}

async function captureFace() {
  const points = state.landmarks;
  const xs = points.map(point => point.x * ui.video.videoWidth);
  const ys = points.map(point => point.y * ui.video.videoHeight);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2 - (maxY - minY) * .04;
  const side = Math.min(
    Math.max(maxX - minX, maxY - minY) * 1.72,
    ui.video.videoWidth,
    ui.video.videoHeight
  );
  const sourceX = clamp(centerX - side / 2, 0, ui.video.videoWidth - side);
  const sourceY = clamp(centerY - side / 2, 0, ui.video.videoHeight - side);
  captureCtx.clearRect(0, 0, 512, 512);
  captureCtx.drawImage(ui.video, sourceX, sourceY, side, side, 0, 0, 512, 512);
  return new Promise(resolve => ui.captureCanvas.toBlob(resolve, "image/jpeg", .92));
}

async function beginLiveScan() {
  if (!state.stream || !state.socketReady || !state.engineReady || state.processing) return;
  resetCaptureState();
  state.scanning = true;
  ui.scanButton.disabled = true;
  ui.startButton.disabled = true;
  ui.scanSweep.hidden = false;
  ui.resultCard.hidden = true;
  ui.processingCard.hidden = true;
  ui.cameraBadge.textContent = "Transmitindo ao processador";
  ui.cameraNote.textContent = "Gire devagar seguindo as setas. Não sorria e não mova os ombros.";
  state.socket.send(JSON.stringify({ type: "start", targets: TARGETS, framesPerTarget: FRAMES_PER_TARGET }));
  updateProgress();
}

function handleFrameAccepted(message) {
  state.sending = false;
  const target = Number(message.targetAngle);
  state.targetCounts.set(target, (state.targetCounts.get(target) || 0) + 1);
  state.acceptedFrames += 1;
  ui.frameCount.textContent = String(state.acceptedFrames);

  if ((state.targetCounts.get(target) || 0) >= FRAMES_PER_TARGET) {
    const marker = ui.angleMap.querySelector(`[data-angle="${target}"]`);
    marker?.classList.add("done");
    marker?.classList.remove("current");
    if (completedTargetCount() >= TARGETS.length) {
      finishCapture();
      return;
    }
    [...ui.angleMap.children].forEach(item => item.classList.remove("current"));
    const nextTarget = getRecommendedTarget();
    ui.angleMap.querySelector(`[data-angle="${nextTarget}"]`)?.classList.add("current");
  }
  updateProgress();
}

function handleFrameRejected(message) {
  state.sending = false;
  ui.cameraNote.textContent = message.reason || "Quadro descartado. Segure o rosto e melhore a iluminação.";
}

function finishCapture() {
  state.scanning = false;
  state.processing = true;
  state.sending = false;
  ui.scanSweep.hidden = true;
  ui.directionHint.hidden = true;
  ui.processingCard.hidden = false;
  ui.processingTitle.textContent = "Reconstruindo seu rosto…";
  ui.processingText.textContent = "A GTX 1060 está combinando os ângulos. Mantenha esta página aberta.";
  ui.cameraBadge.textContent = "Captura concluída";
  ui.cameraNote.textContent = "Captura concluída. O processamento pode levar alguns minutos.";
  state.socket.send(JSON.stringify({ type: "finish" }));
  ui.processingCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

function handleServerMessage(message) {
  switch (message.type) {
    case "hello":
      state.socketReady = true;
      state.engineReady = Boolean(message.engineReady);
      state.sessionId = message.sessionId;
      ui.serverDot.className = `server-dot ${state.engineReady ? "online" : "error"}`;
      ui.serverStatus.textContent = state.engineReady
        ? `GPU conectada · ${message.engine || "motor neural pronto"}`
        : message.message || "Servidor conectado, mas o motor neural ainda não está instalado.";
      ui.engineStatus.textContent = state.engineReady ? (message.engine || "Pronto") : "Configuração pendente";
      ui.connectButton.textContent = "Conectado";
      sessionStorage.setItem("facescanServerUrl", normalizeWsUrl(ui.serverUrl.value));
      sessionStorage.setItem("facescanAccessKey", ui.accessKey.value);
      refreshScanButton();
      break;
    case "frame_accepted":
      handleFrameAccepted(message);
      break;
    case "frame_rejected":
      handleFrameRejected(message);
      break;
    case "processing":
      ui.processingTitle.textContent = message.title || "Reconstruindo seu rosto…";
      ui.processingText.textContent = message.message || "A GPU está processando os quadros efêmeros.";
      break;
    case "complete":
      completeScan(message);
      break;
    case "error":
      state.sending = false;
      failScan(message.message || "O processador encontrou um erro.");
      break;
  }
}

function completeScan(message) {
  state.processing = false;
  ui.processingCard.hidden = true;
  ui.resultCard.hidden = false;
  ui.directionHint.hidden = false;
  ui.downloadButton.href = buildDownloadUrl(message.downloadPath, message.ticket);
  ui.cameraNote.textContent = "Modelo concluído. O servidor já descartou todos os quadros originais.";
  ui.resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

function failScan(message) {
  state.scanning = false;
  state.processing = false;
  state.sending = false;
  ui.scanSweep.hidden = true;
  ui.processingCard.hidden = true;
  ui.directionHint.hidden = false;
  ui.cameraNote.textContent = message;
  ui.serverDot.classList.add("error");
  refreshScanButton();
}

function connectServer(event) {
  event?.preventDefault();
  const serverUrl = normalizeWsUrl(ui.serverUrl.value);
  const accessKey = ui.accessKey.value.trim();
  if (!serverUrl || !accessKey) return;
  state.socket?.close();
  state.socketReady = false;
  state.engineReady = false;
  ui.serverDot.className = "server-dot connecting";
  ui.serverStatus.textContent = "Abrindo canal seguro com o PC remoto…";
  ui.connectButton.disabled = true;
  ui.connectButton.textContent = "Conectando…";

  try {
    const socket = new WebSocket(serverUrl);
    socket.binaryType = "arraybuffer";
    state.socket = socket;
    const timeout = window.setTimeout(() => {
      if (!state.socketReady) socket.close(4000, "Tempo limite");
    }, 9000);
    socket.addEventListener("open", () => socket.send(JSON.stringify({ type: "auth", key: accessKey, client: "facescan-web-v2" })));
    socket.addEventListener("message", event => {
      if (typeof event.data !== "string") return;
      try { handleServerMessage(JSON.parse(event.data)); } catch (error) { console.error("Mensagem inválida", error); }
    });
    socket.addEventListener("close", event => {
      window.clearTimeout(timeout);
      const wasReady = state.socketReady;
      state.socketReady = false;
      state.engineReady = false;
      ui.serverDot.className = "server-dot error";
      ui.connectButton.disabled = false;
      ui.connectButton.textContent = "Testar conexão";
      ui.engineStatus.textContent = "Desconectado";
      if (state.scanning || state.processing) failScan("A conexão com o processador foi interrompida. Reconecte e reinicie o scan.");
      else if (!wasReady) {
        ui.serverDot.className = "server-dot error";
        ui.serverStatus.textContent = event.code === 4001 ? "Chave de acesso incorreta." : "Não foi possível conectar. Confira o endereço WSS e o túnel.";
      }
      refreshScanButton();
    });
    socket.addEventListener("error", () => {
      ui.serverDot.className = "server-dot error";
      ui.serverStatus.textContent = "Falha na conexão segura com o processador.";
    });
  } catch (error) {
    ui.connectButton.disabled = false;
    ui.connectButton.textContent = "Testar conexão";
    ui.serverDot.className = "server-dot error";
    ui.serverStatus.textContent = error.message;
  }
}

function updateProgress() {
  const completed = completedTargetCount();
  const percent = Math.round((completed / TARGETS.length) * 100);
  ui.progressLabel.textContent = `${completed} de ${TARGETS.length} ângulos`;
  ui.progressPercent.textContent = `${percent}%`;
  ui.progressRing.style.setProperty("--progress", percent);
}

function resetCaptureState() {
  state.targetCounts = new Map(TARGETS.map(angle => [angle, 0]));
  state.acceptedFrames = 0;
  state.sending = false;
  state.lastSentAt = 0;
  ui.frameCount.textContent = "0";
  [...ui.angleMap.children].forEach(marker => marker.classList.remove("done", "current"));
  ui.angleMap.querySelector(`[data-angle="${TARGETS[0]}"]`)?.classList.add("current");
  updateProgress();
}

function completedTargetCount() {
  return TARGETS.filter(angle => (state.targetCounts.get(angle) || 0) >= FRAMES_PER_TARGET).length;
}

function getRecommendedTarget() {
  const remaining = TARGETS.filter(angle => (state.targetCounts.get(angle) || 0) < FRAMES_PER_TARGET);
  if (!remaining.length) return null;
  return remaining.reduce((closest, angle) =>
    Math.abs(angle - state.yaw) < Math.abs(closest - state.yaw) ? angle : closest
  );
}

function resetScan() {
  if (state.scanning && socketIsOpen()) state.socket.send(JSON.stringify({ type: "cancel" }));
  state.scanning = false;
  state.processing = false;
  resetCaptureState();
  ui.processingCard.hidden = true;
  ui.resultCard.hidden = true;
  ui.scanSweep.hidden = true;
  ui.directionHint.hidden = !state.stream;
  ui.cameraBadge.textContent = state.stream ? "Câmera ativa" : "Câmera desligada";
  ui.cameraNote.textContent = state.stream ? "Pronto para um novo scan." : "A câmera só será ativada depois da sua permissão.";
  refreshScanButton();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function refreshScanButton() {
  ui.scanButton.disabled = !(state.stream && state.socketReady && state.engineReady) || state.scanning || state.processing;
}

function socketIsOpen() {
  return state.socket?.readyState === WebSocket.OPEN && state.socketReady;
}

function normalizeWsUrl(value) {
  let url = value.trim();
  if (!url) return "";
  url = url.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
  if (!/^wss?:\/\//i.test(url)) url = `wss://${url}`;
  const parsed = new URL(url);
  parsed.pathname = parsed.pathname.replace(/\/$/, "") || "/ws/scan";
  if (!parsed.pathname.endsWith("/ws/scan")) parsed.pathname = `${parsed.pathname}/ws/scan`.replace(/\/+/g, "/");
  return parsed.toString();
}

function buildDownloadUrl(path, ticket) {
  const socketUrl = new URL(normalizeWsUrl(ui.serverUrl.value));
  socketUrl.protocol = socketUrl.protocol === "wss:" ? "https:" : "http:";
  socketUrl.pathname = path;
  socketUrl.search = `ticket=${encodeURIComponent(ticket)}`;
  return socketUrl.toString();
}

function restoreConnectionFields() {
  ui.serverUrl.value = sessionStorage.getItem("facescanServerUrl") || "";
  ui.accessKey.value = sessionStorage.getItem("facescanAccessKey") || "";
}

function formatAngle(angle) {
  return `${angle > 0 ? "+" : ""}${angle}°`;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

ui.serverForm.addEventListener("submit", connectServer);
ui.startButton.addEventListener("click", startCamera);
ui.scanButton.addEventListener("click", beginLiveScan);
ui.resetButton.addEventListener("click", resetScan);
window.addEventListener("beforeunload", () => {
  state.stream?.getTracks().forEach(track => track.stop());
  state.socket?.close();
});
