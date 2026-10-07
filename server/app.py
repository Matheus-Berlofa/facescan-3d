from __future__ import annotations

import asyncio
import io
import os
import secrets
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

from engine import FrameCandidate, create_engine

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

ACCESS_TOKEN = os.getenv("FACESCAN_TOKEN", "")
ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FACESCAN_ALLOWED_ORIGINS",
        "https://matheus-berlofa.github.io,http://127.0.0.1:4317,http://localhost:4317",
    ).split(",")
    if origin.strip()
}
JOB_TTL_SECONDS = max(5, int(os.getenv("FACESCAN_JOB_TTL_MINUTES", "30"))) * 60
MAX_FRAME_BYTES = 500_000
MAX_FRAMES_PER_TARGET = 2
MAX_TOTAL_FRAMES = 24
JOBS_ROOT = ROOT / "jobs"

app = FastAPI(title="FaceScan 3D Private Processor", version="2.0.0", docs_url=None, redoc_url=None)
engine = create_engine()


@dataclass
class ScanSession:
    identifier: str
    frames: dict[float, list[FrameCandidate]] = field(default_factory=dict)
    pending_metadata: dict | None = None

    def add(self, candidate: FrameCandidate) -> None:
        bucket = self.frames.setdefault(candidate.target_angle, [])
        bucket.append(candidate)
        bucket.sort(key=lambda item: item.quality, reverse=True)
        if len(bucket) > MAX_FRAMES_PER_TARGET:
            removed = bucket.pop()
            removed.data = b""

    def all_frames(self) -> list[FrameCandidate]:
        return [candidate for bucket in self.frames.values() for candidate in bucket]

    def wipe(self) -> None:
        for candidate in self.all_frames():
            candidate.data = b""
        self.frames.clear()
        self.pending_metadata = None


@dataclass
class JobArtifact:
    package: Path
    ticket: str
    created_at: float


jobs: dict[str, JobArtifact] = {}
cleanup_task: asyncio.Task | None = None


@app.on_event("startup")
async def start_cleanup_task() -> None:
    global cleanup_task
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    cleanup_task = asyncio.create_task(cleanup_loop())


@app.on_event("shutdown")
async def stop_cleanup_task() -> None:
    if cleanup_task:
        cleanup_task.cancel()


@app.get("/health")
async def health() -> dict:
    status = engine.status()
    return {"ok": True, "engineReady": status.ready, "engine": status.name, "message": status.message}


@app.get("/download/{job_id}")
async def download(job_id: str, ticket: str = Query(...)) -> FileResponse:
    cleanup_expired_jobs()
    artifact = jobs.get(job_id)
    if not artifact or not secrets.compare_digest(artifact.ticket, ticket):
        raise HTTPException(status_code=404, detail="Resultado não encontrado ou expirado.")
    if not artifact.package.exists():
        raise HTTPException(status_code=410, detail="Resultado expirado.")
    return FileResponse(artifact.package, media_type="application/zip", filename="facescan-neural.zip")


@app.websocket("/ws/scan")
async def scan_socket(websocket: WebSocket) -> None:
    origin = (websocket.headers.get("origin") or "").rstrip("/")
    if origin and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008, reason="Origem não autorizada")
        return
    await websocket.accept()
    session = ScanSession(identifier=uuid.uuid4().hex)
    try:
        if not ACCESS_TOKEN:
            await websocket.send_json({"type": "error", "message": "FACESCAN_TOKEN não foi configurado no servidor."})
            await websocket.close(code=1011)
            return
        auth_message = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        supplied_key = str(auth_message.get("key", "")) if auth_message.get("type") == "auth" else ""
        if not secrets.compare_digest(supplied_key, ACCESS_TOKEN):
            await websocket.close(code=4001, reason="Chave incorreta")
            return

        cleanup_expired_jobs()
        status = engine.status()
        await websocket.send_json({
            "type": "hello",
            "sessionId": session.identifier,
            "engineReady": status.ready,
            "engine": status.name,
            "message": status.message,
        })

        while True:
            event = await websocket.receive()
            if event.get("text") is not None:
                await handle_text_message(websocket, session, event["text"])
            elif event.get("bytes") is not None:
                await handle_binary_frame(websocket, session, event["bytes"])
            elif event.get("type") == "websocket.disconnect":
                break
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        session.wipe()


async def handle_text_message(websocket: WebSocket, session: ScanSession, raw_message: str) -> None:
    import json

    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError:
        await websocket.send_json({"type": "error", "message": "Mensagem inválida."})
        return
    message_type = message.get("type")
    if message_type == "start":
        session.wipe()
        await websocket.send_json({"type": "processing", "title": "Scan iniciado", "message": "Aguardando os ângulos guiados."})
    elif message_type == "frame":
        session.pending_metadata = message
    elif message_type == "cancel":
        session.wipe()
    elif message_type == "finish":
        await finish_scan(websocket, session)


async def handle_binary_frame(websocket: WebSocket, session: ScanSession, data: bytes) -> None:
    metadata = session.pending_metadata
    session.pending_metadata = None
    if not metadata:
        await websocket.send_json({"type": "frame_rejected", "reason": "Metadados do quadro ausentes."})
        return
    if len(data) > MAX_FRAME_BYTES:
        await websocket.send_json({"type": "frame_rejected", "reason": "Quadro grande demais."})
        return
    if len(session.all_frames()) >= MAX_TOTAL_FRAMES:
        await websocket.send_json({"type": "frame_rejected", "reason": "Limite de quadros atingido."})
        return

    quality, reason = analyze_frame(data)
    if quality is None:
        await websocket.send_json({"type": "frame_rejected", "reason": reason})
        return
    target = float(metadata.get("targetAngle", 0))
    measured = float(metadata.get("measuredYaw", target))
    session.add(FrameCandidate(data=data, target_angle=target, measured_yaw=measured, quality=quality))
    await websocket.send_json({
        "type": "frame_accepted",
        "targetAngle": target,
        "quality": round(quality, 2),
        "storedFrames": len(session.all_frames()),
    })


async def finish_scan(websocket: WebSocket, session: ScanSession) -> None:
    frames = session.all_frames()
    unique_angles = {candidate.target_angle for candidate in frames}
    if len(unique_angles) < 9 or len(frames) < 12:
        await websocket.send_json({"type": "error", "message": "Cobertura insuficiente. Repita o scan com todos os ângulos."})
        return

    job_id = uuid.uuid4().hex
    ticket = secrets.token_urlsafe(32)
    output_dir = JOBS_ROOT / job_id
    loop = asyncio.get_running_loop()

    def progress(title: str, message: str) -> None:
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "processing", "title": title, "message": message}),
            loop,
        )

    try:
        await websocket.send_json({"type": "processing", "title": "Iniciando a GPU…", "message": "Os quadros continuam somente na memória."})
        package = await asyncio.to_thread(engine.process, list(frames), output_dir, progress)
        jobs[job_id] = JobArtifact(package=package, ticket=ticket, created_at=time.time())
        await websocket.send_json({
            "type": "complete",
            "downloadPath": f"/download/{job_id}",
            "ticket": ticket,
        })
    except Exception as error:
        shutil.rmtree(output_dir, ignore_errors=True)
        await websocket.send_json({"type": "error", "message": f"Falha na reconstrução: {error}"})
    finally:
        session.wipe()


def analyze_frame(data: bytes) -> tuple[float | None, str]:
    try:
        image = Image.open(io.BytesIO(data)).convert("L")
        if image.width < 320 or image.height < 320:
            return None, "Aproxime o rosto da câmera."
        brightness = ImageStat.Stat(image).mean[0]
        if brightness < 45:
            return None, "Imagem escura. Aumente a iluminação frontal."
        if brightness > 238:
            return None, "Imagem clara demais. Reduza a luz direta."
        edges = np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        sharpness = float(edges.var())
        if sharpness < 55:
            return None, "Imagem desfocada. Segure a posição por um instante."
        exposure_score = max(0.0, 1.0 - abs(brightness - 135.0) / 135.0)
        quality = sharpness * 0.75 + exposure_score * 100.0
        return quality, ""
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "Quadro inválido recebido."


def cleanup_expired_jobs() -> None:
    now = time.time()
    expired = [job_id for job_id, artifact in jobs.items() if now - artifact.created_at > JOB_TTL_SECONDS]
    for job_id in expired:
        artifact = jobs.pop(job_id)
        shutil.rmtree(artifact.package.parent, ignore_errors=True)


async def cleanup_loop() -> None:
    while True:
        await asyncio.sleep(60)
        cleanup_expired_jobs()

