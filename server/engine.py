from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image


@dataclass
class FrameCandidate:
    data: bytes
    target_angle: float
    measured_yaw: float
    quality: float


@dataclass(frozen=True)
class EngineStatus:
    ready: bool
    name: str
    message: str


class DecaEngine:
    """Adaptador multi-view para o DECA sem persistir os quadros de câmera."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        configured = root or os.getenv("DECA_ROOT", "")
        self.root = Path(configured).expanduser().resolve() if configured else None
        self._deca = None
        self._torch = None
        self._use_texture_model = False

    def status(self) -> EngineStatus:
        if not self.root:
            return EngineStatus(False, "DECA + FLAME", "DECA_ROOT não foi configurado no arquivo .env.")
        required = [
            self.root / "decalib" / "deca.py",
            self.root / "data" / "deca_model.tar",
            self.root / "data" / "generic_model.pkl",
        ]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            return EngineStatus(False, "DECA + FLAME", f"Arquivos do motor ausentes: {', '.join(missing)}.")
        try:
            import torch
        except ImportError:
            return EngineStatus(False, "DECA + FLAME", "PyTorch com CUDA ainda não está instalado no ambiente do servidor.")
        if not torch.cuda.is_available():
            return EngineStatus(False, "DECA + FLAME", "PyTorch não encontrou a GPU CUDA.")
        return EngineStatus(True, f"DECA + FLAME · {torch.cuda.get_device_name(0)}", "Motor neural pronto.")

    def process(
        self,
        frames: Iterable[FrameCandidate],
        output_dir: Path,
        progress: Callable[[str, str], None] | None = None,
    ) -> Path:
        status = self.status()
        if not status.ready:
            raise RuntimeError(status.message)

        candidates = list(frames)
        if len(candidates) < 9:
            raise RuntimeError("Poucos ângulos válidos para reconstruir o rosto.")

        notify = progress or (lambda _title, _message: None)
        notify("Carregando o motor neural…", "Inicializando DECA e FLAME na GTX 1060.")
        self._load()
        torch = self._torch
        output_dir.mkdir(parents=True, exist_ok=True)

        notify("Analisando os ângulos…", "Estimando identidade, forma e detalhes de cada vista.")
        encoded = []
        for candidate in candidates:
            image = self._tensor_from_bytes(candidate.data).to("cuda")
            with torch.no_grad():
                code = self._deca.encode(image.unsqueeze(0))
            encoded.append((candidate, code))

        frontal_index = min(range(len(encoded)), key=lambda index: abs(encoded[index][0].measured_yaw))
        base_candidate, fused = encoded[frontal_index]
        identity_keys = ("shape", "detail")
        for key in identity_keys:
            values = [code[key] for _, code in encoded if key in code]
            if values:
                fused[key] = torch.stack(values, dim=0).mean(dim=0)

        if self._use_texture_model:
            texture_values = [code["tex"] for _, code in encoded if "tex" in code]
            if texture_values:
                fused["tex"] = torch.stack(texture_values, dim=0).median(dim=0).values

        notify("Construindo a malha…", "Fundindo as vistas em uma única identidade facial.")
        with torch.no_grad():
            operation, _visualization = self._deca.decode(fused)

        model_dir = output_dir / "modelo"
        model_dir.mkdir(parents=True, exist_ok=True)
        obj_path = model_dir / "rosto.obj"
        self._deca.save_obj(str(obj_path), operation)

        glb_path = model_dir / "rosto.glb"
        try:
            import trimesh

            mesh = trimesh.load(obj_path, process=False)
            mesh.export(glb_path)
        except Exception as error:
            (output_dir / "aviso-glb.txt").write_text(
                f"O OBJ foi criado corretamente, mas a conversão automática para GLB falhou: {error}\n",
                encoding="utf-8",
            )

        metadata = {
            "generator": "FaceScan 3D Live",
            "engine": "DECA + FLAME multi-view identity fusion",
            "inputFrames": len(candidates),
            "framesPersisted": False,
            "frontalYaw": base_candidate.measured_yaw,
            "angles": sorted({candidate.target_angle for candidate in candidates}),
        }
        (output_dir / "reconstrucao.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "LEIA-ME.txt").write_text(
            "FaceScan 3D Live\n\n"
            "A pasta modelo contém a reconstrução neural para refinamento e adaptação ao PES 2021.\n"
            "Os quadros originais foram processados em memória e não fazem parte deste pacote.\n",
            encoding="utf-8",
        )

        notify("Compactando o resultado…", "Preparando OBJ, GLB, textura e metadados para download.")
        package_path = output_dir / "facescan-neural.zip"
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in output_dir.rglob("*"):
                if path.is_file() and path != package_path:
                    archive.write(path, path.relative_to(output_dir))
        return package_path

    def _load(self) -> None:
        if self._deca is not None:
            return
        assert self.root is not None
        sys.path.insert(0, str(self.root))
        import torch
        from decalib.deca import DECA
        from decalib.utils.config import cfg as deca_config

        deca_config.model.use_tex = (self.root / "data" / "FLAME_albedo_from_BFM.npz").exists()
        deca_config.rasterizer_type = "standard"
        self._use_texture_model = bool(deca_config.model.use_tex)
        self._torch = torch
        self._deca = DECA(config=deca_config, device="cuda")

    def _tensor_from_bytes(self, data: bytes):
        assert self._torch is not None
        image = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return self._torch.from_numpy(array).permute(2, 0, 1).contiguous()


def create_engine() -> DecaEngine:
    engine_name = os.getenv("FACESCAN_ENGINE", "deca").strip().lower()
    if engine_name != "deca":
        raise RuntimeError(f"Motor desconhecido: {engine_name}")
    return DecaEngine()
