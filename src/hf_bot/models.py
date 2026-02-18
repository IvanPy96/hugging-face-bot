"""Domain data models: ModelInfo, DeployInfo, and derivative-model detection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Derivative-model detection
# ---------------------------------------------------------------------------

DERIVATIVE_SUFFIXES: tuple[str, ...] = (
    "-gguf", "-fp8", "-fp4", "-bf16", "-int4", "-int8",
    "-awq", "-gptq", "-nvfp4", "-onnx",
    "-base", "-pretrain", "-original",
    "-eagle", "-unquantized",
)


def is_derivative_model(model_id: str) -> bool:
    """Return True when the model is a technical variant."""
    name = model_id.split("/", 1)[-1].lower()
    return name.endswith(DERIVATIVE_SUFFIXES)


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ModelInfo:
    """Structured representation of a Hugging Face model."""

    id: str
    author: str
    name: str
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    last_modified: str = ""
    private: bool = False
    library_name: str | None = None
    safetensors: dict[str, Any] | None = None

    @property
    def url(self) -> str:
        """Full URL on Hugging Face Hub."""
        return f"https://huggingface.co/{self.id}"

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ModelInfo:
        """Build from a raw HF API JSON object."""
        model_id: str = data.get("modelId") or data.get("id") or data.get("_id") or ""
        author, _, name = model_id.partition("/")
        if not name:
            author, name = "", model_id
        return cls(
            id=model_id,
            author=author,
            name=name,
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            tags=data.get("tags", []),
            pipeline_tag=data.get("pipeline_tag"),
            last_modified=data.get("lastModified", ""),
            private=data.get("private", False),
            library_name=data.get("library_name"),
            safetensors=data.get("safetensors"),
        )

    # ----- formatting for LLM context --------------------------------------

    def useful_tags(self, limit: int = 10) -> list[str]:
        """Filter out technical noise tags and return the most informative ones."""
        skip = {"transformers", "pytorch", "safetensors", self.pipeline_tag, self.library_name}
        return [
            t for t in self.tags
            if not t.startswith(("license:", "arxiv:")) and t not in skip
        ][:limit]

    def to_context(self, *, readme: str | None = None) -> str:
        """Format model info as plain text for LLM context injection."""
        header = f"=== {self.id} ===" if readme else f"ID: {self.id}"
        lines: list[str] = [
            header,
            f"URL: {self.url}",
            f"Downloads: {self.downloads:,}",
            f"Likes: {self.likes:,}",
        ]
        if self.pipeline_tag:
            lines.append(f"Task: {self.pipeline_tag}")
        if self.library_name:
            lines.append(f"Library: {self.library_name}")
        useful = self.useful_tags(8 if readme else 10)
        if useful:
            lines.append(f"Tags: {', '.join(useful)}")
        if not readme and self.last_modified:
            lines.append(f"Last modified: {self.last_modified[:10]}")
        if readme:
            lines += ["", "--- README/Model Card ---", readme]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DeployInfo — GPU requirements calculator
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DeployInfo:
    """Result of GPU deployment requirements calculation."""

    total_params: int
    weight_gb: float
    total_gb: float
    dtype: str
    h200_count: int
    l40s_fits: bool

    _BYTES: ClassVar[dict[str, float]] = {
        "F64": 8, "I64": 8, "F32": 4, "I32": 4,
        "F16": 2, "BF16": 2, "I16": 2,
        "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
        "BOOL": 0.125,
    }
    _OVERHEAD: ClassVar[float] = 1.20
    _H200_VRAM: ClassVar[int] = 140
    _L40S_VRAM: ClassVar[int] = 48

    @classmethod
    def from_model(cls, model: ModelInfo) -> DeployInfo | None:
        """Calculate deployment requirements from model safetensors metadata."""
        if not model.safetensors:
            return None
        params = model.safetensors.get("parameters", {})
        if not params:
            return None

        total, weight_bytes = 0, 0.0
        main_dtype: str | None = None
        max_cnt = 0
        for dtype, count in params.items():
            bpd = cls._BYTES.get(dtype, 2)
            weight_bytes += count * bpd
            total += count
            if count > max_cnt:
                max_cnt, main_dtype = count, dtype

        if total == 0:
            return None

        weight_gb = weight_bytes / (1024**3)
        total_gb = weight_gb * cls._OVERHEAD
        return cls(
            total_params=total,
            weight_gb=weight_gb,
            total_gb=total_gb,
            dtype=main_dtype or "unknown",
            h200_count=math.ceil(total_gb / cls._H200_VRAM),
            l40s_fits=total_gb <= cls._L40S_VRAM,
        )
