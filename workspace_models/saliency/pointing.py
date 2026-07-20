"""Open-vocabulary pointing backends (MolmoPoint) + a mock for testing.

A `Pointer` maps (image, object name) -> list of normalized points (x, y) in [0, 1]. The salient
patches are the DINO patches those points land in. Backends are pluggable so the labeling pipeline
runs end-to-end without the heavy VLM (via `MockPointer`); `MolmoPointer` loads the real model.
"""

from abc import ABC, abstractmethod

import numpy as np


class Pointer(ABC):
    @abstractmethod
    def point(self, image: np.ndarray, obj: str) -> list[tuple[float, float]]:
        """image: (H, W, 3) uint8 RGB -> list of (x, y) normalized to [0, 1]."""


class MockPointer(Pointer):
    """Deterministic stand-in (no VLM). Points depend only on the object string -> reproducible."""

    def point(self, image, obj):
        h = abs(hash(obj))
        return [((h % 997) / 997.0, ((h // 997) % 991) / 991.0)]


class MolmoPointer(Pointer):
    """MolmoPoint-8B (allenai/MolmoPoint-8B) — the dedicated open-vocab pointing model the paper
    uses. Points are emitted as grounding tokens and decoded via model.extract_image_points ->
    pixel coords, which we normalize to [0, 1]. Requires the weights + a GPU.
    """

    def __init__(self, model_id: str = "allenai/MolmoPoint-8B", device: str = "cuda"):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("MolmoPointer needs a recent `transformers` + `torch`.") from e
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(   # bf16 (~16GB) fits a 48GB GPU;
            model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto").eval()

    def point(self, image, obj):
        from PIL import Image
        torch = self._torch
        messages = [{"role": "user", "content": [
            {"type": "text", "text": f"Point to the {obj}"},
            {"type": "image", "image": Image.fromarray(image)},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, return_pointing_metadata=True)
        meta = inputs.pop("metadata")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                **inputs, logits_processor=self.model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=64)   # a point is a few grounding tokens; 64 is ample and faster
        gen = out[:, inputs["input_ids"].size(1):]
        text = self.processor.post_process_image_text_to_text(gen, skip_special_tokens=False)[0]
        pts = self.model.extract_image_points(
            text, meta["token_pooling"], meta["subpatch_mapping"], meta["image_sizes"])
        H, W = image.shape[:2]                       # extract returns pixel coords -> normalize to [0,1]
        return [(min(px / W, 1.0), min(py / H, 1.0)) for (*_, px, py) in pts]
