"""Frozen VL feature cache extraction for B0/B1 program.

Precomputes PaliGemma (or SigLIP fallback) vision-language features for all RoboCasa
demo frames, stored as fp16 npz files with manifest.json for training.

Supports two extraction variants:
  (a) pooled: mean-pooled last-layer vision features (D,) + instruction embedding (D_l,)
  (b) grid16: 4x4 spatial grid of pooled tokens (16, D) + instruction embedding

Cache format:
  /data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/<variant>/
    manifest.json
    features/<task>/<episode_stem>.npz

Episode npz contains:
  base: (N_frames, D) for pooled, (N_frames, 16, D) for grid16
  wrist: (N_frames, D) for pooled, (N_frames, 16, D) for grid16
  lang: (D_l,) — instruction embedding (same for all frames in episode)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# Try PaliGemma, fallback to SigLIP
try:
    from transformers import AutoModel, AutoProcessor, PaliGemmaProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


class VLFeatureExtractor:
    """Extract frozen VL features from RoboCasa episodes."""

    def __init__(
        self,
        model_id: str = "google/paligemma-3b-pt-224",
        variant: Literal["pooled", "grid16"] = "pooled",
        device: str = "cuda",
        hf_home: str | None = None,
        fallback_to_siglip: bool = True,
    ):
        """Initialize VL feature extractor.

        Args:
            model_id: HF model ID (default: PaliGemma)
            variant: "pooled" or "grid16" extraction
            device: torch device
            hf_home: HF cache directory (default: env HF_HOME)
            fallback_to_siglip: if True, fallback to SigLIP if PaliGemma unavailable
        """
        self.variant = variant
        self.device = device
        self.fallback_to_siglip = fallback_to_siglip
        self.model = None
        self.processor = None
        self.model_id = model_id
        self.model_type = "paligemma"

        # Set HF cache
        if hf_home:
            import os
            os.environ["HF_HOME"] = hf_home

        self._load_model()

    def _load_model(self):
        """Load model and processor, fallback to SigLIP if needed."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required: pip install transformers")

        # Try PaliGemma
        try:
            print(f"[VLFeatureExtractor] Loading {self.model_id}...")
            self.processor = PaliGemmaProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            self.model.eval()
            print(f"[VLFeatureExtractor] Loaded PaliGemma successfully")
            self.model_type = "paligemma"
            return
        except Exception as e:
            print(f"[VLFeatureExtractor] PaliGemma load failed: {e}")
            if not self.fallback_to_siglip:
                raise

        # Fallback to SigLIP
        print("[VLFeatureExtractor] Falling back to SigLIP...")
        try:
            from transformers import AutoImageProcessor

            siglip_id = "google/siglip-so400m-patch14-384"
            self.processor = AutoImageProcessor.from_pretrained(siglip_id)
            self.model = AutoModel.from_pretrained(
                siglip_id,
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            self.model.eval()
            print(f"[VLFeatureExtractor] Loaded SigLIP successfully")
            self.model_type = "siglip"
        except Exception as e:
            print(f"[VLFeatureExtractor] SigLIP load failed: {e}")
            raise

    def extract_vision_features(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract vision features from image.

        Args:
            image: (H, W, 3) uint8 image [0, 255]

        Returns:
            pooled: (D,) mean-pooled features
            grid16: (16, D) 4x4 spatial grid features
        """
        # Prepare image
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image)

        # Process image
        with torch.no_grad():
            if self.model_type == "paligemma":
                # Vision tower + multimodal projector only (no LLM forward needed).
                # 224px / patch14 -> 16x16 = 256 spatial tokens, projected to the
                # 2048-d Gemma embedding space.
                pixel_values = self.processor.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values.to(self.device, dtype=self.model.dtype)
                if hasattr(self.model, "get_image_features"):
                    hidden = self.model.get_image_features(pixel_values)  # (1, 256, 2048)
                else:
                    vis = self.model.vision_tower(pixel_values).last_hidden_state
                    hidden = self.model.multi_modal_projector(vis)

                D = hidden.shape[-1]
                pooled = hidden.mean(dim=1).squeeze().cpu().float()  # (D,)

                n_tokens = hidden.shape[1]
                h_tokens = int(np.sqrt(n_tokens))
                if h_tokens * h_tokens == n_tokens:
                    spatial = hidden.reshape(1, h_tokens, h_tokens, D)
                    grid = F.adaptive_avg_pool2d(
                        spatial.permute(0, 3, 1, 2),  # (1, D, 16, 16)
                        (4, 4),
                    )  # (1, D, 4, 4)
                    grid16 = grid.permute(0, 2, 3, 1).reshape(-1, D).cpu().float()  # (16, D)
                else:
                    grid16 = pooled.unsqueeze(0).repeat(16, 1)  # fallback

            else:  # siglip
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model.vision_model(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state  # (1, num_patches+1, D), e.g., (1, 577, 1152)

                D = hidden.shape[-1]
                # SigLIP includes a class token at position 0, skip it for spatial features
                pooled = hidden[:, 0, :].squeeze().cpu().float()  # Use class token for pooling

                # For grid16, reshape spatial tokens to 4x4
                if hidden.shape[1] > 1:
                    # SigLIP 384px has 577 tokens = 1 class + 576 spatial (24x24)
                    num_spatial = hidden.shape[1] - 1
                    h_tokens = int(np.sqrt(num_spatial))
                    if h_tokens == 24 and h_tokens * h_tokens == num_spatial:
                        spatial = hidden[:, 1:, :].reshape(1, h_tokens, h_tokens, D)
                        grid = F.adaptive_avg_pool2d(
                            spatial.permute(0, 3, 1, 2),  # (1, D, 24, 24)
                            (4, 4)
                        )  # (1, D, 4, 4)
                        grid16 = grid.permute(0, 2, 3, 1).reshape(-1, D).cpu().float()  # (16, D)
                    else:
                        # Fallback: use class token for all 16 positions
                        grid16 = pooled.unsqueeze(0).repeat(16, 1)
                else:
                    grid16 = pooled.unsqueeze(0).repeat(16, 1)

        return pooled.numpy(), grid16.numpy()

    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract text/instruction embedding.

        Args:
            text: instruction string (e.g., "Pick up the apple and place it in the box")

        Returns:
            (D_l,) text embedding
        """
        with torch.no_grad():
            if self.model_type == "paligemma":
                # PaliGemmaProcessor refuses text-only input; tokenize directly and
                # run just the Gemma language model, mean-pool last hidden states.
                input_ids = self.processor.tokenizer(
                    text, return_tensors="pt"
                ).input_ids.to(self.device)
                lm = self.model.language_model
                lm_out = lm(input_ids=input_ids) if not hasattr(lm, "model") else lm.model(input_ids=input_ids)
                hidden = lm_out.last_hidden_state  # (1, seq_len, 2048)
                text_feat = hidden.mean(dim=1).squeeze().cpu().float()  # (2048,)
            else:  # siglip
                # Use processor for tokenization (consistent with vision processor)
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
                inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model.text_model(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state  # (1, seq_len, D)
                # Use mean pooling over text tokens
                text_feat = hidden.mean(dim=1).squeeze().cpu().float()  # (D_l,)

        return text_feat.numpy()

    def get_video_frame(self, video_path: Path, frame_idx: int) -> np.ndarray | None:
        """Decode single frame from mp4 using PyAV (pts-based for correctness).

        Args:
            video_path: path to mp4 file
            frame_idx: target frame index

        Returns:
            (H, W, 3) uint8 image [0, 255], or None on failure
        """
        if not HAS_PYAV:
            raise ImportError("PyAV required: pip install av")

        container = None
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            # Seek by pts for correctness (counting frames after seek is wrong)
            fps = float(stream.guessed_rate or 20.0)
            target_pts = int(round(frame_idx / fps / float(stream.time_base)))
            container.seek(target_pts, stream=stream, backward=True)

            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts >= target_pts:
                    return frame.to_rgb().to_ndarray()
            return None
        except Exception as e:
            print(f"[Warning] Failed to decode {video_path} frame {frame_idx}: {e}")
            return None
        finally:
            if container is not None:
                container.close()

    def process_episode(
        self,
        parquet_path: Path,
        video_dir: Path,
        episode_number: int,
    ) -> dict[str, np.ndarray]:
        """Process one episode, extract features for all frames.

        Args:
            parquet_path: path to episode_{number:06d}.parquet
            video_dir: path to videos/chunk-000
            episode_number: episode number (for video filename)

        Returns:
            dict with keys:
              - "base": (N, D) or (N, 16, D)
              - "wrist": (N, D) or (N, 16, D)
              - "lang": (D_l,)
        """
        # Load parquet
        df = pd.read_parquet(parquet_path)
        N_frames = len(df)

        # Determine feature dims (lazy init)
        D = None
        lang_feat = None

        # Extract vision features frame by frame
        base_features = []
        wrist_features = []

        for frame_i in range(N_frames):
            # Load images
            agentview_path = (
                video_dir / "observation.images.robot0_agentview_left"
                / f"episode_{episode_number:06d}.mp4"
            )
            agentview_frame = self.get_video_frame(agentview_path, frame_i)
            if agentview_frame is None:
                agentview_frame = np.zeros((224, 224, 3), dtype=np.uint8)

            wrist_path = (
                video_dir / "observation.images.robot0_eye_in_hand"
                / f"episode_{episode_number:06d}.mp4"
            )
            wrist_frame = self.get_video_frame(wrist_path, frame_i)
            if wrist_frame is None:
                wrist_frame = np.zeros((224, 224, 3), dtype=np.uint8)

            # Resize and normalize
            agentview_frame = cv2.resize(agentview_frame, (224, 224))
            wrist_frame = cv2.resize(wrist_frame, (224, 224))
            agentview_frame = agentview_frame.astype(np.float32) / 255.0
            wrist_frame = wrist_frame.astype(np.float32) / 255.0

            # Extract features
            pooled_base, grid_base = self.extract_vision_features(agentview_frame)
            pooled_wrist, grid_wrist = self.extract_vision_features(wrist_frame)

            if D is None:
                D = pooled_base.shape[0]

            if self.variant == "pooled":
                base_features.append(pooled_base)
                wrist_features.append(pooled_wrist)
            else:  # grid16
                base_features.append(grid_base)
                wrist_features.append(grid_wrist)

            # Extract text once (instruction is constant per episode)
            if lang_feat is None and frame_i == 0:
                # Get task name from parquet metadata or use a default
                # For now, we'll handle this in the main loop by passing task name
                pass

        # Stack features
        if self.variant == "pooled":
            base_features = np.stack(base_features, axis=0)  # (N, D)
            wrist_features = np.stack(wrist_features, axis=0)  # (N, D)
        else:  # grid16
            base_features = np.stack(base_features, axis=0)  # (N, 16, D)
            wrist_features = np.stack(wrist_features, axis=0)  # (N, 16, D)

        return {
            "base": base_features.astype(np.float16),
            "wrist": wrist_features.astype(np.float16),
            "lang": lang_feat,  # To be filled by caller
        }


# Task-to-instruction mapping
TASK_INSTRUCTIONS = {
    "TurnOnElectricKettle": "Turn on the electric kettle",
    "PickPlaceCounterToCabinet": "Pick up an object from the counter and place it in the cabinet",
    "PickPlaceCounterToStove": "Pick up an object from the counter and place it on the stove",
    "SlideDishwasherRack": "Slide the dishwasher rack in or out",
    "KettleBoiling": "Make water boil in the kettle",
    "LoadDishwasher": "Load dishes into the dishwasher",
    "PrepareCoffee": "Prepare coffee using the machine",
    "PreSoakPan": "Pre-soak a pan with water",
    "WashLettuce": "Wash lettuce under running water",
}


def extract_all_features(
    root_dir: Path,
    output_dir: Path,
    variant: Literal["pooled", "grid16"] = "pooled",
    task_names: list[str] | None = None,
    device: str = "cuda",
    hf_home: str | None = None,
    skip_existing: bool = True,
) -> None:
    """Extract features for all episodes in a task.

    Args:
        root_dir: /data/group_data/maxlab/common_datasets/amagnuso/robocasa/v1.0/target
        output_dir: /data/group_data/maxlab/common_datasets/pandaliza/b0b1_features/<variant>
        variant: "pooled" or "grid16"
        task_names: list of tasks (default: all 9)
        device: torch device
        hf_home: HF cache directory
        skip_existing: skip episodes whose .npz already exists
    """
    if task_names is None:
        task_names = list(TASK_INSTRUCTIONS.keys())

    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = VLFeatureExtractor(variant=variant, device=device, hf_home=hf_home)

    # Write manifest
    manifest = {
        "variant": variant,
        "vision_dim": None,  # Will be determined after first extraction
        "num_vision_tokens": 1 if variant == "pooled" else 16,
        "lang_dim": None,
        "cameras": ["base", "wrist"],
        "dtype": "float16",
        "frame_indexing": "matches LeRobot episode frame index",
    }

    for task_name in task_names:
        print(f"\n[extract_all_features] Processing task: {task_name}")

        task_output_dir = output_dir / "features" / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Find task directory
        found = False
        for category in ["atomic", "composite"]:
            task_dir = root_dir / category / task_name
            if not task_dir.exists():
                continue
            found = True

            for date_dir in sorted(task_dir.iterdir()):
                if not date_dir.is_dir():
                    continue

                lerobot_dir = date_dir / "lerobot"
                if not lerobot_dir.exists():
                    continue

                meta_dir = lerobot_dir / "meta"
                episodes_path = meta_dir / "episodes.jsonl"
                data_dir = lerobot_dir / "data" / "chunk-000"
                video_dir = lerobot_dir / "videos" / "chunk-000"

                if not episodes_path.exists() or not data_dir.exists():
                    continue

                # Process episodes
                with open(episodes_path) as f:
                    episodes_meta = [json.loads(line) for line in f]

                for ep_idx, ep_meta in enumerate(episodes_meta):
                    ep_num = ep_meta["episode_index"]
                    ep_len = ep_meta["length"]

                    parquet_path = data_dir / f"episode_{ep_num:06d}.parquet"
                    if not parquet_path.exists():
                        continue

                    output_path = task_output_dir / f"episode_{ep_num:06d}.npz"
                    if skip_existing and output_path.exists():
                        print(f"  [skip] {task_name} episode {ep_num}")
                        continue

                    try:
                        print(f"  [process] {task_name} episode {ep_num} ({ep_len} frames)")

                        # Extract features
                        features = extractor.process_episode(
                            parquet_path=parquet_path,
                            video_dir=video_dir,
                            episode_number=ep_num,
                        )

                        # Extract text features once per task
                        if features["lang"] is None:
                            text = TASK_INSTRUCTIONS.get(task_name, task_name)
                            features["lang"] = extractor.extract_text_features(text).astype(
                                np.float16
                            )

                        # Update manifest dims if first time
                        if manifest["vision_dim"] is None:
                            base_feat = features["base"]
                            if base_feat.ndim == 2:
                                manifest["vision_dim"] = int(base_feat.shape[1])
                            else:
                                manifest["vision_dim"] = int(base_feat.shape[2])
                            manifest["lang_dim"] = int(features["lang"].shape[0])

                        # Save npz
                        np.savez_compressed(
                            output_path,
                            base=features["base"],
                            wrist=features["wrist"],
                            lang=features["lang"],
                        )
                        print(f"    Saved: {output_path}")

                    except Exception as e:
                        print(f"    ERROR: {e}")

            if found:
                break

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[extract_all_features] Manifest written to {manifest_path}")
