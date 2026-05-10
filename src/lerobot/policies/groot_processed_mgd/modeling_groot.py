#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Groot Policy Wrapper for LeRobot Integration

Minimal integration that delegates to Isaac-GR00T components where possible
without porting their code. The intent is to:

- Download and load the pretrained GR00T model via GR00TN15.from_pretrained
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via GR00T model.get_action
- Provide a training forward that can call the GR00T model forward if batch
  structure matches.

Notes:
- Dataset loading and full training orchestration is handled by Isaac-GR00T
  TrainRunner in their codebase. If you want to invoke that flow end-to-end
  from LeRobot, see `GrootPolicy.finetune_with_groot_runner` below.
"""

import builtins
import json
from contextlib import ExitStack, nullcontext
import logging
import os
from collections import deque
from pathlib import Path
from typing import TypeVar

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.groot_processed_mgd.action_head.mgd_heads import (
    ActionTargetProjector,
    SequenceMGDHead,
    TokenMask,
    masked_mean_pool,
    mgd_reconstruction_loss,
)
from lerobot.policies.groot_processed_mgd.configuration_groot import GrootMGDConfig
from lerobot.policies.groot_processed_mgd.groot_n1 import GR00TN15
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

T = TypeVar("T", bound="GrootMGDPolicy")

logger = logging.getLogger(__name__)

_GROOT_MODEL_PREFIX = "_groot_model."
_SAFETENSORS_FILENAME = "model.safetensors"


class GrootMGDPolicy(PreTrainedPolicy):
    """Wrapper around external Groot model for LeRobot integration."""

    name = "groot_processed_mgd"
    config_class = GrootMGDConfig

    def __init__(self, config: GrootMGDConfig, **kwargs):
        """Initialize Groot policy wrapper."""
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize GR00T model using ported components
        self._groot_model = self._create_groot_model()
        self._init_mgd_modules()
        self._apply_trainable_mode()
        self._log_mgd_setup()

        self.reset()

    def _init_mgd_modules(self) -> None:
        self.token_mask = TokenMask(self.config.mgd_token_mask_ratio)

        action_horizon = int(self._groot_model.action_head.action_horizon)
        action_latent_dim = int(self._groot_model.action_head.input_embedding_dim)
        backbone_dim = int(self._groot_model.action_head.config.backbone_embedding_dim)

        self.action_target_projector = ActionTargetProjector(
            action_horizon=action_horizon,
            action_latent_dim=action_latent_dim,
            target_dim=self.config.mgd_target_dim,
            pooling=self.config.mgd_target_pooling,
            projection=self.config.mgd_target_projection,
            pretrained_projector_path=self.config.mgd_pretrained_projector_path,
            trainable=self.config.mgd_backprop_action_target_projector,
        )
        self.sequence_mgd_head = SequenceMGDHead(
            in_dim=backbone_dim,
            hidden_dim=self.config.mgd_sequence_hidden_dim,
            out_dim=self.config.mgd_target_dim,
        )

    @staticmethod
    def _build_groot_inputs(batch: dict[str, Tensor], include_action: bool = True) -> dict[str, Tensor]:
        allowed_base = {"state", "state_mask", "embodiment_id"}
        if include_action:
            allowed_base = allowed_base | {"action", "action_mask"}
        return {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }

    def _create_groot_model(self):
        """Create and initialize the GR00T model using Isaac-GR00T API.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps (delegating to Isaac-GR00T):
        1) Download and load pretrained model via GR00TN15.from_pretrained
        2) Align action horizon with data_config if provided
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target=self.config.lora_target,
        )

        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        if self.config.groot_pretrained_path:
            pretrained_dir = self._resolve_groot_pretrained_dir()
            state_dict = self._load_groot_state_dict(pretrained_dir)
            first_key = next(iter(state_dict), "")
            if not first_key.startswith(_GROOT_MODEL_PREFIX) and not any(
                k.startswith("backbone.") or k.startswith("action_head.") for k in state_dict
            ):
                raise ValueError(
                    f"No compatible keys found in {pretrained_dir}. "
                    f"Expected '{_GROOT_MODEL_PREFIX}*' or GR00T bare keys."
                )

            groot_state = self._normalize_groot_state_dict(state_dict)
            missing, unexpected = model.load_state_dict(groot_state, strict=False)
            if missing:
                logger.warning(
                    "After loading groot_pretrained_path, missing keys (%d): %s ...",
                    len(missing),
                    missing[:3],
                )
            if unexpected:
                logger.warning(
                    "After loading groot_pretrained_path, unexpected keys (%d): %s ...",
                    len(unexpected),
                    unexpected[:3],
                )
            logger.info(
                "Loaded %d keys from groot_pretrained_path '%s'.",
                len(groot_state),
                self.config.groot_pretrained_path,
            )

        return model

    def _resolve_groot_pretrained_dir(self) -> Path:
        """Resolve `groot_pretrained_path` as a local dir/file or HF repo id snapshot."""
        pretrained_path = self.config.groot_pretrained_path
        if not pretrained_path:
            raise ValueError("groot_pretrained_path is empty.")

        candidate = Path(pretrained_path)
        if candidate.is_file():
            return candidate.parent

        if candidate.is_dir():
            return candidate

        try:
            downloaded = snapshot_download(
                repo_id=pretrained_path,
                repo_type="model",
                allow_patterns=[
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "model-*.safetensors",
                ],
            )
            logger.info(
                "Downloaded groot_pretrained_path from HuggingFace repo '%s' to %s",
                pretrained_path,
                downloaded,
            )
            return Path(downloaded)
        except (HFValidationError, RepositoryNotFoundError, HfHubHTTPError) as e:
            raise FileNotFoundError(
                f"groot_pretrained_path is not a local file/dir and HF download failed for repo id: {pretrained_path}"
            ) from e

    def _resolve_groot_pretrained_files(self, pretrained_dir: Path) -> list[Path]:
        """Resolve local safetensors files for a checkpoint directory."""
        direct_file = pretrained_dir / _SAFETENSORS_FILENAME
        if direct_file.exists():
            return [direct_file]

        index_file = pretrained_dir / "model.safetensors.index.json"
        if index_file.exists():
            index_data = json.loads(index_file.read_text())
            weight_map = index_data.get("weight_map", {})
            shard_names = sorted(set(weight_map.values()))
            if not shard_names:
                raise FileNotFoundError(
                    f"Checkpoint index exists but weight_map is empty: {index_file}"
                )
            shard_files = [pretrained_dir / name for name in shard_names]
            missing = [path for path in shard_files if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Checkpoint shards referenced by {index_file} are missing: {missing}"
                )
            return shard_files

        shard_files = sorted(pretrained_dir.glob("*.safetensors"))
        if shard_files:
            return shard_files

        raise FileNotFoundError(
            f"No safetensors checkpoint found in {pretrained_dir}. "
            "Expected model.safetensors, model.safetensors.index.json, or sharded *.safetensors files."
        )

    def _load_groot_state_dict(self, pretrained_dir: Path) -> dict[str, Tensor]:
        """Load and merge checkpoint state dict shards."""
        state_dict: dict[str, Tensor] = {}
        for shard_file in self._resolve_groot_pretrained_files(pretrained_dir):
            shard_state = safetensors_load_file(str(shard_file))
            state_dict.update(shard_state)
        return state_dict

    @staticmethod
    def _normalize_groot_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Strip LeRobot prefix if present and fall back to bare GR00T keys."""
        groot_state = {
            k[len(_GROOT_MODEL_PREFIX):]: v
            for k, v in state_dict.items()
            if k.startswith(_GROOT_MODEL_PREFIX)
        }
        if groot_state:
            return groot_state
        if any(k.startswith("backbone.") or k.startswith("action_head.") for k in state_dict):
            return state_dict
        return {}

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _apply_trainable_mode(self) -> None:
        mode = self.config.mgd_trainable_mode
        if mode == "default":
            return
        if mode == "head_only":
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.sequence_mgd_head.parameters():
                p.requires_grad_(True)
            return
        if mode == "lora_only":
            for p in self.parameters():
                p.requires_grad_(False)

            lora_trainable_tensors = 0
            for name, p in self._groot_model.backbone.named_parameters():
                if "lora_" in name.lower():
                    p.requires_grad_(True)
                    lora_trainable_tensors += 1

            # In lora_only mode we always keep the action target path frozen.
            self.action_target_projector.requires_grad_(False)
            for p in self.sequence_mgd_head.parameters():
                p.requires_grad_(True)

            if lora_trainable_tensors == 0:
                logger.warning(
                    "mgd_trainable_mode='lora_only' but no LoRA parameters were found. "
                    "Did you set policy.lora_rank > 0?"
                )
            return
        if mode == "processed_only":
            for p in self.parameters():
                p.requires_grad_(False)

            # Train only processed feature generation modules.
            self._groot_model.action_head.vlln.requires_grad_(True)
            self._groot_model.action_head.vl_self_attention.requires_grad_(True)
            self.sequence_mgd_head.requires_grad_(True)
            self.action_target_projector.requires_grad_(False)
            return
        if mode == "dit_only":
            # Freeze everything (LoRA weights stay frozen but still contribute to forward pass).
            for p in self.parameters():
                p.requires_grad_(False)
            # Open the entire action_head (vlln + vl_self_attention + DiT core).
            for p in self._groot_model.action_head.parameters():
                p.requires_grad_(True)
            # Keep action_encoder frozen — it was the MGD target path and is not needed for FM.
            for p in self._groot_model.action_head.action_encoder.parameters():
                p.requires_grad_(False)
            return
        raise ValueError(f"Unsupported mgd_trainable_mode: {mode!r}")

    def _log_mgd_setup(self) -> None:
        logger.info(
            "MGD setup: enabled=%s mode=%s target_pooling=%s target_projection=%s "
            "target_dim=%d seq_hidden_dim=%d token_mask_ratio=%.3f loss_w=%.4f fm_w=%.4f",
            self.config.mgd_enabled,
            self.config.mgd_trainable_mode,
            self.config.mgd_target_pooling,
            self.config.mgd_target_projection,
            self.config.mgd_target_dim,
            self.config.mgd_sequence_hidden_dim,
            self.config.mgd_token_mask_ratio,
            self.config.mgd_loss_weight,
            self.config.mgd_fm_loss_weight,
        )

        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        trainable_params = sum(p.numel() for _, p in self.named_parameters() if p.requires_grad)
        logger.info(
            "Trainable parameters: tensors=%d params=%d",
            len(trainable_names),
            trainable_params,
        )
        if trainable_names:
            preview_n = min(10, len(trainable_names))
            logger.info(
                "Trainable parameter sample (%d/%d): %s",
                preview_n,
                len(trainable_names),
                ", ".join(trainable_names[:preview_n]),
            )

    def _disable_lora_adapters(self):
        eagle_model = getattr(self._groot_model.backbone, "eagle_model", None)
        if eagle_model is None:
            return nullcontext()

        adapter_modules = []
        for module_name in ("language_model", "vision_model"):
            module = getattr(eagle_model, module_name, None)
            if module is not None and hasattr(module, "disable_adapter"):
                adapter_modules.append(module)

        if not adapter_modules:
            return nullcontext()

        stack = ExitStack()
        for module in adapter_modules:
            stack.enter_context(module.disable_adapter())
        return stack

    def _compute_pooled_vlm_feature(self, groot_inputs: dict[str, Tensor], disable_lora: bool) -> Tensor:
        device = next(self.parameters()).device
        adapter_ctx = self._disable_lora_adapters() if disable_lora else nullcontext()
        with adapter_ctx:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
                backbone_inputs, _ = self._groot_model.prepare_input(groot_inputs)
                backbone_outputs = self._groot_model.backbone(backbone_inputs)
                self._groot_model.action_head.set_frozen_modules_to_eval_mode()
                backbone_outputs = self._groot_model.action_head.process_backbone_output(backbone_outputs)

            backbone_features = backbone_outputs["backbone_features"]
            backbone_mask = backbone_outputs.get("backbone_attention_mask")
            return masked_mean_pool(backbone_features.float(), backbone_mask)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: GrootMGDConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load Groot policy from pretrained model.

        Handles two cases:
        1. Base GR00T models (e.g., 'nvidia/GR00T-N1.5-3B') - loads the raw model
        2. Fine-tuned LeRobot checkpoints - loads config and weights from safetensors

        Args:
            pretrained_name_or_path: Path to the GR00T model or fine-tuned checkpoint
            config: Optional GrootMGDConfig. If None, loads from checkpoint or creates default
            force_download: Force download even if cached
            resume_download: Resume interrupted download
            proxies: Proxy settings
            token: HuggingFace authentication token
            cache_dir: Cache directory path
            local_files_only: Only use local files
            revision: Specific model revision
            strict: Strict state dict loading
            **kwargs: Additional arguments (passed to config)

        Returns:
            Initialized GrootPolicy instance with loaded model
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        from huggingface_hub.errors import HfHubHTTPError

        print(
            "The Groot policy is a wrapper around Nvidia's GR00T N1.5 model.\n"
            f"Loading pretrained model from: {pretrained_name_or_path}"
        )

        model_id = str(pretrained_name_or_path)
        is_finetuned_checkpoint = False

        # Check if this is a fine-tuned LeRobot checkpoint (has model.safetensors)
        try:
            if os.path.isdir(model_id):
                is_finetuned_checkpoint = os.path.exists(os.path.join(model_id, SAFETENSORS_SINGLE_FILE))
            else:
                # Try to download the safetensors file to check if it exists
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=SAFETENSORS_SINGLE_FILE,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=False,  # Just check, don't force download
                        proxies=proxies,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    is_finetuned_checkpoint = True
                except HfHubHTTPError:
                    is_finetuned_checkpoint = False
        except Exception:
            is_finetuned_checkpoint = False

        if is_finetuned_checkpoint:
            print("Detected fine-tuned LeRobot checkpoint, loading with state dict...")

            # Load config if not provided
            if config is None:
                from lerobot.configs.policies import PreTrainedConfig

                config = PreTrainedConfig.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    revision=revision,
                    **kwargs,
                )

            # Create policy instance (loads base GR00T model)
            instance = cls(config, **kwargs)

            # Load safetensors and auto-detect key format
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            from safetensors import safe_open

            with safe_open(model_file, framework="pt") as f:
                first_key = next(iter(f.keys()), "")

            # Isaac-GR00T checkpoints use bare keys (e.g. "action_head.*").
            # LeRobot checkpoints use "_groot_model." prefix.
            # Remap Isaac-GR00T keys so they match the wrapped model.
            if not first_key.startswith("_groot_model."):
                print("Isaac-GR00T key format detected — remapping keys with '_groot_model.' prefix")
                import tempfile

                import torch
                from safetensors.torch import load_file, save_file

                raw_sd = load_file(model_file)
                remapped = {"_groot_model." + k: v for k, v in raw_sd.items()}
                with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
                    tmp_path = tmp.name
                save_file(remapped, tmp_path)
                model_file = tmp_path

            policy = cls._load_as_safetensor(instance, model_file, config.device, strict=False)
            policy.to(config.device)
            policy.eval()
            return policy

        # This is a base GR00T model - load it fresh
        print("Detected base GR00T model, loading from HuggingFace...")

        if config is None:
            # Create default config with the pretrained path
            config = GrootMGDConfig(base_model_path=str(pretrained_name_or_path))

            # Add minimal visual feature required for validation
            # validate_features() will automatically add state and action features
            # These are placeholders - actual robot features come from the preprocessor
            if not config.input_features:
                config.input_features = {
                    f"{OBS_IMAGES}.camera": PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, 224, 224),  # Default image size from config
                    ),
                }
        else:
            # Override the base_model_path with the provided path
            config.base_model_path = str(pretrained_name_or_path)

        # Pass through any additional config overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create a fresh policy instance - this will automatically load the GR00T model
        # in __init__ via _create_groot_model()
        policy = cls(config)

        policy.eval()
        return policy

    def get_optim_params(self) -> dict:
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Delegates to Isaac-GR00T model.forward when inputs are compatible.
        """
        groot_inputs = self._build_groot_inputs(batch, include_action=True)
        compute_vlm_drift = bool(batch.get("compute_vlm_drift", False)) and self.config.vlm_drift_logging_enabled

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs, return_intermediate=True)

        fm_loss = outputs["loss"]

        backbone_features = outputs["backbone_features"]  # (B, T_seq, D_vlm)
        backbone_mask = outputs.get("backbone_attention_mask")  # (B, T_seq)
        z_v_current = masked_mean_pool(backbone_features.float(), backbone_mask)
        z_v_current_for_logging = z_v_current.detach()

        if not self.config.mgd_enabled:
            loss_dict = {
                "loss": fm_loss.item(),
                "flow_matching_loss": fm_loss.item(),
                "mgd_loss": 0.0,
                "mgd_token_mask_ratio_cfg": float(self.token_mask.mask_ratio),
            }
            if compute_vlm_drift:
                with torch.no_grad():
                    z_v_base = self._compute_pooled_vlm_feature(groot_inputs, disable_lora=True).detach()
                loss_dict["vlm_drift_cos"] = F.cosine_similarity(
                    z_v_base, z_v_current_for_logging, dim=-1
                ).mean().item()
                loss_dict["vlm_drift_l2"] = torch.norm(z_v_current_for_logging - z_v_base, dim=-1).mean().item()
            return fm_loss, loss_dict

        z_v_tokens = backbone_features.float()
        if not self.config.mgd_backprop_backbone:
            z_v_tokens = z_v_tokens.detach()

        if backbone_mask is None:
            valid_token_mask = torch.ones(
                z_v_tokens.shape[0],
                z_v_tokens.shape[1],
                device=z_v_tokens.device,
                dtype=torch.bool,
            )
        else:
            valid_token_mask = backbone_mask.bool()

        z_v_masked_tokens, kept_token_mask, token_stats = self.token_mask(z_v_tokens, valid_token_mask)
        z_a_hat_raw = self.sequence_mgd_head(
            z_v_masked_tokens,
            valid_token_mask=valid_token_mask,
            kept_token_mask=kept_token_mask,
        )
        z_a_hat = F.normalize(z_a_hat_raw, dim=-1)

        action_traj = batch["action"].to(device=device, dtype=torch.float32)
        B = action_traj.shape[0]
        embodiment_id = batch["embodiment_id"].to(device=device, dtype=torch.long)
        t_clean = torch.full((B,), 999, dtype=torch.long, device=device)

        with torch.no_grad():
            action_enc_out = self._groot_model.action_head.action_encoder(
                action_traj,
                t_clean,
                embodiment_id,
            )

        if self.config.mgd_backprop_action_target_projector:
            z_a_target_raw = self.action_target_projector(action_enc_out.float())
            z_a_target = F.normalize(z_a_target_raw, dim=-1)
            z_a_target_for_loss = z_a_target
        else:
            with torch.no_grad():
                z_a_target_raw = self.action_target_projector(action_enc_out.float())
                z_a_target = F.normalize(z_a_target_raw, dim=-1)
            z_a_target_for_loss = z_a_target.detach()

        if self.config.mgd_use_cosine_loss and self.config.mgd_use_mse_loss:
            loss_mgd = (
                mgd_reconstruction_loss(z_a_hat, z_a_target_for_loss, kind="cosine")
                + mgd_reconstruction_loss(z_a_hat, z_a_target_for_loss, kind="mse")
            )
        elif self.config.mgd_use_mse_loss:
            loss_mgd = mgd_reconstruction_loss(z_a_hat, z_a_target_for_loss, kind="mse")
        else:
            loss_mgd = mgd_reconstruction_loss(z_a_hat, z_a_target_for_loss, kind="cosine")

        if self.config.mgd_preserve_weight > 0:
            # Placeholder: preserve loss branch is intentionally skipped by default.
            loss_preserve = fm_loss.new_zeros(())
        else:
            loss_preserve = fm_loss.new_zeros(())

        total_loss = (
            fm_loss * self.config.mgd_fm_loss_weight
            + loss_mgd * self.config.mgd_loss_weight
            + loss_preserve * self.config.mgd_preserve_weight
        )

        loss_dict = {
            "loss": total_loss.item(),
            "flow_matching_loss": fm_loss.item(),
            "mgd_loss": loss_mgd.item(),
            "mgd_token_mask_ratio_cfg": float(self.token_mask.mask_ratio),
            "valid_token_count": token_stats["valid_token_count"].mean().item(),
            "kept_token_count": token_stats["kept_token_count"].mean().item(),
            "actual_token_mask_ratio": token_stats["actual_token_mask_ratio"].mean().item(),
            "mgd_target_norm_raw": z_a_target_raw.norm(dim=-1).mean().item(),
            "mgd_pred_norm_raw": z_a_hat_raw.norm(dim=-1).mean().item(),
            "mgd_target_norm_post": z_a_target.norm(dim=-1).mean().item(),
            "mgd_pred_norm_post": z_a_hat.norm(dim=-1).mean().item(),
            "mgd_cos_sim": F.cosine_similarity(z_a_hat, z_a_target, dim=-1).mean().item(),
        }

        if compute_vlm_drift:
            with torch.no_grad():
                z_v_base = self._compute_pooled_vlm_feature(groot_inputs, disable_lora=True).detach()
            z_v_current = z_v_tokens.detach().float()
            if valid_token_mask is not None:
                mask_f = valid_token_mask.unsqueeze(-1).to(dtype=z_v_current.dtype)
                z_v_current = (z_v_current * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            else:
                z_v_current = z_v_current.mean(dim=1)
            loss_dict["vlm_drift_cos"] = F.cosine_similarity(z_v_base, z_v_current, dim=-1).mean().item()
            loss_dict["vlm_drift_l2"] = torch.norm(z_v_current - z_v_base, dim=-1).mean().item()

        return total_loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions for inference by delegating to Isaac-GR00T.

        Returns a tensor of shape (B, n_action_steps, action_dim).
        """
        self.eval()

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        groot_inputs = self._build_groot_inputs(batch, include_action=False)

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.

        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """

        # Set environment variables to handle Flash Attention compatibility
        # These help with symbol resolution issues
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn

            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")


# Backward-compat alias
GrootPolicy = GrootMGDPolicy
