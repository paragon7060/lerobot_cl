import logging
import os

import torch
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor

from lerobot.policies.groot_cl.action_head.contrastive_heads import (
    ActionContrastiveHead,
    ContrastiveHeadConfig,
    VLMContrastiveHead,
    info_nce_fallback,
    triplet_contrastive_loss,
)
from lerobot.policies.groot_cl.configuration_groot_cl import GrootCLConfig
from lerobot.policies.groot_cl.groot_n1 import GR00TN15
from lerobot.policies.groot_cl.modeling_groot import GrootPolicy

logger = logging.getLogger(__name__)

_GROOT_MODEL_PREFIX = "_groot_model."
_SAFETENSORS_FILENAME = "model.safetensors"


class GrootCLPolicy(GrootPolicy):
    name = "groot_cl"
    config_class = GrootCLConfig

    def __init__(self, config: GrootCLConfig, **kwargs):
        super().__init__(config, **kwargs)

        contrastive_cfg = ContrastiveHeadConfig(
            latent_dim=config.contrastive_latent_dim,
            vlm_input_dim=1536,
            action_input_dim=config.max_action_dim,
            cnn_hidden_dim=config.contrastive_cnn_hidden_dim,
            proj_hidden_dim=config.contrastive_proj_hidden_dim,
            triplet_margin=config.contrastive_triplet_margin,
        )
        self.vlm_contrastive_head = VLMContrastiveHead(contrastive_cfg)
        self.action_contrastive_head = ActionContrastiveHead(contrastive_cfg)

        self.set_contrastive_phase(config.contrastive_phase)

    def _create_groot_model(self):
        """GR00TN15를 생성하고, groot_pretrained_path가 설정된 경우
        NVIDIA 원본 weight 대신 사전학습된 GrootPolicy 체크포인트의
        _groot_model.* weights를 로드한다.
        """
        self._handle_flash_attention_compatibility()

        model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
        )
        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        if self.config.groot_pretrained_path:
            safetensors_path = os.path.join(self.config.groot_pretrained_path, _SAFETENSORS_FILENAME)
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(
                    f"groot_pretrained_path가 설정되었으나 "
                    f"{safetensors_path} 파일을 찾을 수 없습니다."
                )

            full_state = safetensors_load_file(safetensors_path)
            groot_state = {
                k[len(_GROOT_MODEL_PREFIX):]: v
                for k, v in full_state.items()
                if k.startswith(_GROOT_MODEL_PREFIX)
            }

            if not groot_state:
                raise ValueError(
                    f"{safetensors_path} 에서 '{_GROOT_MODEL_PREFIX}*' 키를 찾을 수 없습니다. "
                    "GrootPolicy로 저장된 체크포인트인지 확인하세요."
                )

            missing, unexpected = model.load_state_dict(groot_state, strict=False)
            if missing:
                logger.warning("groot_pretrained_path 로드 후 missing keys: %s", missing)
            if unexpected:
                logger.warning("groot_pretrained_path 로드 후 unexpected keys: %s", unexpected)
            logger.info(
                "groot_pretrained_path '%s' 에서 _groot_model weights 로드 완료 (%d keys).",
                self.config.groot_pretrained_path,
                len(groot_state),
            )

        return model

    def set_contrastive_phase(self, phase: str) -> None:
        if phase == "phase1":
            self._groot_model.requires_grad_(False)
            self.vlm_contrastive_head.requires_grad_(True)
            self.action_contrastive_head.requires_grad_(True)
        elif phase == "phase2a":
            self._restore_groot_trainability()
            self.vlm_contrastive_head.requires_grad_(True)
            self.action_contrastive_head.requires_grad_(True)
        elif phase == "phase2b":
            self._restore_groot_trainability()
            self.vlm_contrastive_head.requires_grad_(False)
            self.action_contrastive_head.requires_grad_(False)
        else:
            raise ValueError(f"Unknown contrastive_phase: {phase!r}")
        self.config.contrastive_phase = phase

    def _restore_groot_trainability(self) -> None:
        cfg = self.config
        self._groot_model.backbone.set_trainable_parameters(
            tune_visual=cfg.tune_visual,
            tune_llm=cfg.tune_llm,
        )
        self._groot_model.action_head.set_trainable_parameters(
            tune_projector=cfg.tune_projector,
            tune_diffusion_model=cfg.tune_diffusion_model,
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_"))
            and not (k.startswith("next.") or k == "info")
        }

        device = next(self.parameters()).device
        use_contrastive = self.training

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(
                groot_inputs,
                return_intermediate=use_contrastive,
            )

        loss_fm = outputs.get("loss")
        loss_dict = {"flow_matching_loss": loss_fm.item()}

        if use_contrastive:
            backbone_features = outputs.get("backbone_features")
            attn_mask = outputs.get("backbone_attention_mask")
            actions = groot_inputs.get("action")
            negative_action = batch.get("negative_action")

            if not self.config.contrastive_backprop_backbone:
                backbone_features = backbone_features.detach()

            vlm_z = self.vlm_contrastive_head(backbone_features, attn_mask)
            pos_action_z = self.action_contrastive_head(actions)

            if negative_action is not None:
                neg_action_z = self.action_contrastive_head(negative_action)
                loss_cont = triplet_contrastive_loss(
                    vlm_z,
                    pos_action_z,
                    neg_action_z,
                    margin=self.config.contrastive_triplet_margin,
                )
            elif self.config.contrastive_fallback_to_in_batch:
                loss_cont = info_nce_fallback(vlm_z, pos_action_z)
            else:
                loss_cont = loss_fm.new_tensor(0.0)

            loss_total = loss_fm + self.config.contrastive_loss_weight * loss_cont
            loss_dict["contrastive_loss"] = loss_cont.item()
            loss_dict["loss"] = loss_total.item()
            return loss_total, loss_dict

        loss_dict["loss"] = loss_fm.item()
        return loss_fm, loss_dict
