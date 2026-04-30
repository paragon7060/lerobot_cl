#!/usr/bin/env python
"""GR00T Keyframe 데이터셋 검증 및 시각화 스크립트.

에피소드별로 4개 카메라(guide | right_shoulder | wrist | keyframe_slot)를 시각화하고
keyframe 시점 전후 동작을 검증한다. 추가로:
  - Action/state feature-dim 뒤쪽이 실제로 zero-padded인지 sanity check.
  - --prompt_mode {guide, non_guide} 로 description 매핑 전환 확인.

실행 방법:
  conda activate lerobot050_groot
  cd /home/seonho/clvla/lerobot_cl/src

  # guide 모드 + slice sanity check
  python lerobot/policies/groot_keyframe/verify_dataset.py \
      --episode 1819 \
      --save_dir /tmp/verify_guide \
      --prompt_mode guide \
      --dataset_root /mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3

  # non_guide 모드
  python lerobot/policies/groot_keyframe/verify_dataset.py \
      --episode 1819 \
      --save_dir /tmp/verify_nonguide \
      --prompt_mode non_guide \
      --dataset_root /mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3

  # 제외된 에피소드 확인 (registry에 cropped=true 없음)
  python lerobot/policies/groot_keyframe/verify_dataset.py --episode 0
  # → "제외된 에피소드" 출력 후 종료
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

_SCRIPT_DIR = Path(__file__).parent
DEFAULT_REGISTRY_PATH = Path(
    "/home/seonho/clvla/memory_module/INSIGHTfixposV3_Keyframe_right_ver/frame_index_registry.json"
)
TASK_DESCRIPTIONS_PATHS: dict[str, Path] = {
    "guide": _SCRIPT_DIR / "task_descriptions.json",
    "non_guide": _SCRIPT_DIR / "task_descriptions_non_guide.json",
}
DATASET_REPO_ID = "paragon7060/INSIGHTfixposV3"
# INSIGHTfixposV3: action stored as [32] with real 8 dims (j1-j7, gripper),
# state [32] with real 16 dims (ee7 + joint7 + gripper2).
DEFAULT_ACTION_DIM = 8
DEFAULT_STATE_DIM = 16


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def _remap_file_path(file_path: str, registry_path: Path) -> str:
    """JSON에 기록된 절대 경로를 실제 registry 위치 기준으로 치환한다."""
    marker = "keyframe_output/"
    idx = file_path.find(marker)
    if idx == -1:
        return file_path
    rel = file_path[idx + len(marker):]
    actual_base = registry_path.parent
    return str(actual_base / rel)


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """(C, H, W) float32 [0,1] 텐서 → PIL RGB 이미지."""
    arr = (img_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def _add_red_border(img: Image.Image, border: int = 6) -> Image.Image:
    inner_w = max(1, img.width - 2 * border)
    inner_h = max(1, img.height - 2 * border)
    inner = img.resize((inner_w, inner_h), Image.BILINEAR)
    bordered = Image.new("RGB", img.size, (220, 30, 30))
    bordered.paste(inner, (border, border))
    return bordered


def _add_text_overlay(img: Image.Image, text: str, position: str = "top") -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(0, (img.width - tw) // 2)
    y = 4 if position == "top" else img.height - th - 8

    draw.rectangle([x - 2, y - 2, x + tw + 2, y + th + 2], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(255, 255, 0), font=font)
    return img


def _add_description_bar(img: Image.Image, description: str, bar_height: int = 30) -> Image.Image:
    bar = Image.new("RGB", (img.width, bar_height), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    max_chars = img.width // 7
    display_text = description if len(description) <= max_chars else description[: max_chars - 3] + "..."

    bbox = draw.textbbox((0, 0), display_text, font=font)
    tx = max(4, (img.width - (bbox[2] - bbox[0])) // 2)
    ty = (bar_height - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), display_text, fill=(255, 255, 255), font=font)

    combined = Image.new("RGB", (img.width, img.height + bar_height))
    combined.paste(img, (0, 0))
    combined.paste(bar, (0, img.height))
    return combined


def _add_status_bar(
    img: Image.Image,
    text: str,
    ok: bool,
    bar_height: int = 24,
) -> Image.Image:
    bg = (20, 90, 40) if ok else (140, 30, 30)
    bar = Image.new("RGB", (img.width, bar_height), bg)
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    max_chars = img.width // 7
    display_text = text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    bbox = draw.textbbox((0, 0), display_text, font=font)
    tx = max(4, (img.width - (bbox[2] - bbox[0])) // 2)
    ty = (bar_height - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), display_text, fill=(255, 255, 255), font=font)

    combined = Image.new("RGB", (img.width, img.height + bar_height))
    combined.paste(img, (0, 0))
    combined.paste(bar, (0, img.height))
    return combined


def build_composite(
    guide_img: Image.Image,
    rightshoulder_img: Image.Image,
    wrist_img: Image.Image,
    keyframe_slot_img: Image.Image,
    is_keyframe: bool,
    description: str,
    frame_index: int,
    keyframe_frame_index: int,
    prompt_mode: str,
    padding_status_text: str | None = None,
    padding_ok: bool = True,
    per_frame_padding_text: str | None = None,
    per_frame_padding_ok: bool = True,
) -> Image.Image:
    """4개 카메라를 가로로 나열한 합성 이미지를 반환한다."""
    target_h = 224
    target_w = 224

    def resize_pil(img: Image.Image) -> Image.Image:
        return img.resize((target_w, target_h), Image.BILINEAR)

    imgs = [
        resize_pil(guide_img),
        resize_pil(rightshoulder_img),
        resize_pil(wrist_img),
        resize_pil(keyframe_slot_img),
    ]
    labels = ["guide", "right_shoulder", "wrist", "keyframe"]

    panels = []
    for i, (img, label) in enumerate(zip(imgs, labels)):
        if i == 3 and is_keyframe:
            img = _add_red_border(img, border=6)
            img = _add_text_overlay(img, "KEYFRAME", position="top")
        img = _add_text_overlay(
            img,
            f"{label} | fr={frame_index}" if i != 3 else f"kf_slot (kf@{keyframe_frame_index})",
            position="bottom",
        )
        panels.append(img)

    total_w = sum(p.width for p in panels)
    composite_h = max(p.height for p in panels)
    composite = Image.new("RGB", (total_w, composite_h), (20, 20, 20))
    x = 0
    for panel in panels:
        composite.paste(panel, (x, 0))
        x += panel.width

    composite = _add_description_bar(composite, f"[MODE={prompt_mode}] {description}")
    if per_frame_padding_text:
        composite = _add_status_bar(composite, per_frame_padding_text, per_frame_padding_ok)
    if padding_status_text:
        composite = _add_status_bar(composite, padding_status_text, padding_ok)
    return composite


# ---------------------------------------------------------------------------
# Action/state zero-padding sanity check
# ---------------------------------------------------------------------------

def verify_feature_dim_padding(
    dataset,
    action_dim: int,
    state_dim: int,
    max_frames_to_check: int = 200,
) -> tuple[str, bool]:
    """Action/state 뒤쪽 dim이 정말 전부 0인지 확인하고 로그. 요약 문자열과 ok 플래그 반환."""
    n_check = min(len(dataset), max_frames_to_check)
    print(f"\n=== Zero-padding sanity check (first {n_check} frames) ===")

    action_tail_max = 0.0
    state_tail_max = 0.0
    action_full_shape = None
    state_full_shape = None

    for idx in range(n_check):
        item = dataset[idx]
        act = item["action"]                # (T, D_act) or (D_act,)
        st = item["observation.state"]      # (D_state,)
        if action_full_shape is None:
            action_full_shape = tuple(act.shape)
            state_full_shape = tuple(st.shape)

        act_tail = act[..., action_dim:]
        st_tail = st[..., state_dim:]
        action_tail_max = max(action_tail_max, act_tail.abs().max().item() if act_tail.numel() else 0.0)
        state_tail_max = max(state_tail_max, st_tail.abs().max().item() if st_tail.numel() else 0.0)

    print(f"  action shape (raw):          {action_full_shape}  → slice to [..., :{action_dim}]")
    print(f"  observation.state shape:     {state_full_shape}  → slice to [..., :{state_dim}]")
    print(f"  max |action[..., {action_dim}:]|  across checked frames = {action_tail_max:.6e}")
    print(f"  max |state[..., {state_dim}:]|  across checked frames = {state_tail_max:.6e}")

    ok = action_tail_max < 1e-6 and state_tail_max < 1e-6
    act_full = action_full_shape[-1] if action_full_shape else "?"
    st_full = state_full_shape[-1] if state_full_shape else "?"
    mark_a = "OK" if action_tail_max < 1e-6 else "FAIL"
    mark_s = "OK" if state_tail_max < 1e-6 else "FAIL"
    summary = (
        f"PAD[{n_check}f] act[{action_dim}:{act_full}] {mark_a} max={action_tail_max:.1e} | "
        f"st[{state_dim}:{st_full}] {mark_s} max={state_tail_max:.1e}"
    )

    if ok:
        print(f"  ✓ Zero-padded tail confirmed: action dims {action_dim}-? and state dims {state_dim}-? are all ~0.")
    else:
        print(f"  ✗ WARNING: tail dims are NOT all zero — slicing will DROP real information.")
    print("=" * 60)

    return summary, ok


def per_frame_padding_status(
    item: dict,
    action_dim: int,
    state_dim: int,
) -> tuple[str, bool]:
    """현재 프레임의 action/state tail이 실제로 0인지 확인."""
    act = item["action"]
    st = item["observation.state"]
    act_tail = act[..., action_dim:]
    st_tail = st[..., state_dim:]
    a_max = act_tail.abs().max().item() if act_tail.numel() else 0.0
    s_max = st_tail.abs().max().item() if st_tail.numel() else 0.0
    ok = a_max < 1e-6 and s_max < 1e-6
    mark_a = "OK" if a_max < 1e-6 else "FAIL"
    mark_s = "OK" if s_max < 1e-6 else "FAIL"
    text = (
        f"FRAME act[{action_dim}:] {mark_a} max={a_max:.1e} | "
        f"st[{state_dim}:] {mark_s} max={s_max:.1e}"
    )
    return text, ok


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="groot_keyframe 데이터셋 에피소드별 시각화 + slice/prompt 검증 도구"
    )
    parser.add_argument("--episode", type=int, required=True, help="시각화할 에피소드 ID")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="타임스텝별 이미지 저장 디렉토리 (미지정 시 matplotlib 표시)",
    )
    parser.add_argument(
        "--registry_path",
        type=str,
        default=str(DEFAULT_REGISTRY_PATH),
        help="frame_index_registry.json 경로",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="guide",
        choices=list(TASK_DESCRIPTIONS_PATHS.keys()),
        help="guide=기존 구체 회전방향, non_guide='following the guide' 패턴",
    )
    parser.add_argument(
        "--descriptions_path",
        type=str,
        default=None,
        help="task_descriptions.json 경로 (미지정 시 --prompt_mode 기준 자동 선택)",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=DEFAULT_ACTION_DIM,
        help="실제 유효 action dim (학습 시 슬라이스 대상)",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=DEFAULT_STATE_DIM,
        help="실제 유효 state dim (학습 시 슬라이스 대상)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DATASET_REPO_ID,
        help="LeRobot 데이터셋 repo ID",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="로컬 데이터셋 루트 경로 (None이면 HF 캐시 사용)",
    )
    args = parser.parse_args()

    episode_id: int = args.episode
    save_dir: Path | None = Path(args.save_dir) if args.save_dir else None
    registry_path = Path(args.registry_path)

    descriptions_path = (
        Path(args.descriptions_path)
        if args.descriptions_path
        else TASK_DESCRIPTIONS_PATHS[args.prompt_mode]
    )
    print(f"Prompt mode: {args.prompt_mode} (descriptions: {descriptions_path})")

    # ------------------------------------------------------------------
    # Registry 로드 + 에피소드 검증
    # ------------------------------------------------------------------
    with open(registry_path) as f:
        registry_entries = json.load(f)

    ep_entries_cropped = [
        e for e in registry_entries if e["episode_id"] == episode_id and e["cropped"]
    ]
    if not ep_entries_cropped:
        print("제외된 에피소드")
        sys.exit(0)

    kf_entry = ep_entries_cropped[0]
    kf_frame_index: int = kf_entry["frame_index"]
    kf_file_path: str = _remap_file_path(kf_entry["file_path"], registry_path)
    m = re.search(r"task_(\w+)[/\\]", kf_entry["file_path"])
    task_id = m.group(1) if m else ""
    pre_kf_cam = "right_shoulder" if task_id.startswith("3") else "wrist"
    print(
        f"에피소드 {episode_id} | keyframe: frame={kf_frame_index}, "
        f"pre_kf_cam={pre_kf_cam}, file={kf_file_path}"
    )

    # ------------------------------------------------------------------
    # Task description 매핑 + Dataset 로드
    # ------------------------------------------------------------------
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_meta = LeRobotDatasetMetadata(
        repo_id=args.repo_id,
        root=args.dataset_root,
    )
    dataset_root: Path = ds_meta.root

    with open(descriptions_path) as f:
        name_to_desc: dict = json.load(f)

    tasks_parquet = dataset_root / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_parquet).reset_index()
    task_name_to_desc: dict[str, str] = {
        row["index"]: name_to_desc.get(row["index"], row["index"])
        for _, row in tasks_df.iterrows()
    }

    # ------------------------------------------------------------------
    # Dataset 로드 (해당 에피소드만)
    # ------------------------------------------------------------------
    print(f"Dataset 로드 중 (episode={episode_id}, root={dataset_root})...")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=[episode_id],
        video_backend="pyav",
    )
    print(f"총 {len(dataset)}개 프레임 로드됨.")

    # Zero-padding sanity check
    padding_summary, padding_ok = verify_feature_dim_padding(
        dataset, args.action_dim, args.state_dim
    )

    # Keyframe 이미지 로드 (공유)
    kf_pil_cached: Image.Image | None = None

    # ------------------------------------------------------------------
    # 저장 디렉토리 준비
    # ------------------------------------------------------------------
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"이미지 저장 경로: {save_dir}")
    else:
        import matplotlib.pyplot as plt
        plt.ion()

    # ------------------------------------------------------------------
    # 타임스텝 순회
    # ------------------------------------------------------------------
    for idx in range(len(dataset)):
        item = dataset[idx]

        frame_index: int = item["frame_index"].item()

        guide_t = item["observation.images.guide"]
        rightshoulder_t = item["observation.images.right_shoulder"]
        wrist_t = item["observation.images.wrist"]

        guide_pil = _tensor_to_pil(guide_t)
        rightshoulder_pil = _tensor_to_pil(rightshoulder_t)
        wrist_pil = _tensor_to_pil(wrist_t)

        is_keyframe_frame = frame_index == kf_frame_index
        if frame_index < kf_frame_index:
            kf_slot_pil = (rightshoulder_pil if pre_kf_cam == "right_shoulder" else wrist_pil).copy()
        else:
            if kf_pil_cached is None:
                raw = Image.open(kf_file_path).convert("RGB")
                kf_pil_cached = raw.resize((224, 224), Image.BILINEAR)
            kf_slot_pil = kf_pil_cached.copy()

        task_name: str = item["task"]
        description: str = task_name_to_desc.get(task_name, task_name)

        per_frame_text, per_frame_ok = per_frame_padding_status(
            item, args.action_dim, args.state_dim
        )

        composite = build_composite(
            guide_img=guide_pil,
            rightshoulder_img=rightshoulder_pil,
            wrist_img=wrist_pil,
            keyframe_slot_img=kf_slot_pil,
            is_keyframe=is_keyframe_frame,
            description=description,
            frame_index=frame_index,
            keyframe_frame_index=kf_frame_index,
            prompt_mode=args.prompt_mode,
            padding_status_text=padding_summary,
            padding_ok=padding_ok,
            per_frame_padding_text=per_frame_text,
            per_frame_padding_ok=per_frame_ok,
        )

        if save_dir:
            out_path = save_dir / f"frame_{frame_index:04d}.png"
            composite.save(out_path)
            if frame_index % 10 == 0 or is_keyframe_frame:
                marker = " ← KEYFRAME" if is_keyframe_frame else ""
                print(f"  저장: {out_path}{marker}")
        else:
            import matplotlib.pyplot as plt

            plt.clf()
            plt.imshow(np.array(composite))
            plt.title(
                f"episode={episode_id} | frame={frame_index}"
                + (" [KEYFRAME]" if is_keyframe_frame else ""),
                color="red" if is_keyframe_frame else "black",
                fontsize=11,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.pause(0.05)

    if save_dir:
        print(f"\n완료. {len(dataset)}개 프레임 → {save_dir}")
    else:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()
        print(f"\n완료. episode={episode_id}, {len(dataset)}개 프레임 시각화.")


if __name__ == "__main__":
    main()