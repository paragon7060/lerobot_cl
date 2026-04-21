# Modified LeRobot Library Files

기본 LeRobot 라이브러리에서 수정된 부분을 기록합니다.
수정 이유와 원본 코드, 변경 내용을 포함합니다.

---

## 1. `src/lerobot/datasets/utils.py`

### 수정 위치
`get_safe_version()` 함수 (line ~288)

### 원인
`huggingface_hub` 최신 버전에서 `RevisionNotFoundError.__init__()`이 `response` keyword-only 인자를 필수로 요구하도록 변경됨. 기존 코드가 `raise RevisionNotFoundError(message)` 형태로 호출하여 `TypeError` 발생.

또한, 버전 태그가 없는 HuggingFace 데이터셋(예: `Whalswp/robocasa_merged_pretrain`)을 사용할 때 학습이 불가능한 문제가 있었음.

### 원본 코드
```python
if not hub_versions:
    raise RevisionNotFoundError(
        f"""Your dataset must be tagged with a codebase version.
        Assuming _version_ is the codebase_version value in the info.json, you can run this:
        ```python
        from huggingface_hub import HfApi

        hub_api = HfApi()
        hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
        ```
        """
    )
```

### 변경 코드
```python
if not hub_versions:
    logging.warning(
        f"Dataset '{repo_id}' has no version tags on the Hub. Falling back to 'main' revision. "
        f"To pin a version, run: "
        f"HfApi().create_tag('{repo_id}', tag='{target_version}', repo_type='dataset')"
    )
    return "main"
```

### 효과
- 버전 태그가 없는 데이터셋에서도 `main` 브랜치로 폴백하여 학습 가능
- `huggingface_hub` 버전 호환성 문제 해결

---

## 2. `src/lerobot/datasets/video_utils.py`

### 수정 위치 1: `decode_video_frames()` 라우팅 변경 (line ~149)

#### 원인
`pyav` 백엔드가 `decode_video_frames_torchvision()`을 경유하도록 되어 있었는데, `torchvision.io.VideoReader`가 일부 H264 영상 파일에서 `avcodec_send_packet()` 오류로 크래시 발생.

#### 원본 코드
```python
elif backend in ["pyav", "video_reader"]:
    return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
```

#### 변경 코드
```python
elif backend == "pyav":
    return decode_video_frames_pyav(video_path, timestamps, tolerance_s)
elif backend == "video_reader":
    return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
```

---

### 수정 위치 2: `decode_video_frames_pyav()` 함수 추가 (line ~161)

#### 원인
torchvision의 `VideoReader`를 우회하여 PyAV를 직접 사용하는 디코더 경로 추가. `torchcodec`도 해당 환경에서 FFmpeg 공유 라이브러리(`libavutil.so`) 버전 불일치로 동작하지 않으므로 안정적인 대안 필요.

#### 추가된 코드
```python
def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    """Decode video frames using PyAV directly (without torchvision)."""
    import av

    video_path = str(video_path)
    query_ts = torch.tensor(timestamps)

    container = av.open(video_path)
    stream = container.streams.video[0]

    first_ts = min(timestamps)
    last_ts = max(timestamps)

    seek_pts = int(first_ts / stream.time_base)
    container.seek(seek_pts, backward=True, any_frame=False, stream=stream)

    loaded_frames = []
    loaded_ts = []

    for frame in container.decode(stream):
        ts = float(frame.pts * stream.time_base)
        if ts < first_ts - tolerance_s:
            continue
        img = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
        loaded_frames.append(img)
        loaded_ts.append(ts)
        if ts >= last_ts:
            break

    container.close()

    if not loaded_frames:
        raise RuntimeError(f"No frames decoded from {video_path} for timestamps {timestamps}")

    loaded_ts_t = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(...)

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    return closest_frames
```

#### 효과
- torchvision 의존 없이 PyAV 직접 사용으로 H264/AV1 등 다양한 코덱 안정적 디코딩
- `torchcodec` 환경 문제와 무관하게 동작

---

## 3. `src/lerobot/policies/groot/action_head/flow_matching_action_head.py`

### 수정 위치: `FlowmatchingActionHead.forward()` — per-joint weighted FM loss

#### 원인
GR00T-CL v2 Phase 1에서 wrist joint (index 6)에 3배 가중치를 주어
action encoder가 wrist 상태를 더 잘 구분하도록 학습하기 위함.

#### 원본 코드
```python
def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
    ...
    action_mask = action_input.action_mask
    loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    loss = loss.sum() / action_mask.sum()
```

#### 변경 코드
```python
def forward(self, backbone_output: BatchFeature, action_input: BatchFeature, joint_weights=None) -> BatchFeature:
    ...
    action_mask = action_input.action_mask
    if joint_weights is None:
        weight = action_mask.float()
    else:
        # joint_weights: (max_action_dim,) float tensor. padding dims = 0.0
        weight = joint_weights.to(pred_actions.device).view(1, 1, -1).expand_as(pred_actions)
    loss = F.mse_loss(pred_actions, velocity, reduction="none") * weight
    loss = loss.sum() / weight.sum().clamp(min=1e-8)
```

#### 효과
- `joint_weights=None`일 때 기존 동작과 완전 동일 (하위 호환)
- padding 차원(index 8~31)은 `joint_weights=0.0`으로 자동 제외
- 기존 `action_mask` 역할을 `joint_weights`가 흡수하여 중복 적용 없음

---

## 4. `src/lerobot/policies/groot_cl/groot_n1.py`

### 수정 위치: `GR00TN15.forward()` — joint_weights 파라미터 전달

#### 원본 코드
```python
def forward(self, inputs: dict, return_intermediate: bool = False) -> BatchFeature:
    ...
    action_head_outputs = self.action_head(backbone_outputs, action_inputs)
```

#### 변경 코드
```python
def forward(self, inputs: dict, return_intermediate: bool = False, joint_weights=None) -> BatchFeature:
    ...
    action_head_outputs = self.action_head(backbone_outputs, action_inputs, joint_weights=joint_weights)
```

#### 효과
- `joint_weights=None`이면 기존 동작 완전 유지
- `GrootCLv2Policy._forward_phase1()`에서 `joint_weights=self._joint_weights` 전달

---

## 환경 참고

- `torchcodec` 비동작 이유: 설치된 FFmpeg 7.1.1(`libavutil.so.59`)와 torchcodec이 지원하는 버전(`.so.56~.60`) ABI 불일치
- `torchvision VideoReader` 비동작 이유: 일부 H264 영상에서 `seek` 후 `avcodec_send_packet()` 실패
- 권장 video_backend: `pyav` (`--dataset.video_backend=pyav`)
