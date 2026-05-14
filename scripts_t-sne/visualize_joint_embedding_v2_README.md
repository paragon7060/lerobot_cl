# visualize_joint_embedding_v2.py

action_encoder 임베딩과 VLM backbone 임베딩을 동일 N개 샘플에 대해 나란히 t-SNE로 시각화하는 인터랙티브 HTML 도구.

**v1 대비 추가된 기능:**
- 두 점 클릭으로 PCA-50 공간에서의 거리(cosine/L2) 계산 및 표시
- A/B 비디오 동시 재생
- `--no_zscore` 플래그로 z-score 정규화 없이 PCA → t-SNE 진행 가능

---

## 실행 커맨드

### 기본 (delta_mean, zscore 포함)
```bash
conda run -n groot python src/lerobot/scripts/visualize_joint_embedding_v2.py \
    --cache_dir /home/seonho/ws3/outputs/action_emb_vis_2048 \
    --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
    --output_dir /home/seonho/ws3/outputs/action_emb_vis_2048/joint_v2
```

### flatten + z-score (action encoder 전체 시간 구조 보존)
```bash
conda run -n groot python src/lerobot/scripts/visualize_joint_embedding_v2.py \
    --cache_dir /home/seonho/ws3/outputs/action_emb_vis_2048 \
    --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
    --action_agg flatten \
    --output_dir /home/seonho/ws3/outputs/action_emb_vis_2048/joint_v2_flatten
```

### flatten + no_zscore (joint 간 분산 스케일 보존)
```bash
conda run -n groot python src/lerobot/scripts/visualize_joint_embedding_v2.py \
    --cache_dir /home/seonho/ws3/outputs/action_emb_vis_2048 \
    --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
    --action_agg flatten \
    --no_zscore \
    --output_dir /home/seonho/ws3/outputs/action_emb_vis_2048/joint_v2_flatten_noz
```

### raw_flatten (원본 GT action 궤적, joint space)
```bash
conda run -n groot python src/lerobot/scripts/visualize_joint_embedding_v2.py \
    --cache_dir /home/seonho/ws3/outputs/action_emb_vis_2048 \
    --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
    --action_agg raw_flatten \
    --output_dir /home/seonho/ws3/outputs/action_emb_vis_2048/joint_v2_raw_flatten
```

### clips 없이 빠른 HTML만 재생성 (--skip_clips)
```bash
conda run -n groot python src/lerobot/scripts/visualize_joint_embedding_v2.py \
    --cache_dir /home/seonho/ws3/outputs/action_emb_vis_2048 \
    --action_agg flatten \
    --output_dir /home/seonho/ws3/outputs/action_emb_vis_2048/joint_v2_flatten \
    --skip_clips
```

---

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--cache_dir` | (필수) | action_encoder_cache.npz, dit_features_cache.npz가 있는 디렉토리 |
| `--dataset_root` | None | 비디오 클립 추출용 데이터셋 경로. None이면 클립 생략 |
| `--output_dir` | `<cache_dir>/joint_v2` | 출력 디렉토리 |
| `--action_agg` | `delta_mean` | action feature 집계 방법 (아래 표 참고) |
| `--backbone_agg` | `mask_mean` | backbone feature 집계 방법 |
| `--perplexity` | 30 | t-SNE perplexity |
| `--no_zscore` | False | z-score 정규화 생략 (joint 간 분산 차이 보존) |
| `--skip_clips` | False | 비디오 클립 추출 건너뜀 |
| `--cam_keys` | right_shoulder, wrist | 클립 추출 카메라 |
| `--clip_size` | `320x240` | 클립 해상도 |
| `--action_horizon` | 16 | 클립 프레임 수 |

---

## action_agg 방법 비교

| 방법 | 입력 | 출력 차원 | 특징 |
|------|------|-----------|------|
| `delta_mean` | encoder (N,16,1536) | (N,1536) | 평균 변화량. 반대 방향 동작 구분 어려울 수 있음 |
| `flatten` | encoder (N,16,1536) | (N,24576) | 시간 구조 완전 보존. 방향 구분 가능 |
| `first_mid_last` | encoder (N,16,1536) | (N,4608) | 시작/중간/끝 keyframe |
| `mean` | encoder (N,16,1536) | (N,1536) | 전체 평균 |
| `raw_flatten` | gt_actions (N,16,32) | (N,512) | 원본 joint 궤적 그대로 |
| `raw_delta_mean` | gt_actions (N,16,32) | (N,32) | joint space 평균 속도 |
| `raw_delta_total` | gt_actions (N,16,32) | (N,32) | joint space 총 변위 |

---

## z-score 유무 비교

| | z-score 적용 (기본) | `--no_zscore` |
|--|-------------------|---------------|
| **처리** | 각 차원을 평균 0, 분산 1로 정규화 후 PCA | 분산 0인 열만 제거, 정규화 없이 PCA |
| **의미** | 모든 feature 차원을 동등하게 취급 | 분산이 큰 차원(joint)이 PCA 주성분을 지배 |
| **적합한 경우** | feature 차원 간 스케일이 임의적일 때 | joint 간 분산 차이 자체가 의미 있는 신호일 때 |
| **CW/CCW 구분** | 분산 작은 rotation joint 희석 가능 | rotation joint의 분산이 상대적으로 크면 주성분에 반영 |

---

## HTML 사용법

1. **첫 번째 클릭** → 파란 링(A) 표시 + 왼쪽 비디오 재생 + A 정보 표시
2. **두 번째 클릭** → 초록 링(B) 표시 + 오른쪽 비디오 재생 + 거리 계산
   - Action PCA-50 공간: cosine distance, L2 distance
   - Backbone PCA-50 공간: cosine distance, L2 distance
3. **같은 점 클릭** → 초기화
4. 다시 첫 번째 클릭부터 반복

### 거리 해석 가이드

- **Cosine distance ≈ 0**: 방향이 거의 동일 (같은 task의 비슷한 phase)
- **Cosine distance ≈ 1**: 방향이 직교
- **Cosine distance ≈ 2**: 반대 방향 (CW vs CCW, push vs pull 후보)
- **L2 distance**: 크기 차이도 반영. cosine은 같지만 L2가 크면 → magnitude만 다른 동작

CL training positive/negative pair 선정 기준:
- **positive**: cosine distance < 0.3 (같은 task, 비슷한 phase)
- **hard negative**: cosine distance 0.5~1.0 (비슷해 보이지만 다른 task)
- **easy negative**: cosine distance > 1.5 (명확히 다른 category)

---

## 출력 파일

```
<output_dir>/
├── joint_tsne_p30.html           (기본)
├── joint_tsne_p30_noz.html       (--no_zscore 시)
├── joint_tsne_p30_combined.png   정적 PNG (task/category 4-panel)
└── clips/ → joint/clips/         비디오 클립 (symlink 또는 직접 생성)
    ├── right_shoulder/{0000..N}.mp4
    └── wrist/{0000..N}.mp4
```

---

## clips 재활용

`joint/clips/`가 이미 존재하면 `clips/` symlink를 자동으로 생성합니다.
같은 `cache_dir`에서 다른 `action_agg`로 실행하면 클립 재추출 없이 HTML만 재생성됩니다.

캐시를 새로 추출(`visualize_action_embedding_tsne.py`)하면 `joint/clips/`와의 sample 순서가 달라지므로,
반드시 `joint/clips/`를 삭제하고 `joint/`부터 재실행해야 합니다.
