import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Sampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset # LeRobotDataset 임포트
from typing import List, Iterator

class ProportionalTaskSampler(Sampler[int]):
    """
    사용자가 지정한 비율에 따라 태스크를 샘플링하여 데이터 분포를 제어하는 샘플러 (수정된 버전).
    """
    def __init__(self,
                 dataset: LeRobotDataset,
                 valid_episode_indices: List[int],
                 target_proportions: dict,
                 epoch_size: int,
                 shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size

        # 목표 비율 정규화
        total_prop = sum(target_proportions.values())
        if not np.isclose(total_prop, 1.0):
            print(f"⚠️ Warning: Target proportions sum to {total_prop}, not 1.0. Normalizing them.")
            target_proportions = {k: v / total_prop for k, v in target_proportions.items()}

        TASK_TO_OBJECT_MAP = {
            '1ext': 'cabinet', '3a': 'door', '3b': 'door', '3c': 'door', '3d': 'door',
            '5a': 'bottle', '5b': 'bottle', '5c': 'bottle', '5d': 'bottle',
            '5e': 'bottle', '5f': 'bottle', '5g': 'bottle', '5h': 'bottle'
        }
        object_to_subtasks = defaultdict(lambda: defaultdict(list))
        self.episode_to_frames = {}

        for ep_idx in valid_episode_indices:
            
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [수정된 부분] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # 1. 'episode_data_index' 대신 'dataset.meta.episodes[ep_idx]' 사용
            #    'LeRobotDataset'는 'meta' (LeRobotDatasetMetadata) 객체를 통해 에피소드 정보를 관리합니다.
            episode_meta = dataset.meta.episodes[ep_idx]
            
            # 2. 'dataset_from_index'와 'dataset_to_index' 키를 사용
            #    (이전의 'from', 'to' 키가 이름이 변경되었습니다.)
            start_frame = episode_meta["dataset_from_index"]
            end_frame = episode_meta["dataset_to_index"]

            # 3. 'hf_dataset'에서 프레임 데이터를 가져와 'task_index' 추출
            frame_data = dataset.hf_dataset[start_frame]
            task_index = frame_data['task_index'].item()
            
            # 4. 'task_index' (정수)를 'task_name' (문자열)으로 변환
            #    dataset.meta.tasks는 인덱스가 task 이름인 DataFrame입니다.
            #    정수 인덱스로 task 이름을 찾으려면 .iloc[...].name을 사용해야 합니다.
            task_name = dataset.meta.tasks.iloc[task_index].name
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [수정 완료] ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            object_name = TASK_TO_OBJECT_MAP.get(task_name)
            if object_name:
                object_to_subtasks[object_name][task_name].append(ep_idx)
            
            self.episode_to_frames[ep_idx] = list(range(start_frame, end_frame))

        # (이하 로직은 동일)
        balanced_episodes_per_object = defaultdict(list)
        for object_name, subtasks in object_to_subtasks.items():
            if not subtasks: continue
            max_subtask_episodes = max(len(ep_list) for ep_list in subtasks.values())
            balanced_list = []
            for _, ep_list in subtasks.items():
                if not ep_list: continue # 에피소드 리스트가 비어있으면 건너뛰기
                repeat_factor = max_subtask_episodes // len(ep_list)
                balanced_list.extend(ep_list * repeat_factor)
                balanced_list.extend(random.sample(ep_list, max_subtask_episodes % len(ep_list)))
            balanced_episodes_per_object[object_name] = balanced_list
            
        self.balanced_epoch_indices = []
        for object_name, proportion in target_proportions.items():
            num_episodes_to_sample = int(self.epoch_size * proportion)
            source_episodes = balanced_episodes_per_object.get(object_name)
            if not source_episodes:
                print(f"  - ⚠️ Warning: No episodes found for object '{object_name}'. Skipping.")
                continue
            sampled_episodes = random.choices(source_episodes, k=num_episodes_to_sample)
            self.balanced_epoch_indices.extend(sampled_episodes)
        
        self.total_frames = sum(len(self.episode_to_frames[ep_idx]) for ep_idx in self.balanced_epoch_indices)
        print(f"\n✅ Proportional sampling setup complete. Epoch size: {len(self.balanced_epoch_indices)} episodes, Total frames per epoch: {self.total_frames}")

    def __iter__(self) -> Iterator[int]:
        indices = self.balanced_epoch_indices
        if self.shuffle:
            random.shuffle(indices)
        
        frame_indices = []
        for ep_idx in indices:
            frame_indices.extend(self.episode_to_frames[ep_idx])
            
        # 프레임 레벨에서도 셔플을 적용 (선택 사항)
        if self.shuffle:
            random.shuffle(frame_indices)
            
        yield from frame_indices

    def __len__(self) -> int:
        return self.total_frames