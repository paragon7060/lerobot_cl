#!/usr/bin/env python

import logging
import time
import torch
import tqdm
from termcolor import colored
from pprint import pformat

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
import lerobot.policies.act.configuration_act
import lerobot.policies.diffusion.configuration_diffusion
import lerobot.policies.vqbet.configuration_vqbet
from lerobot.datasets.factory import make_dataset
from lerobot.utils.utils import init_logging, format_big_number

@parser.wrap()
def check_dataset(cfg: TrainPipelineConfig):
    init_logging()
    logging.info(colored("Starting Dataset Integrity Check...", "cyan", attrs=["bold"]))

    # 1. Dataset 생성
    try:
        logging.info(f"Loading dataset with repo_id: {cfg.dataset.repo_id}")
        dataset = make_dataset(cfg)
        logging.info(colored("Dataset loaded successfully.", "green"))
    except Exception as e:
        logging.error(colored(f"Failed to load dataset: {e}", "red"))
        raise e

    # 2. 메타데이터 출력
    logging.info(colored("Dataset Information:", "yellow"))
    logging.info(f"  - Root Directory: {cfg.dataset.root}")
    logging.info(f"  - Num Episodes: {dataset.num_episodes}")
    logging.info(f"  - Num Frames: {format_big_number(dataset.num_frames)}")
    logging.info(f"  - Features: {list(dataset.features.keys())}")
    logging.info(f"  - FPS: {dataset.fps}")

    # 3. DataLoader 생성
    logging.info("Creating DataLoader...")
    # 검증을 위해 shuffle=False로 설정하여 순차적으로 읽습니다.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False, 
        drop_last=False,
    )

    # 4. 배치 반복 테스트
    logging.info(f"Starting iteration check over {len(dataloader)} batches...")
    start_time = time.time()
    
    try:
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc="Checking batches")):
            # 첫 번째 배치의 정보만 자세히 출력하여 데이터 형태 확인
            if i == 0:
                logging.info(colored("First batch structure:", "blue"))
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        logging.info(f"  - {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.2f}, {value.max():.2f}]")
                    elif isinstance(value, list):
                        logging.info(f"  - {key}: list of length {len(value)} (e.g. {value[0]})")
                    else:
                        logging.info(f"  - {key}: {type(value)}")
            
            # 간단한 데이터 무결성 체크 (NaN / Inf 체크)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    if torch.isnan(value).any():
                        raise ValueError(f"NaN values detected in batch {i}, key '{key}'")
                    if torch.isinf(value).any():
                        raise ValueError(f"Inf values detected in batch {i}, key '{key}'")

    except Exception as e:
        logging.error(colored(f"Error occurred during iteration at batch {i}: {e}", "red"))
        raise e

    duration = time.time() - start_time
    logging.info(colored(f"Successfully iterated over all batches in {duration:.2f} seconds.", "green", attrs=["bold"]))
    logging.info(colored("Dataset check passed! No errors found.", "green", attrs=["bold"]))

if __name__ == "__main__":
    check_dataset()
