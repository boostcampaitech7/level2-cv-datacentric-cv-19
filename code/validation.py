import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
import random
import json

import torch
from torch import cuda
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

def parse_args():
    parser = ArgumentParser()

    # 기존 인자들
    parser.add_argument('--model_dir', type=str, default='trained_models', help='학습된 모델을 저장할 디렉토리')
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu', help='학습에 사용할 디바이스')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader의 작업자 스레드 수')
    parser.add_argument('--image_size', type=int, default=2048, help='이미지 크기 조정할 크기')
    parser.add_argument('--input_size', type=int, default=1024, help='이미지 크롭할 크기')
    parser.add_argument('--batch_size', type=int, default=8, help='학습 배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='초기 학습률')
    parser.add_argument('--max_epoch', type=int, default=400, help='최대 학습 에포크 수')
    parser.add_argument('--save_interval', type=int, default=10, help='모델 체크포인트를 저장할 에포크 간격')
    parser.add_argument('--val_split', type=float, default=0.2, help='검증에 사용할 데이터셋 비율')

    # Early Stopping 인자 추가
    parser.add_argument('--early_stop', action='store_true', help='Early Stopping 활성화')
    parser.add_argument('--patience', type=int, default=10, help='개선이 없을 때 중단하기 전 기다릴 에포크 수')
    parser.add_argument('--min_delta', type=float, default=0.0, help='개선으로 간주하기 위한 최소 변화량')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size`는 32의 배수여야 합니다.')

    return args

def create_datasets(root_dirs, image_size, input_size):
    dataset = SceneTextDataset(
        root_dirs=root_dirs,
        split='train',
        image_size=image_size,
        crop_size=input_size
    )
    dataset = EASTDataset(dataset)
    return dataset

def do_training(root_dirs, model_dir, device, image_size, input_size, num_workers, batch_size,
               learning_rate, max_epoch, save_interval, val_split, early_stop, patience, min_delta):
    # 재현성을 위한 시드 설정
    torch.manual_seed(42)
    random.seed(42)

    # 데이터셋 생성
    combined_dataset = create_datasets(root_dirs, image_size, input_size)
    total_size = len(combined_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # 학습 및 검증 세트로 분할
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    print(f'전체 데이터셋 크기: {total_size}')
    print(f'학습 세트 크기: {train_size}')
    print(f'검증 세트 크기: {val_size}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch//2, eta_min=1e-6)  # eta_min을 0에서 1e-6으로 변경

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_flag = False

    for epoch in range(max_epoch):
        if early_stop_flag:
            print("Early Stopping이 발동되었습니다. 학습을 중단합니다.")
            break

        # 학습 단계
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=len(train_loader), desc=f'에포크 {epoch+1}/{max_epoch}', unit='batch') as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                img = img.to(device)
                gt_score_map = gt_score_map.to(device)
                gt_geo_map = gt_geo_map.to(device)
                roi_mask = roi_mask.to(device)

                optimizer.zero_grad()
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                # 프로그레스 바 업데이트
                pbar.set_postfix({
                    'Cls loss': extra_info['cls_loss'],
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                })
                pbar.update(1)

        scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader)
        elapsed_time = timedelta(seconds=time.time() - epoch_start)
        print(f'에포크 {epoch+1} | 학습 손실: {avg_train_loss:.4f} | 소요 시간: {elapsed_time}')

        # 검증 단계
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc='검증', unit='batch') as pbar_val:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    img = img.to(device)
                    gt_score_map = gt_score_map.to(device)
                    gt_geo_map = gt_geo_map.to(device)
                    roi_mask = roi_mask.to(device)

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    loss_val = loss.item()
                    val_loss += loss_val

                    pbar_val.update(1)

        avg_val_loss = val_loss / len(val_loader)
        print(f'에포크 {epoch+1} | 검증 손실: {avg_val_loss:.4f}')

        # Early Stopping 로직
        if early_stop:
            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # 최적 모델 저장
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)
                best_ckpt_fpath = osp.join(model_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_ckpt_fpath)
                print(f'최적 모델이 저장되었습니다. 검증 손실: {best_val_loss:.4f}')
            else:
                epochs_no_improve += 1
                print(f'검증 손실이 {epochs_no_improve} 에포크 동안 개선되지 않았습니다.')

                if epochs_no_improve >= patience:
                    early_stop_flag = True
                    # 학습 중단 전에 최적 모델 로드 (선택 사항)
                    if osp.exists(osp.join(model_dir, 'best_model.pth')):
                        model.load_state_dict(torch.load(osp.join(model_dir, 'best_model.pth')))
                        print("최적 모델 가중치를 로드했습니다.")

        # 주기적인 체크포인트 저장
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            latest_ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), latest_ckpt_fpath)
            print(f'{epoch+1} 에포크에서 체크포인트가 저장되었습니다.')

    print('학습이 완료되었습니다.')

def main(args):
    # 데이터 디렉토리 경로를 코드 내에 직접 지정
    root_dirs = [
        '/data/ephemeral/home/code/data/',
        '/data/ephemeral/home/code/data/',
        '/data/ephemeral/home/code/data/',
        '/data/ephemeral/home/code/data/'
    ]

    do_training(
        root_dirs=root_dirs,
        model_dir=args.model_dir,
        device=args.device,
        image_size=args.image_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epoch=args.max_epoch,
        save_interval=args.save_interval,
        val_split=args.val_split,
        early_stop=args.early_stop,
        patience=args.patience,
        min_delta=args.min_delta
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
