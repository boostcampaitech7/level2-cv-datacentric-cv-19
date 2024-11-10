import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
import numpy as np  # NumPy 임포트 추가
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect

# 상수 정의
CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # 기존 인자들
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'),
                        help='데이터 디렉토리 경로')
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'),
                        help='모델 체크포인트 디렉토리 경로')
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'),
                        help='출력 디렉토리 경로')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu',
                        help='사용할 디바이스 (cuda 또는 cpu)')
    parser.add_argument('--input_size', type=int, default=2048,
                        help='모델 입력 크기 (32의 배수)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='배치 크기')

    # 추가된 인자: 데이터셋 분할 (train/test)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='데이터셋 분할 선택: train 또는 test')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def load_ground_truth(data_dir, split='train'):
    """
    Ground Truth 어노테이션을 로드합니다.
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        split (str): 데이터셋 분할 ('train' 또는 'test')
    
    Returns:
        dict: 이미지 파일 이름을 키로 하고, 해당 이미지의 어노테이션을 값으로 하는 딕셔너리
    """
    ground_truth = {}
    # 수정된 경로: data_dir/{lang}_receipt/ufo/{split}.json
    for lang in LANGUAGE_LIST:
        ann_fpath = osp.join(data_dir, f'{lang}_receipt', 'ufo', f'{split}.json')
        if osp.exists(ann_fpath):
            with open(ann_fpath, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
                ground_truth.update(ann_data['images'])
        else:
            print(f'Warning: Annotation file not found for {lang} in split {split} at {ann_fpath}')
    return ground_truth

def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test', device='cpu'):
    """
    모델을 사용하여 주어진 데이터셋에 대해 추론을 수행합니다.
    
    Args:
        model (torch.nn.Module): 학습된 모델
        ckpt_fpath (str): 체크포인트 파일 경로
        data_dir (str): 데이터 디렉토리 경로
        input_size (int): 모델 입력 크기
        batch_size (int): 배치 크기
        split (str): 데이터셋 분할 ('train' 또는 'test')
        device (str): 사용 디바이스 ('cuda' 또는 'cpu')
    
    Returns:
        dict: 예측 결과와 ground truth가 포함된 딕셔너리
    """
    # 체크포인트 로드
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.to(device)
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    images = []

    # 데이터 로드 및 배치 처리
    image_paths = sum([glob(osp.join(data_dir, f'{lang}_receipt', 'img', split, '*')) for lang in LANGUAGE_LIST], [])
    for image_fpath in tqdm(image_paths, desc=f'Processing {split} data'):
        image_fname = osp.basename(image_fpath)
        image_fnames.append(image_fname)
        image = cv2.imread(image_fpath)
        if image is None:
            print(f'Warning: Unable to read image {image_fpath}')
            continue
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if len(images) == batch_size:
            bboxes = detect(model, images, input_size)
            by_sample_bboxes.extend(bboxes)
            images = []

    # 남은 이미지 처리
    if len(images):
        bboxes = detect(model, images, input_size)
        by_sample_bboxes.extend(bboxes)

    # 예측 결과 저장
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    # Ground Truth 로드
    ground_truth = load_ground_truth(data_dir, split=split)

    # 결과에 Ground Truth 추가
    for image_fname in image_fnames:
        if image_fname in ground_truth:
            ufo_result['images'][image_fname]['ground_truth'] = ground_truth[image_fname]['words']
        else:
            print(f'Warning: Ground truth not found for image {image_fname}')

    return ufo_result

def visualize_predictions(data_dir, output_dir, split, results):
    """
    예측된 bounding box와 ground truth를 이미지에 시각화하여 저장합니다.
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        output_dir (str): 시각화된 이미지 저장 디렉토리 경로
        split (str): 데이터셋 분할 ('train' 또는 'test')
        results (dict): 예측 및 ground truth 결과
    """
    for image_fname, data in tqdm(results['images'].items(), desc='Visualizing predictions'):
        image_fpath = None
        for lang in LANGUAGE_LIST:
            potential_path = osp.join(data_dir, f'{lang}_receipt', 'img', split, image_fname)
            if osp.exists(potential_path):
                image_fpath = potential_path
                break
        if image_fpath is None:
            print(f'Warning: Image file not found for visualization {image_fname}')
            continue

        image = cv2.imread(image_fpath)
        if image is None:
            print(f'Warning: Unable to read image for visualization {image_fpath}')
            continue

        # 예측된 bounding box 그리기 (파란색)
        for bbox in data.get('words', {}).values():
            points = bbox['points']
            try:
                pts = np.array(points).reshape(-1, 1, 2).astype(int)  # 리스트를 NumPy 배열로 변환
                cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            except Exception as e:
                print(f'Error drawing predicted bbox for {image_fname}: {e}')

        # Ground Truth bounding box 그리기 (녹색)
        for gt_bbox in data.get('ground_truth', {}).values():
            gt_points = gt_bbox['points']
            try:
                gt_pts = np.array(gt_points).reshape(-1, 1, 2).astype(int)  # 리스트를 NumPy 배열로 변환
                cv2.polylines(image, [gt_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            except Exception as e:
                print(f'Error drawing ground truth bbox for {image_fname}: {e}')

        # 결과 이미지 저장
        output_image_fpath = osp.join(output_dir, f'visual_{image_fname}')
        cv2.imwrite(output_image_fpath, image)

def main(args):
    # 모델 초기화
    model = EAST(pretrained=False).to(args.device)

    # 체크포인트 파일 경로
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')
    if not osp.exists(ckpt_fpath):
        raise FileNotFoundError(f'Checkpoint file not found: {ckpt_fpath}')

    # 출력 디렉토리 생성
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Inference 수행
    print(f'Inference in progress on {args.split} dataset')
    ufo_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                              args.batch_size, split=args.split, device=args.device)

    # 결과 JSON 저장
    output_fname = f'output_{args.split}.json'
    with open(osp.join(args.output_dir, output_fname), 'w', encoding='utf-8') as f:
        json.dump(ufo_result, f, indent=4, ensure_ascii=False)

    print(f'Inference results saved to {osp.join(args.output_dir, output_fname)}')

    # 시각화 수행
    visualize_output_dir = osp.join(args.output_dir, 'visualizations')
    if not osp.exists(visualize_output_dir):
        os.makedirs(visualize_output_dir)

    print('Starting visualization of predictions and ground truth')
    visualize_predictions(args.data_dir, visualize_output_dir, args.split, ufo_result)
    print(f'Visualization images are saved in {visualize_output_dir}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
