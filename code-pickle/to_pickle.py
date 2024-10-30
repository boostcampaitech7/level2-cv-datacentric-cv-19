import os
import os.path as osp
import pickle
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
import albumentations as A

def main():
    # 사용자 환경에 맞게 기본 경로 설정
    data_dir = './data'  # 이미지와 JSON 데이터가 포함된 기본 데이터 경로로 수정
    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']
    
    # 사용자 정의 데이터 증강 설정
    custom_augmentation_dict = {
        'CJ': A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        'GB': A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        'B': A.Blur(blur_limit=7, p=0.5),
        'GN': A.GaussNoise(p=0.5),
        'HSV': A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        'RBC': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        'N': A.Normalize(mean=(0.776, 0.772, 0.767), std=(0.171, 0.179, 0.187), p=1.0)
    }
    
    # 이미지 및 크롭 크기, 증강 설정
    image_sizes = [1024, 1536, 2048]
    crop_size = 1024
    aug_select = ['CJ', 'GB', 'HSV', 'N']
    
    # fold 설정 (기본값: 0)
    fold = 0
    
    # 선택된 증강만으로 albumentations 설정
    custom_augmentation = [custom_augmentation_dict[s] for s in aug_select]

    # pickle 저장 디렉토리 설정
    pkl_dir = f'pickle_data/is{image_sizes}_cs{crop_size}_aug{aug_select}/train/'
    os.makedirs(osp.join(data_dir, pkl_dir), exist_ok=True)
    
    for i_size in image_sizes:
        if crop_size > i_size:
            continue  # crop_size가 image_size보다 큰 경우 무시
        # SceneTextDataset과 EASTDataset 생성
        train_dataset = SceneTextDataset(
            data_dir,
            split='train',
            image_size=2048,
            crop_size=2048,
        )
        train_dataset = EASTDataset(train_dataset)

        # 각 데이터셋 항목을 pickle로 저장
        for idx in tqdm(range(len(train_dataset)), desc=f"Processing size {i_size}"):
            data = train_dataset[idx]
            pickle_path = osp.join(data_dir, pkl_dir, f"{i_size}_{crop_size}_{idx}.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(data, f)

if __name__ == '__main__':
    main()
