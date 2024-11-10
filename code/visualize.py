import os
import json
from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw

def read_json(filename: str):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann


nation_dict = {
    'vi': 'vietnamese_receipt',
    'th': 'thai_receipt',
    'zh': 'chinese_receipt',
    'ja': 'japanese_receipt',
}

from PIL import Image, ImageDraw, ImageOps

def save_vis_to_img(save_dir: str | os.PathLike, inference_dir: str | os.PathLike = '/data/ephemeral/home/code/ensemble_results/merged_result.csv') -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)    
    data = read_json(inference_dir)
    for im, points in data['images'].items():
        # change to 'train' for train dataset 
        im_path = Path('data') / nation_dict[im.split('.')[1]] / 'img' / 'test' / im
        
        # EXIF 회전 정보를 적용하여 이미지 로드
        img = Image.open(im_path).convert("RGB")
        img = ImageOps.exif_transpose(img)  # EXIF 회전 정보 적용
        draw = ImageDraw.Draw(img)
        
        for obj_k, obj_v in points['words'].items():
            # bbox points
            pts = [(int(p[0]), int(p[1])) for p in obj_v['points']]
            pt1 = sorted(pts, key=lambda x: (x[1], x[0]))[0]

            # 바운딩 박스 그리기 (두께 추가)
            draw.line(pts + [pts[0]], fill=(255, 0, 0), width=3) 
                       
            draw.text(
                (pt1[0]-3, pt1[1]-12),
                obj_k,
                fill=(0, 0, 0)
            )
        img.save(os.path.join(save_dir, im))

save_vis_to_img(save_dir='visualized_output')
