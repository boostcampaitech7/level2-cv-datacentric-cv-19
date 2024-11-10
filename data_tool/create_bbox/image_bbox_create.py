import os
import random
import json
from PIL import Image, ImageOps

# 이미지 폴더 경로와 JSON 파일 저장 경로 설정
image_folder = ''
output_json_path = 'bounding_boxes.json'

# 이미지의 bounding box 추출 함수
def get_bounding_box(image):
    width, height = image.size
    # 좌상단에서 시계 방향으로 bounding box 정의
    bounding_box = [
        (0, 0),          # 좌상단
        (width - 1, 0),  # 우상단
        (width - 1, height - 1), # 우하단
        (0, height - 1)  # 좌하단
    ]
    return bounding_box

# 이미지 패딩 및 bounding box 수정 함수
def pad_image_and_update_bbox(image, bbox):
    width, height = image.size
    padding_top = random.randint(15, 100)     # 위쪽 패딩 높이
    padding_bottom = random.randint(15, 100)  # 아래쪽 패딩 높이
    new_height = height + padding_top + padding_bottom

    # 위아래로 패딩 추가
    padded_image = ImageOps.expand(image, (0, padding_top, 0, padding_bottom), fill=(255, 255, 255))

    # 패딩 후 bounding box 수정
    new_bbox = [
        (x, y + padding_top) for x, y in bbox
    ]
    
    return padded_image, new_bbox

# 이미지 처리 및 bounding box 수집
bounding_boxes_data = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)

        # Step 1: bounding box 추출
        bbox = get_bounding_box(image)

        # Step 2: 이미지 패딩 및 bounding box 수정
        padded_image, new_bbox = pad_image_and_update_bbox(image, bbox)

        # Step 3: 패딩된 이미지 저장 (필요 시)
        padded_image.save(os.path.join(image_folder, f"{filename}"))

        # Step 4: bounding box 데이터 저장
        bounding_boxes_data.append({
            "image_name": f"{filename}",
            "bounding_box": new_bbox
        })

# JSON 파일로 bounding box 데이터 저장
with open(output_json_path, 'w') as f:
    json.dump(bounding_boxes_data, f, indent=4)

print(f"Bounding box 정보가 '{output_json_path}'에 저장되었습니다.")
