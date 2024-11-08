import os
import random
import json
from PIL import Image, ImageOps
from datetime import datetime

image_folder = ''
output_json_path = 'bounding_boxes.json'

# 설정
image_folder = ''  # 이미지 폴더 경로
output_folder = ''  # 결과 이미지 저장 폴더 경로
output_json = 'bounding_boxes.json'  # 메타데이터 JSON 파일 이름
min_padding = 50  # 최소 패딩 간격
max_padding = 200  # 최대 패딩 간격
num_images = 100  # 생성할 최종 이미지 개수
image_width = 500  # 최종 이미지의 폭 (고정)
license_tag = {
    "usability": True,
    "public": False,
    "commercial": True,
    "type": None,
    "holder": "Upstage"
}

# 현재 날짜 가져오기
today = datetime.now().strftime("%Y-%m-%d")

# 메타데이터 저장을 위한 딕셔너리
metadata_dict = {"images": {}}

# 파일 목록 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# output_folder가 존재하지 않으면 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 생성
for i in range(num_images):
    # n개의 랜덤 이미지 선택
    selected_images = random.sample(image_files, 1)  
    images = [Image.open(os.path.join(image_folder, img)).convert('L') for img in selected_images]

    # 이미지 크기 조정 (폭을 image_width로 고정하고 비율에 맞게 높이 조절)
    resized_images = [img.resize((image_width, int(img.height * (image_width / img.width)))) for img in images]

    # 각 이미지 사이의 랜덤 패딩 간격을 계산하여 전체 높이 계산
    paddings = [random.randint(min_padding, max_padding) for _ in range(len(resized_images) + 1)]
    total_height = sum(img.height for img in resized_images) + sum(paddings)
    
    # 최종 이미지 생성
    final_image = Image.new('L', (image_width, total_height), color=255)
    
    # bounding box 정보 초기화
    word_data = {}
    y_offset = paddings[0]  # 첫 번째 패딩 간격으로 초기 설정

    # 이미지 붙이기와 bounding box 설정
    for idx, (img, img_name) in enumerate(zip(resized_images, selected_images)):
        final_image.paste(img, (0, y_offset))
        
        # bounding box 설정
        word_data[f"{idx + 1:04d}"] = {
            "transcription": "----",
            "points": [[0, y_offset], [image_width, y_offset], [image_width, y_offset + img.height], [0, y_offset + img.height]]
        }
        # y_offset 갱신 (현재 이미지 높이와 다음 패딩 간격 추가)
        y_offset += img.height + paddings[idx + 1]

    # 최종 이미지 이름과 경로
    final_image_name = f"extractor.ld.in_house.appen_{i:06d}_page0001.png"
    final_image_path = os.path.join(output_folder, final_image_name)
    
    # 최종 이미지 저장
    final_image.save(final_image_path)
    
    # 메타데이터 추가
    metadata = {
        "paragraphs": {},
        "words": word_data,
        "chars": {},
        "img_w": image_width,
        "img_h": total_height,
        "num_patches": None,
        "tags": [],
        "relations": {},
        "annotation_log": {
            "worker": "worker",
            "timestamp": today,
            "tool_version": "",
            "source": None
        },
        "license_tag": license_tag
    }

    # JSON 메타데이터에 이미지 이름별 데이터 추가
    metadata_dict["images"][final_image_name] = metadata

# JSON 파일로 저장
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(metadata_dict, f, ensure_ascii=False, indent=4)

print("이미지 생성 및 메타데이터 저장 완료!")
