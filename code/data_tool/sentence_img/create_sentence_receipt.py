import os
import cv2
import json
import random
import datetime
from PIL import Image

# 배경 이미지 경로
background_image_path = "bg.png"
# 이미지 폴더 경로
image_folder = ""
# 결과 저장할 json 파일 경로
output_json_path = "sentence.json"

output_folder = "output_images"

# 저장 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# Bounding Box 데이터 초기화
data = {
    "images": {}
}

def get_random_position(bg_w, bg_h, img_w, img_h, occupied_areas):
    """이미지가 겹치지 않도록 배치 위치를 계산하는 함수"""
    for _ in range(100):  # 최대 100번 시도
        x = random.randint(0, bg_w - img_w)
        y = random.randint(0, bg_h - img_h)
        new_box = [x, y, x + img_w, y + img_h]
        
        # 겹치지 않는지 확인
        if all(not (new_box[0] < area[2] and new_box[2] > area[0] and new_box[1] < area[3] and new_box[3] > area[1]) for area in occupied_areas):
            occupied_areas.append(new_box)
            return x, y, new_box
    return None  # 실패 시

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# 이미지 폴더에서 파일 읽기
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

for img_index in range(100):
    # 배경 이미지 로드
    background = cv2.imread(background_image_path)
    bg_h, bg_w, _ = background.shape

    # 랜덤하게 8개의 이미지 선택
    selected_images = random.sample(image_files, 15)

    occupied_areas = []
    words_data = {}

    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 투명도를 고려하여 이미지 로드
        
        img_h, img_w = img.shape[:2]

        # 이미지가 배경보다 크면 건너뜀
        if img_w > bg_w or img_h > bg_h:
            continue

        # 위치 결정
        position = get_random_position(bg_w, bg_h, img_w, img_h, occupied_areas)
        if position is None:
            continue  # 적절한 위치를 찾지 못한 경우 생략
        
        x, y, box = position

        # 투명 채널이 있는 경우, RGB와 알파 채널 분리
        if img.shape[2] == 4:
            rgb_img = img[:, :, :3]
            alpha_channel = img[:, :, 3] / 255.0
            # alpha 채널 적용
            for c in range(3):  # R, G, B 채널에 대해 각각 적용
                background[y:y+img_h, x:x+img_w, c] = (
                    rgb_img[:, :, c] * alpha_channel + 
                    background[y:y+img_h, x:x+img_w, c] * (1 - alpha_channel)
                )
        else:
            rgb_img = img  # RGBA 채널이 아니면 바로 사용
            background[y:y+img_h, x:x+img_w] = rgb_img

        # Bounding Box 정보 추가
        bbox_key = str(i+1).zfill(4)
        words_data[bbox_key] = {
            "transcription": "",
            "points": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        }

    # 이미지 파일 이름 생성 (확장자 포함)
    file_name = f"extractor.st.in_house.appen_{str(img_index).zfill(5)}_page0001.jpg"

    # JSON 데이터 구성
    data["images"][file_name] = {
        "paragraphs": {},
        "words": words_data,
        "chars": {},
        "img_w": bg_w,
        "img_h": bg_h,
        "tags": ["re-annotated"],
        "relations": {},
        "annotation_log": {
            "worker": "",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool_version": "CVAT",
            "source": None
        },
        "license_tag": {
            "usability": True,
            "public": False,
            "commercial": True,
            "type": None,
            "holder": "Upstage"
        }
    }

    # 결과 이미지 저장
    output_image_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_image_path, background)
    print(f"Image {img_index+1} saved as {output_image_path}")

# 최종 JSON 데이터 저장
save_json(data, output_json_path)
print(f"Bounding Box data saved as {output_json_path}")
