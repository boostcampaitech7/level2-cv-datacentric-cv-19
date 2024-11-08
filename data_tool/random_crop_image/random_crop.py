import json
import os
import random
from PIL import Image

def crop_and_adjust_bbox(json_path, img_folder, output_json_path, crop_width, crop_height):
    # UTF-8 인코딩으로 JSON 파일 열기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {"images": {}}

    for img_name, img_data in data["images"].items():
        img_path = os.path.join(img_folder, img_name)
        img = Image.open(img_path)
        img_w, img_h = img.size

        # 크롭 가능한 최대 위치를 기준으로 랜덤 시작점 생성
        crop_x1 = random.randint(0, max(0, img_w - crop_width))
        crop_y1 = random.randint(0, max(0, img_h - crop_height))
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        # 이미지 크롭 및 크롭된 이미지 크기 저장
        cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        cropped_img_w, cropped_img_h = cropped_img.size

        # 크롭된 이미지와 바운딩 박스 정보를 저장할 딕셔너리 초기화
        new_img_data = {
            "paragraphs": {},
            "words": {},
            "chars": {},
            "img_w": cropped_img_w,
            "img_h": cropped_img_h,
            "tags": img_data["tags"],
            "relations": img_data["relations"],
            "annotation_log": img_data["annotation_log"],
            "license_tag": img_data["license_tag"]
        }

        for word_id, word_data in img_data["words"].items():
            adjusted_points = []
            is_within_crop = False

            # 좌상단 좌표가 크롭 영역 내에 있는지 확인 및 조정
            for (x, y) in word_data["points"]:
                new_x = x - crop_x1
                new_y = y - crop_y1

                # 좌표가 크롭된 이미지 내에 존재하는 경우 조정
                if 0 <= new_x < cropped_img_w and 0 <= new_y < cropped_img_h:
                    is_within_crop = True
                    adjusted_points.append([max(0, min(cropped_img_w, new_x)),
                                            max(0, min(cropped_img_h, new_y))])
                else:
                    # 좌표가 크롭 영역 밖에 있을 경우 가장자리에 맞게 조정
                    adjusted_points.append([max(0, min(cropped_img_w, new_x)),
                                            max(0, min(cropped_img_h, new_y))])

            # 바운딩 박스의 일부라도 크롭 영역에 포함되는 경우에만 포함
            if is_within_crop:
                new_img_data["words"][word_id] = {
                    "transcription": word_data["transcription"],
                    "points": adjusted_points,
                    # "orientation": word_data["orientation"],
                    # "language": word_data["language"],
                    # "tags": word_data["tags"],
                    # "confidence": word_data["confidence"],
                    # "illegibility": word_data["illegibility"]
                }

        # 수정된 이미지 데이터 저장
        new_data["images"][img_name] = new_img_data

        # 크롭한 이미지를 RGB로 변환한 후 새로운 파일로 저장 (예: "cropped_{이미지이름}.jpg")
        if cropped_img.mode in ["RGBA", "P"]:  # RGBA 또는 P 모드인 경우 RGB로 변환
            cropped_img = cropped_img.convert("RGB")
        
        cropped_img.save(os.path.join(img_folder, f"{img_name}"), format="JPEG")

    # 결과 JSON 파일 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)


# 사용 예제
crop_and_adjust_bbox(
    json_path="",
    img_folder="",
    output_json_path="",
    crop_width=500,   # 원하는 크롭 너비
    crop_height=500   # 원하는 크롭 높이
)