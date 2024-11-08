import json
import os
from datetime import datetime

# bounding_boxes.json 파일 경로와 최종 결과를 저장할 JSON 파일 경로
input_json_path = 'bounding_boxes.json'
output_json_path = 'formatted_bounding_boxes.json'

# bounding_boxes.json 파일 읽기
with open(input_json_path, 'r') as f:
    bounding_boxes_data = json.load(f)

# 현재 날짜 가져오기 (예: 2024-05-30)
current_date = datetime.now().strftime("%Y-%m-%d")

# 새로운 JSON 구조 생성
formatted_data = {"images": {}}

for entry in bounding_boxes_data:
    image_name = entry['image_name']
    bounding_box = entry['bounding_box']
    
    # 이미지 정보와 bounding box 데이터를 형식에 맞게 구성
    formatted_data["images"][image_name] = {
        "paragraphs": {},
        "words": {
            "0001": {
                "transcription": "",
                "points": bounding_box,  # bounding box를 이차원 리스트로 추가
                "orientation": "",
                "language": None,
                "tags": ["Auto"],
                "confidence": None,
                "illegibility": False
            }
        },
        "chars": {},
        "img_w": max(x for x, y in bounding_box) + 1,  # 이미지 width (bounding box의 x 최댓값 + 1)
        "img_h": max(y for x, y in bounding_box) + 1,  # 이미지 height (bounding box의 y 최댓값 + 1)
        "tags": ["re-annotated"],
        "relations": {},
        "annotation_log": {
            "worker": "",
            "timestamp": current_date,
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

# 결과를 JSON 파일로 저장
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=4, ensure_ascii=False)

print(f"변환된 JSON 데이터가 '{output_json_path}'에 저장되었습니다.")
