import json
import os
import random
import shutil

# 각 폴더의 경로 설정
folders = [
    "R_chinese_receipt",
    "R_japanese_receipt",
    "R_thai_receipt",
    "R_vietnamese_receipt"
]

# 최종 결과를 저장할 폴더 및 파일 경로
output_img_folder = "train"
output_json_path = "train.json"
os.makedirs(output_img_folder, exist_ok=True)

# 통합 JSON 데이터 구조
combined_data = {"images": {}}

# 각 폴더에서 25장씩 선택하여 처리
for folder in folders:
    # 이미지와 JSON 파일 경로 설정
    img_folder_path = os.path.join(folder, "img", "train")
    json_file_path = os.path.join(folder, "ufo", "train.json")

    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 폴더 내 이미지 파일 이름 리스트 생성 및 랜덤으로 25장 선택
    img_files = [f for f in os.listdir(img_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_imgs = random.sample(img_files, 25)

    # 선택된 이미지와 해당하는 JSON 데이터 처리
    for img_name in selected_imgs:
        # 이미지 파일 복사
        src_img_path = os.path.join(img_folder_path, img_name)
        dst_img_path = os.path.join(output_img_folder, img_name)
        shutil.copy2(src_img_path, dst_img_path)

        # JSON 데이터 가져와 통합 JSON에 추가
        if img_name in json_data["images"]:
            combined_data["images"][img_name] = json_data["images"][img_name]
        else:
            print(f"경고: {img_name}에 대한 JSON 데이터가 {json_file_path}에 없습니다.")

# 통합 JSON 파일로 저장
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print(f"통합된 JSON 파일이 '{output_json_path}'에 저장되었고, 이미지들은 '{output_img_folder}' 폴더에 복사되었습니다.")
