import os

def rename_images_in_folder(folder_path):
    # 폴더 내 파일 목록을 가져옵니다.
    files = sorted([f for f in os.listdir(folder_path) 
                    if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))])
    
    # 파일별 번호 시작 값
    for i, file_name in enumerate(files, start=1):
        # 기존 파일 경로와 확장자 추출
        old_path = os.path.join(folder_path, file_name)
        ext = os.path.splitext(file_name)[1]  # 파일 확장자 유지

        # 새로운 파일명 설정 (5자리 번호로 포맷)
        new_name = f"extractor.st.in_house.appen_{i:05d}_page0001{ext}"
        new_path = os.path.join(folder_path, new_name)

        # 파일 이름 변경
        os.rename(old_path, new_path)
        print(f"Renamed '{file_name}' to '{new_name}'")

# 실행 부분
if __name__ == "__main__":
    rename_images_in_folder('')