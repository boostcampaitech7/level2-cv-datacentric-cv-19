import json
import sys
import glob
import os
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
import warnings
from PIL import Image, ImageOps  # ImageOps 추가

def load_json_from_csv(file_path):
    """
    CSV 파일에서 JSON 데이터를 로드합니다.
    파일 확장자가 .csv여도 JSON 형식의 내용이어야 합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content)
            return data
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        sys.exit(1)

def get_all_input_files(input_dir, pattern='*.csv'):
    """
    지정된 디렉토리에서 모든 입력 파일 경로를 가져옵니다.
    """
    search_pattern = os.path.join(input_dir, pattern)
    files = glob.glob(search_pattern)
    if not files:
        print(f"No files found in {input_dir} with pattern {pattern}")
        sys.exit(1)
    return files

def round_bbox_coords(bbox, decimals=3):
    """
    주어진 bbox의 좌표를 지정된 소수점 자리수로 반올림합니다.
    """
    return [round(coord, decimals) for coord in bbox]

def convert_points_to_bbox(points, decimals=3):
    """
    4개의 점으로부터 축에 정렬된 bbox [x1, y1, x2, y2]를 생성합니다.
    좌표를 반올림하여 소수점 자릿수를 제한합니다.
    """
    if not isinstance(points, list):
        raise ValueError("Points should be a list.")
    
    # 점이 문자열로 되어 있는 경우 처리
    if all(isinstance(p, str) for p in points):
        # 문자열을 공백으로 분할하여 숫자로 변환
        try:
            flat_points = []
            for p in points:
                split_coords = p.strip().split()
                flat_points.extend([float(coord) for coord in split_coords])
            if len(flat_points) != 8:
                raise ValueError("After parsing, points do not have 8 values.")
            points = [[flat_points[i], flat_points[i+1]] for i in range(0, 8, 2)]
        except Exception as e:
            raise ValueError(f"Error parsing points from string: {e}")
    
    if len(points) != 4:
        raise ValueError("Each word must have exactly 4 points.")
    
    # 각 점이 리스트 또는 튜플인지 확인
    for point in points:
        if not (isinstance(point, list) or isinstance(point, tuple)) or len(point) != 2:
            raise ValueError("Each point must be a list or tuple of two numeric values.")
        if not all(isinstance(coord, (int, float)) for coord in point):
            raise ValueError("Point coordinates must be numeric.")
    
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    bbox = [x1, y1, x2, y2]
    bbox = clip_bbox(bbox)
    
    # 좌표 반올림
    bbox = round_bbox_coords(bbox, decimals=decimals)
    
    if not is_finite_bbox(bbox):
        raise ValueError(f"Non-finite bbox coordinates: {bbox}")
    if not is_valid_bbox(bbox):
        raise ValueError(f"Invalid bbox after clipping: {bbox}")
    
    # 추가 유효성 검사
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"Box has zero or negative area after processing: {bbox}")
    
    return bbox

def convert_bbox_to_points(bbox):
    """
    bbox [x1, y1, x2, y2]로부터 4개의 점을 생성합니다.
    """
    x1, y1, x2, y2 = bbox
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]

def clip_bbox(bbox, min_val=0):
    """
    박스 좌표가 최소값(min_val) 이상이 되도록 클리핑합니다.
    """
    return [max(coord, min_val) for coord in bbox]

def compute_area(bbox):
    """
    bbox의 면적을 계산합니다.
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def is_valid_bbox(bbox):
    """
    bbox가 유효한지 확인합니다. (x2 > x1 and y2 > y1)
    """
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1

def is_finite_bbox(bbox):
    """
    bbox의 모든 좌표가 유한한지 확인합니다.
    """
    return all(np.isfinite(coord) for coord in bbox)

def compute_iou(box1, box2):
    """
    두 박스 간의 IoU를 계산합니다.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = compute_area(box1)
    box2_area = compute_area(box2)
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    else:
        return inter_area / union_area

def ensemble_wbf(boxes_list, num_models, iou_thr=0.5, skip_box_thr=0.0, decimals=3, epsilon=1e-6):
    """
    Weighted Box Fusion을 적용하여 박스를 앙상블합니다.
    하나의 모델에서만 검출된 박스는 score를 0으로 설정하여 무시합니다.
    """
    if not boxes_list:
        print("No boxes to ensemble.")
        return []
    
    filtered_boxes_list = [model_boxes for model_boxes in boxes_list if len(model_boxes) > 0]
    
    if not filtered_boxes_list:
        print("All models have no boxes to ensemble.")
        return []
    
    boxes_list_np = [
        np.array([round_bbox_coords(bbox, decimals) for bbox in model_boxes], dtype=np.float32)
        for model_boxes in filtered_boxes_list
    ]
    
    # 모든 모델에 동일한 가중치 부여
    weights = [1.0] * len(filtered_boxes_list)
    
    # 각 모델의 박스에 동일한 점수 할당
    scores = [np.ones(len(model_boxes), dtype=np.float32) for model_boxes in filtered_boxes_list]
    labels = [np.ones(len(model_boxes), dtype=int) for model_boxes in filtered_boxes_list]
    
    try:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list_np, scores, labels, weights=weights,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
    except Exception as e:
        print(f"Error during WBF: {e}")
        return []
    
    # 각 박스가 몇 개의 모델에서 검출되었는지 계산
    model_counts = []
    for fused_box in fused_boxes:
        count = 0
        for model_boxes in boxes_list_np:
            for box in model_boxes:
                iou = compute_iou(fused_box, box)
                if iou >= iou_thr:
                    count += 1
                    break  # 한 모델에서 하나의 박스만 카운트
        model_counts.append(count)
    
    # 하나의 모델에서만 검출된 박스의 점수를 0으로 설정
    for idx, count in enumerate(model_counts):
        if count <= 1:
            fused_scores[idx] = 0.0
    
    valid_fused = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        if score > 0.0 and compute_area(box) > epsilon:
            valid_fused.append((box.tolist(), score, label))
    
    return valid_fused

def ensemble_jsons_wbf(input_files, image_dir, iou_thr=0.5, skip_box_thr=0.0, decimals=3, epsilon=1e-6):
    """
    여러 개의 JSON 데이터를 WBF를 사용하여 앙상블합니다.
    """
    merged_json = {"images": {}}
    
    # 모든 JSON 데이터를 로드
    json_data_list = []
    for f in input_files:
        data = load_json_from_csv(f)
        json_data_list.append(data)
        num_images = len(data.get('images', {}))
        print(f"Loaded {f}: {num_images} images")
    
    num_models = len(json_data_list)
    
    # 모든 이미지 키를 수집
    all_images = set()
    for json_data in json_data_list:
        images = json_data.get("images", {}).keys()
        all_images.update(images)
    print(f"Total unique images to process: {len(all_images)}")
    
    for image in all_images:
        all_boxes = []
        image_path = os.path.join(image_dir, image)
        
        # 이미지 크기 가져오기 및 EXIF 회전 처리
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)  # EXIF 회전 정보 적용
                width, height = img.size
        except Exception as e:
            print(f"Error opening image '{image}': {e}")
            continue  # 이미지 크기를 알 수 없으면 건너뜀
    
        for json_idx, json_data in enumerate(json_data_list):
            image_data = json_data.get("images", {}).get(image, {})
            words = image_data.get("words", {})
            model_boxes = []
            for word_key, word_info in words.items():
                points = word_info.get("points", [])
                if not points:
                    continue
                if len(points) == 4:
                    try:
                        bbox = convert_points_to_bbox(points, decimals=decimals)
                        if bbox and is_valid_bbox(bbox):
                            # 박스를 이미지 크기로 클리핑
                            x1, y1, x2, y2 = bbox
                            x1 = max(0, min(x1, width))
                            y1 = max(0, min(y1, height))
                            x2 = max(0, min(x2, width))
                            y2 = max(0, min(y2, height))
                            bbox = [x1, y1, x2, y2]
                            
                            if not is_valid_bbox(bbox):
                                continue
                            
                            # 정규화
                            x1_norm = x1 / width
                            y1_norm = y1 / height
                            x2_norm = x2 / width
                            y2_norm = y2 / height
                            
                            # 정규화된 박스를 리스트에 추가
                            model_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                        else:
                            continue
                    except ValueError as ve:
                        continue
                else:
                    continue
            if model_boxes:
                all_boxes.append(model_boxes)
    
        if all_boxes:
            # WBF 앙상블 적용
            merged = ensemble_wbf(all_boxes, num_models, iou_thr=iou_thr, skip_box_thr=skip_box_thr, decimals=decimals, epsilon=epsilon)
            if merged:
                merged_words = {}
                for idx, (bbox_norm, _, _) in enumerate(merged):
                    # 정규화된 박스를 다시 원래 좌표로 변환
                    x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
                    x1 = x1_norm * width
                    y1 = y1_norm * height
                    x2 = x2_norm * width
                    y2 = y2_norm * height
                    bbox = [x1, y1, x2, y2]
                    if is_valid_bbox(bbox):
                        points = convert_bbox_to_points(bbox)
                        merged_words[str(idx)] = {
                            "points": points
                        }
                if merged_words:
                    merged_json["images"][image] = {"words": merged_words}
    
        # else:
            # 이미지에 대해 앙상블된 박스가 없는 경우 처리하지 않음
    
    return merged_json

def save_json_to_csv(json_data, output_path):
    """
    JSON 데이터를 CSV 파일로 저장합니다.
    파일 확장자는 .csv이지만 내용은 JSON 형식입니다.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"Merged JSON saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {e}")
        sys.exit(1)

def main():
    # 입력 디렉토리와 출력 파일 경로를 여기에서 지정하세요.
    input_dir = '/data/ephemeral/home/code/no_line_friends'  # 입력 파일들이 있는 디렉토리
    input_pattern = '*.csv'  # 입력 파일 패턴
    output_path = '/data/ephemeral/home/code/ensemble_results/merged_result.csv'  # 앙상블된 결과를 저장할 파일 경로
    image_dir = '/data/ephemeral/home/code/images/'  # 이미지 파일들이 있는 디렉토리 (수정 필요)
    
    # 파일 경로 출력 (선택 사항)
    print(f"Loading input files from directory: {input_dir} with pattern: {input_pattern}")

    # 모든 입력 파일 로드
    input_files = get_all_input_files(input_dir, input_pattern)
    for f in input_files:
        print(f"Loading input file: {f}")

    print(f"Output will be saved to: {output_path}")
    print(f"Images are expected to be in: {image_dir}")

    # JSON 데이터 앙상블 (WBF 사용)
    merged_json = ensemble_jsons_wbf(input_files, image_dir, iou_thr=0.35, skip_box_thr=0.0, decimals=3, epsilon=1e-6)

    # 앙상블된 JSON 데이터 저장
    save_json_to_csv(merged_json, output_path)

if __name__ == "__main__":
    main()
