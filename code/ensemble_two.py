import json
import sys
import glob
import os
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
import warnings
from PIL import Image

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

def log_and_check_boxes(boxes_list):
    """
    박스 리스트의 유효성을 검사하고 로그를 남깁니다.
    """
    for model_idx, model_boxes in enumerate(boxes_list):
        for box_idx, box in enumerate(model_boxes):
            x1, y1, x2, y2 = box
            area = compute_area(box)
            print(f"Model {model_idx}, Box {box_idx}: {box.tolist()}, Area: {area:.10f}, Dtype: {box.dtype}")
            if area <= 0.0:
                print(f"Warning: Box with zero or negative area detected - Model {model_idx}, Box {box_idx}: {box.tolist()}")

def ensemble_wbf(boxes_list, iou_thr=0.5, skip_box_thr=0.0, decimals=3, epsilon=1e-6):
    """
    Weighted Box Fusion을 적용하여 박스를 앙상블합니다.
    모든 박스에 동일한 점수(1.0)를 할당하고, 단일 레이블을 사용합니다.
    """
    if not boxes_list:
        print("No boxes to ensemble.")
        return []

    # Filter out models with no boxes
    filtered_boxes_list = []
    for idx, model_boxes in enumerate(boxes_list):
        if len(model_boxes) > 0:
            filtered_boxes_list.append(model_boxes)
        else:
            print(f"Model {idx} has no boxes and will be skipped.")

    if not filtered_boxes_list:
        print("All models have no boxes to ensemble.")
        return []

    # 좌표 반올림 및 float32로 변환
    boxes_list_np = [
        np.array([round_bbox_coords(bbox, decimals) for bbox in model_boxes], dtype=np.float32)
        for model_boxes in filtered_boxes_list
    ]

    # 추가적인 유효성 검사: 모든 박스가 x2 > x1, y2 > y1인지 확인
    for model_idx, model_boxes in enumerate(boxes_list_np):
        for box_idx, box in enumerate(model_boxes):
            if box[2] <= box[0] or box[3] <= box[1]:
                print(f"Invalid box detected before WBF - Model {model_idx}, Box {box_idx}: {box.tolist()}")

    scores = [np.ones(len(model_boxes), dtype=np.float32) for model_boxes in filtered_boxes_list]
    labels = [np.ones(len(model_boxes), dtype=int) for model_boxes in filtered_boxes_list]

    # Log boxes, scores, labels
    print(f"Boxes list for WBF (numpy arrays): {boxes_list_np}")
    print(f"Scores list for WBF (numpy arrays): {scores}")
    print(f"Labels list for WBF (numpy arrays): {labels}")

    # Print each box with shape, dtype, and area
    for model_idx, model_boxes in enumerate(boxes_list_np):
        for box_idx, box in enumerate(model_boxes):
            area = compute_area(box)
            print(f"Model {model_idx}, Box {box_idx}: {box.tolist()}, Shape: {box.shape}, Dtype: {box.dtype}, Area: {area:.10f}")

    # Suppress specific WBF warnings temporarily
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # WBF 적용
        try:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list_np, scores, labels,
                weights=None,  # 모든 모델에 동일한 가중치
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr
            )
        except Exception as e:
            print(f"Error during WBF: {e}")
            return []

        # Log fused boxes
        print(f"Fused boxes: {fused_boxes}")
        print(f"Fused scores: {fused_scores}")
        print(f"Fused labels: {fused_labels}")

        # Capture and log WBF warnings
        for warning in w:
            if issubclass(warning.category, UserWarning):
                print(f"WBF Warning: {warning.message}")

    # Check for zero area fused boxes
    valid_fused = []
    for box in fused_boxes:
        area = compute_area(box)
        if area > epsilon:
            valid_fused.append(box.tolist())
        else:
            print(f"Warning: Fused box has zero or negative area: {box.tolist()}")

    return list(zip(valid_fused, fused_scores, fused_labels))

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

    # 모든 이미지 키를 수집
    all_images = set()
    for json_data in json_data_list:
        images = json_data.get("images", {}).keys()
        all_images.update(images)
    print(f"Total unique images to process: {len(all_images)}")

    for image in all_images:
        all_boxes = []
        image_path = os.path.join(image_dir, image)
        
        # 이미지 크기 가져오기
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            print(f"Image '{image}': Size = ({width}, {height})")
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
                    print(f"Warning: No points found for image '{image}', word '{word_key}'")
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
                                print(f"Warning: Invalid bbox after clipping in image '{image}', word '{word_key}': {bbox}")
                                continue
                            
                            # 정규화
                            x1_norm = x1 / width
                            y1_norm = y1 / height
                            x2_norm = x2 / width
                            y2_norm = y2 / height
                            
                            # 추가 유효성 검사: 정규화된 좌표가 [0,1] 범위 내에 있는지 확인
                            if not (0.0 <= x1_norm <= 1.0 and 0.0 <= y1_norm <= 1.0 and
                                    0.0 <= x2_norm <= 1.0 and 0.0 <= y2_norm <= 1.0):
                                print(f"Warning: Normalized bbox out of range in image '{image}', word '{word_key}': {[x1_norm, y1_norm, x2_norm, y2_norm]}")
                                continue
                            
                            # 정규화된 박스를 리스트에 추가
                            model_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                        else:
                            print(f"Warning: Invalid bbox (None or invalid) in image '{image}', word '{word_key}': {bbox}")
                    except ValueError as ve:
                        print(f"Warning: {ve} in image '{image}', word '{word_key}'")
                else:
                    print(f"Warning: Word '{word_key}' in image '{image}' does not have exactly 4 points.")
            if model_boxes:
                all_boxes.append(model_boxes)

        print(f"Image '{image}': Collected {len(all_boxes)} models with boxes")
        print(f"Image '{image}': All collected boxes: {all_boxes}")

        if all_boxes:
            # WBF 앙상블 적용
            merged = ensemble_wbf(all_boxes, iou_thr=iou_thr, skip_box_thr=skip_box_thr, decimals=decimals, epsilon=epsilon)
            print(f"Image '{image}': Merged {len(merged)} boxes")
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
                    else:
                        print(f"Warning: Merged bbox has zero or negative area for image '{image}', box '{idx}': {bbox}")
                if merged_words:
                    merged_json["images"][image] = {"words": merged_words}
                else:
                    print(f"Image '{image}': No valid boxes after conversion")
            else:
                print(f"Image '{image}': No boxes after WBF")
        else:
            print(f"Image '{image}': No boxes collected from any model")

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

def test_wbf():
    """
    WBF가 올바르게 동작하는지 확인하기 위한 테스트 함수입니다.
    """
    print("Running WBF test with sample data...")
    boxes_list = [
        [[100, 100, 200, 200], [300, 300, 400, 400]],  # 모델 1
        [[110, 110, 210, 210], [310, 310, 410, 410]]   # 모델 2
    ]
    # 정규화된 박스 계산 (임의의 이미지 크기 예: width=500, height=500)
    width, height = 500, 500
    boxes_list_norm = []
    for model_boxes in boxes_list:
        model_boxes_norm = []
        for box in model_boxes:
            x1, y1, x2, y2 = box
            model_boxes_norm.append([x1 / width, y1 / height, x2 / width, y2 / height])
        boxes_list_norm.append(model_boxes_norm)
    
    merged = ensemble_wbf(boxes_list_norm, iou_thr=0.5, skip_box_thr=0.0)
    # Denormalize fused boxes
    merged_denorm = []
    for box, score, label in merged:
        x1, y1, x2, y2 = box
        merged_denorm.append([x1 * width, y1 * height, x2 * width, y2 * height])
    print("Test WBF Merged Boxes (denormalized):", merged_denorm)

def test_specific_boxes():
    """
    문제가 되는 특정 박스를 사용하여 WBF가 정상적으로 동작하는지 확인하는 테스트 함수입니다.
    """
    print("Running specific boxes WBF test...")
    boxes_list = [
        [[52.506, 944.017, 1406.334, 1081.947]],  # 모델 1
        [[295.584, 973.694, 1714.947, 1027.632]]  # 모델 2
    ]
    
    # 이미지 크기 예: width=2000, height=1500
    width, height = 2000, 1500
    
    # 정규화된 박스 계산
    boxes_list_norm = []
    for model_boxes in boxes_list:
        model_boxes_norm = []
        for box in model_boxes:
            x1, y1, x2, y2 = box
            model_boxes_norm.append([x1 / width, y1 / height, x2 / width, y2 / height])
        boxes_list_norm.append(model_boxes_norm)
    
    # WBF 적용
    merged = ensemble_wbf(boxes_list_norm, iou_thr=0.3, skip_box_thr=0.0, decimals=3, epsilon=1e-6)
    
    # Denormalize fused boxes
    merged_denorm = []
    for box_norm, score, label in merged:
        x1_norm, y1_norm, x2_norm, y2_norm = box_norm
        x1 = x1_norm * width
        y1 = y1_norm * height
        x2 = x2_norm * width
        y2 = y2_norm * height
        merged_denorm.append([x1, y1, x2, y2])
    
    print("Fused boxes (denormalized):", merged_denorm)

def main():
    # 입력 디렉토리와 출력 파일 경로를 여기에서 지정하세요.
    input_dir = '/data/ephemeral/home/code/ensemble_files'  # 입력 파일들이 있는 디렉토리
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

    # WBF 동작 확인을 위한 테스트 실행
    test_wbf()
    test_specific_boxes()

    # JSON 데이터 앙상블 (WBF 사용)
    merged_json = ensemble_jsons_wbf(input_files, image_dir, iou_thr=0.2, skip_box_thr=0.0, decimals=3, epsilon=1e-6)  # iou_thr을 낮춤

    # 앙상블된 JSON 데이터 저장
    save_json_to_csv(merged_json, output_path)

if __name__ == "__main__":
    main()
