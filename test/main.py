import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "soccer-video-analytics"))
from inference.colors import red, white


# ==========================================
# 1. 기본 유틸리티 및 색상 처리
# ==========================================
# from typing import List, Tuple
# import numpy as np

def filter_bboxes_by_fill_ratio(
    bboxes: List[Tuple[int, int, int, int]],
    mask: np.ndarray,
    min_fill_ratio: float = 0.08,
    min_white_pixels: int = 25,
) -> List[Tuple[int, int, int, int]]:
    """
    bboxes: (x,y,w,h) 리스트
    mask : 0/255 이진 마스크 (예: mask_team1_final)
    min_fill_ratio: 박스 내부에서 흰색 픽셀 비율 최소값
    min_white_pixels: 흰색 픽셀 절대 개수도 너무 작으면 제거(얼굴/점 노이즈 방지)
    """
    if mask is None or mask.size == 0:
        return bboxes

    H, W = mask.shape[:2]
    filtered = []

    for (x, y, w, h) in bboxes:
        if w <= 0 or h <= 0:
            continue

        # 안전하게 클램핑
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = mask[y1:y2, x1:x2]
        area = roi.size
        if area <= 0:
            continue

        white = int(np.count_nonzero(roi))  # mask가 0/255면 nonzero가 곧 흰 픽셀 수미
        fill_ratio = white / float(area)

        if white >= min_white_pixels and fill_ratio >= min_fill_ratio:
            filtered.append((x, y, w, h))

    return filtered

def bgr_range(b: int, g: int, r: int, tolerance: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    b, g, r = int(b), int(g), int(r)
    tolerance = int(tolerance)
    lower_color = np.array(
        [
            max(0, b - tolerance),
            max(0, g - tolerance),
            max(0, r - tolerance),
        ],
        dtype=np.uint8,
    )
    upper_color = np.array(
        [
            min(255, b + tolerance),
            min(255, g + tolerance),
            min(255, r + tolerance),
        ],
        dtype=np.uint8,
    )
    return lower_color, upper_color


def create_uniform_mask(frame: np.ndarray, team_color_bgr: Tuple[int, int, int]) -> np.ndarray:
    lower_color, upper_color = bgr_range(
        team_color_bgr[0],
        team_color_bgr[1],
        team_color_bgr[2],
    )
    return cv2.inRange(frame, lower_color, upper_color)


def create_hsv_mask(frame: np.ndarray, hsv_color_filters: List[dict]) -> np.ndarray:
    """
    hsv_color_filters: [{"lower_hsv": (...), "upper_hsv": (...)}] 형태
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    merged_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    for color_filter in hsv_color_filters:
        lower = np.array(color_filter["lower_hsv"], dtype=np.uint8)
        upper = np.array(color_filter["upper_hsv"], dtype=np.uint8)
        merged_mask = cv2.bitwise_or(merged_mask, cv2.inRange(hsv_frame, lower, upper))

    return merged_mask


def get_dominant_colors(
    image: np.ndarray, mask: np.ndarray, n_colors: int = 3, color_space: str = "bgr"
) -> List[dict]:
    masked_pixels = image[mask == 255]
    if len(masked_pixels) == 0:
        return []

    reduced_pixels = (masked_pixels // 32) * 32
    unique_colors, counts = np.unique(reduced_pixels.reshape(-1, 3), axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]

    result = []
    total_pixels = len(masked_pixels)
    for i in range(min(n_colors, len(unique_colors))):
        idx = sorted_indices[i]
        color = unique_colors[idx]
        if color_space.lower() == "rgb":
            color_rgb = tuple(color[::-1])
            color_bgr = tuple(color)
        elif color_space.lower() == "bgr":
            color_bgr = tuple(color)
            color_rgb = tuple(color[::-1])
        else:
            color_bgr = tuple(color)
            color_rgb = tuple(color[::-1])

        percentage = counts[idx] / total_pixels
        result.append(
            {
                "color_bgr": color_bgr,  #opencv는 BGR이 기본이므로 BGR도 함께 저장
                "color_rgb": color_rgb,  #사용자 친화적 RGB도 함께 저장
                "percentage": percentage,
            }
        )
    return result


def integrate_realtime_colors(
    frame: np.ndarray, mask: np.ndarray, color_space: str = "bgr"
) -> Optional[Tuple[int, int, int]]:
    dominant_colors = get_dominant_colors(frame, mask, n_colors=3, color_space=color_space)
    if not dominant_colors:
        return None

    dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)
    top_color = dominant_colors[0]

    if color_space.lower() == "rgb":
        return tuple(c + 16 for c in top_color["color_rgb"])
    if color_space.lower() == "bgr":
        return tuple(c + 16 for c in top_color["color_bgr"])
    return top_color["color_bgr"]


def is_on_field(frame: np.ndarray, x: int, y: int, w: int, h: int, grass_color, tolerance: int) -> bool: #이거 없애도 괜찮은지 추후 검토 ... bounding box 주변 point 들이 과연 의미가 있는지 잘 모르는 위치에 찍힌다. 
    sample_points = []
    height, width = frame.shape[:2]

    left_x = max(0, x - 10)
    for i in range(3):
        sample_y = y + (h * (i + 1) // 4)
        if 0 <= sample_y < height:
            sample_points.append(frame[sample_y, left_x])

    right_x = min(width - 1, x + w + 10)
    for i in range(3):
        sample_y = y + (h * (i + 1) // 4)
        if 0 <= sample_y < height:
            sample_points.append(frame[sample_y, right_x])

    if not sample_points:
        return False

    avg_color = np.mean(sample_points, axis=0)
    color_diff = np.abs(avg_color - grass_color)
    return np.all(color_diff <= tolerance)


def get_bounding_boxes(
    frame: np.ndarray, mask: np.ndarray, grass_color: Tuple[int, int, int], min_area: int = 10, tolerance: int = 30
) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if is_on_field(frame, x, y, w, h, grass_color, tolerance):
                bounding_boxes.append((x, y, w, h))
    return bounding_boxes


# ==========================================
# 2. [NEW] 자동 팀 컬러 감지 (K-Means)
# ==========================================
'''
def detect_team_colors_automatically(
    frame: np.ndarray, grass_color_bgr: Tuple[int, int, int]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    
    print(">>> Initializing automatic team color detection (K-Means Clustering Mode)...")
    
    # 1. 잔디 마스크 생성 및 객체 분리
    lower_green, upper_green = bgr_range(grass_color_bgr[0], grass_color_bgr[1], grass_color_bgr[2], tolerance=60)
    mask_green = cv2.inRange(frame, lower_green, upper_green)
    mask_objects = cv2.bitwise_not(mask_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_objects = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_colors = []
    
    # 디버깅 캔버스
    debug_canvas = np.zeros((300, 600, 3), dtype=np.uint8)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0 < area < 5000: 
            x, y, w, h = cv2.boundingRect(cnt)
            # 사람 비율 확인
            if float(h) / w < 1.2: continue
            # 경기장 내부 확인
            if not is_on_field(frame, x, y, w, h, grass_color_bgr, tolerance=40): continue
            
            # 상반신 ROI (바지 제외)
            roi_y = y + int(h * 0.1)
            roi_h = int(h * 0.4) 
            roi_x = x + int(w * 0.25)
            roi_w = int(w * 0.5)
            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            if roi.size == 0: continue

            # [수정 1] 개별 선수 ROI 내부에서 K-Means 수행
            # 속도 향상을 위해 작게 리사이즈
            roi_small = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
            data_roi = roi_small.reshape(-1, 3).astype(np.float32)
            
            # K=2로 나누어 (유니폼 색 vs 그림자/마킹/노이즈) 분리
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k_roi = 2
            try:
                _, labels, centers = cv2.kmeans(data_roi, k_roi, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            except Exception:
                continue

            # 가장 많은 픽셀이 속한 클러스터(주 색상) 선택
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]
            dominant_color = centers[dominant_label] # [B, G, R] Float
            
            # [수정 2] 필터링 조건 대폭 완화 (흰/검 포함)
            b, g, r = dominant_color
            hsv_pixel = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
            hue, sat, val = hsv_pixel

            # 조건: 
            # 1. 유채색: 채도(S) > 20 (기존 40에서 완화)
            # 2. 흰색: 명도(V) > 160 (기존 200에서 완화)
            # 3. 검정: 명도(V) < 70 (기존 30에서 완화)
            is_colorful = sat > 20
            is_white = val > 160
            is_black = val < 70
            
            # 애매한 회색(채도 낮고 명도 중간)만 제외하고 다 허용
            if is_colorful or is_white or is_black:
                 player_colors.append(dominant_color)
                 # 디버깅 점 찍기
                 pt_color = tuple(map(int, dominant_color))
                 cv2.circle(debug_canvas, (len(player_colors) * 5 % 600, 50 + (len(player_colors) // 120) * 10), 4, pt_color, -1)

    if len(player_colors) < 2:
        print("Warning: Not enough players. Using Default.")
        return (0, 0, 255), (255, 0, 0)

    # 3. 전체 팀 분류 (Global K-Means)
    data_global = np.float32(player_colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k_global = 2
    _, labels_global, centers_global = cv2.kmeans(data_global, k_global, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # [수정 3] Group Center에서 가장 가까운 실제 데이터 점 찾기 (Medoid 개념)
    # 이유: 평균값은 실제 유니폼에 없는 탁한 색일 수 있음. 실제 수집된 데이터 중 가장 대표성을 띠는 것을 선택.
    
    def get_closest_point(target_center, points):
        min_dist = float('inf')
        closest = target_center
        for pt in points:
            dist = np.linalg.norm(pt - target_center)
            if dist < min_dist:
                min_dist = dist
                closest = pt
        return closest

    # 라벨별로 데이터 분리
    labels_global = labels_global.flatten()
    group0 = data_global[labels_global == 0]
    group1 = data_global[labels_global == 1]
    
    if len(group0) > 0:
        center0 = get_closest_point(centers_global[0], group0)
    else:
        center0 = centers_global[0]
        
    if len(group1) > 0:
        center1 = get_closest_point(centers_global[1], group1)
    else:
        center1 = centers_global[1]

    team1_color = (int(center0[0]), int(center0[1]), int(center0[2]))
    team2_color = (int(center1[0]), int(center1[1]), int(center1[2]))
    
    # 결과 시각화
    cv2.rectangle(debug_canvas, (50, 150), (250, 250), team1_color, -1)
    cv2.putText(debug_canvas, "Team 1", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(debug_canvas, (350, 150), (550, 250), team2_color, -1)
    cv2.putText(debug_canvas, "Team 2", (350, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    print(f">>> Auto-detected Team 1: {team1_color}")
    print(f">>> Auto-detected Team 2: {team2_color}")
    
    cv2.imshow("Color Detection Debug", debug_canvas)
    cv2.waitKey(3000) 
    cv2.destroyWindow("Color Detection Debug")
    
    return team1_color, team2_color
'''# ==========================================
# 3. 공 감지 및 시각화
# ==========================================
def detect_team_colors_automatically(
    frame: np.ndarray, grass_color_bgr: Tuple[int, int, int]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    
    print(">>> Initializing automatic team color detection (Global Clustering & Medoid Mode)...")
    
    # 1. 잔디 마스크 생성 (배경 제거)
    lower_green, upper_green = bgr_range(grass_color_bgr[0], grass_color_bgr[1], grass_color_bgr[2], tolerance=60)
    mask_green = cv2.inRange(frame, lower_green, upper_green)
    mask_objects = cv2.bitwise_not(mask_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_objects = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_colors = []
    
    # 디버깅용 캔버스
    debug_canvas = np.zeros((300, 600, 3), dtype=np.uint8)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0 < area < 5000: 
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 비율 체크 (사람 형태)
            if float(h) / w < 1.2: continue
            # 경기장 내부인지 체크
            if not is_on_field(frame, x, y, w, h, grass_color_bgr, tolerance=40): continue
            
            # 상반신 ROI (바지 색 간섭 제외)
            roi_y = y + int(h * 0.1)
            roi_h = int(h * 0.4) 
            roi_x = x + int(w * 0.25)
            roi_w = int(w * 0.5)
            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            if roi.size == 0: continue

            # [Step 1] 각 선수마다 하나의 대표 색상 추출 (Histogram 방식)
            # 이미지를 작게 줄여서 노이즈를 뭉갭니다.
            roi_small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_NEAREST)
            pixels = roi_small.reshape(-1, 3)
            
            # 색상을 단순화 (10단위로 끊음) 하여 최빈값 추출   
            pixels_quantized = (pixels // 10) * 10 
            colors, counts = np.unique(pixels_quantized, axis=0, return_counts=True)
            dominant_color = colors[np.argmax(counts)] # [B, G, R]
            
            # [Step 2] 필터링 조건 복구 (Strict Mode)
            # 완화했던 조건을 다시 엄격하게 되돌려 배경/회색 노이즈를 차단합니다.
            b, g, r = dominant_color
            hsv_pixel = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
            hue, sat, val = hsv_pixel

            # 엄격한 필터: 
            # 채도(Sat)가 40 미만이면 무조건 탈락 (회색/탁한색 제거)
            # 명도(Val)가 너무 어둡거나(30 미만) 너무 밝으면(230 초과) 탈락
            if sat < 40 or val < 30 or val > 230:
                continue

            # 유효한 선수 색상으로 등록
            player_colors.append(dominant_color)
            
            # 디버깅용 점 찍기
            pt_color = tuple(map(int, dominant_color))
            cv2.circle(debug_canvas, (len(player_colors) * 5 % 600, 50 + (len(player_colors) // 120) * 10), 4, pt_color, -1)

    if len(player_colors) < 2:
        print("Warning: Not enough players detected after strict filtering. Using Default.")
        return (0, 0, 255), (255, 0, 0)

    # [Step 3] 전체 선수 색상 클러스터링 (Global K-Means)
    # 모아진 모든 선수의 색상을 2개 그룹으로 나눕니다.
    data = np.float32(player_colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # [Step 4] Medoid 방식: 중심점에서 가장 가까운 '실제 데이터' 찾기
    labels = labels.flatten()
    
    def get_medoid(target_center, points):
        if len(points) == 0: return target_center
        min_dist = float('inf')
        closest_point = target_center
        for pt in points:
            # 유클리드 거리 계산
            dist = np.linalg.norm(pt - target_center)
            if dist < min_dist:
                min_dist = dist
                closest_point = pt
        return closest_point

    # 그룹 0의 대표색 찾기
    group0_points = data[labels == 0]
    final_color0 = get_medoid(centers[0], group0_points)
    
    # 그룹 1의 대표색 찾기
    group1_points = data[labels == 1]
    final_color1 = get_medoid(centers[1], group1_points)

    # 정수형 변환
    team1_color = (int(final_color0[0]), int(final_color0[1]), int(final_color0[2]))
    team2_color = (int(final_color1[0]), int(final_color1[1]), int(final_color1[2]))
    
    # K-Means 결과 시각화 (클러스터 라벨, 중심, 최종 팀 색)
    kmeans_vis = np.zeros((420, 920, 3), dtype=np.uint8)
    cv2.putText(
        kmeans_vis,
        "K-Means Result Visualization",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        kmeans_vis,
        "Scatter (HSV: H on X, S on Y) with cluster labels",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )

    # Scatter 영역
    sx1, sy1, sx2, sy2 = 20, 80, 600, 400
    cv2.rectangle(kmeans_vis, (sx1, sy1), (sx2, sy2), (80, 80, 80), 1)

    cluster_colors = [(0, 255, 255), (255, 255, 0)]  # label 0, label 1
    hsv_points = cv2.cvtColor(np.uint8(data.reshape(-1, 1, 3)), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    for i, hsv_pt in enumerate(hsv_points):
        label = int(labels[i])
        x = int(sx1 + 10 + (float(hsv_pt[0]) / 179.0) * (sx2 - sx1 - 20))
        y = int(sy2 - 10 - (float(hsv_pt[1]) / 255.0) * (sy2 - sy1 - 20))
        cv2.circle(kmeans_vis, (x, y), 2, cluster_colors[label], -1)

    # K-Means center 시각화
    centers_uint8 = np.clip(centers, 0, 255).astype(np.uint8)
    hsv_centers = cv2.cvtColor(centers_uint8.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    for i, hsv_c in enumerate(hsv_centers):
        cx = int(sx1 + 10 + (float(hsv_c[0]) / 179.0) * (sx2 - sx1 - 20))
        cy = int(sy2 - 10 - (float(hsv_c[1]) / 255.0) * (sy2 - sy1 - 20))
        cv2.circle(kmeans_vis, (cx, cy), 8, (255, 255, 255), 2)
        cv2.putText(
            kmeans_vis,
            f"C{i}",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # 우측: cluster center / medoid / final team color
    panel_x = 640
    cv2.putText(kmeans_vis, "Cluster 0", (panel_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cluster_colors[0], 2)
    cv2.putText(kmeans_vis, "Cluster 1", (panel_x, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cluster_colors[1], 2)

    center0 = tuple(int(v) for v in np.clip(centers[0], 0, 255))
    center1 = tuple(int(v) for v in np.clip(centers[1], 0, 255))
    cv2.rectangle(kmeans_vis, (panel_x, 120), (panel_x + 80, 175), center0, -1)  # KMeans center 0
    cv2.rectangle(kmeans_vis, (panel_x + 90, 120), (panel_x + 170, 175), team1_color, -1)  # Medoid(final 0)
    cv2.rectangle(kmeans_vis, (panel_x, 240), (panel_x + 80, 295), center1, -1)  # KMeans center 1
    cv2.rectangle(kmeans_vis, (panel_x + 90, 240), (panel_x + 170, 295), team2_color, -1)  # Medoid(final 1)

    cv2.putText(kmeans_vis, "center", (panel_x, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(kmeans_vis, "medoid/final", (panel_x + 90, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(kmeans_vis, f"Team1 BGR: {team1_color}", (640, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(kmeans_vis, f"Team2 BGR: {team2_color}", (640, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    
    print(f">>> Auto-detected Team 1: {team1_color}")
    print(f">>> Auto-detected Team 2: {team2_color}")
    
    cv2.imshow("Color Detection Debug", kmeans_vis)
    cv2.waitKey(3000) 
    cv2.destroyWindow("Color Detection Debug")
    
    return team1_color, team2_color

def draw_boxes_on_frame(
    frame: np.ndarray, bounding_boxes: List[Tuple[int, int, int, int]], color: Tuple[int, int, int]
) -> np.ndarray:
    result = frame.copy()
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    return result


def is_near_player( # 검토 필요..
    ball_center: Tuple[int, int],
    player_bboxes: List[Tuple[int, int, int, int]],
    distance_threshold: int = 20,
) -> bool:
    if not player_bboxes:
        return False
    ball_x, ball_y = ball_center
    for bbox in player_bboxes:
        px, py, pw, ph = bbox
        if px <= ball_x <= px + pw and py <= ball_y <= py + ph:
            return True
        expanded_x1 = px - distance_threshold
        expanded_y1 = py - distance_threshold
        expanded_x2 = px + pw + distance_threshold
        expanded_y2 = py + ph + distance_threshold
        if expanded_x1 <= ball_x <= expanded_x2 and expanded_y1 <= ball_y <= expanded_y2:
            return True
    return False


def is_ball_on_field(ball_center: Tuple[int, int], grass_mask: np.ndarray, surrounding_radius: int = 15) -> bool: #검토 필요 
    if grass_mask is None:
        return True
    ball_x, ball_y = ball_center
    height, width = grass_mask.shape
    if ball_x < 0 or ball_x >= width or ball_y < 0 or ball_y >= height:
        return False
    x1 = max(0, ball_x - surrounding_radius)
    y1 = max(0, ball_y - surrounding_radius)
    x2 = min(width, ball_x + surrounding_radius + 1)
    y2 = min(height, ball_y + surrounding_radius + 1)
    surrounding_area = grass_mask[y1:y2, x1:x2]
    if surrounding_area.size == 0:
        return False
    grass_ratio = np.sum(surrounding_area > 0) / surrounding_area.size
    return grass_ratio > 0.6


def detect_ball(
    frame: np.ndarray,
    grass_mask: Optional[np.ndarray],
    ball_color: Tuple[int, int, int],
    player_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ball_mask = create_uniform_mask(frame, ball_color)
    ball_mask = cv2.GaussianBlur(ball_mask, (3, 3), 0)
    gray = cv2.Canny(gray, 50, 150)

    if grass_mask is not None:
        not_grass_mask = cv2.bitwise_not(grass_mask)
    else:
        not_grass_mask = np.ones_like(gray) * 255

    gray = cv2.bitwise_or(ball_mask, not_grass_mask)  #굳이..? 검토 필요 

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 4
    params.maxArea = 40
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    ball_candidates = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        size = keypoint.size
        radius = size / 2
        bbox_x = int(x - radius)
        bbox_y = int(y - radius)
        bbox_size = int(size)
        if (
            bbox_x >= 0
            and bbox_y >= 0
            and bbox_x + bbox_size < frame.shape[1]
            and bbox_y + bbox_size < frame.shape[0]
        ):
            ball_center = (int(x), int(y))
            if player_bboxes and is_near_player(ball_center, player_bboxes):
                continue
            if not is_ball_on_field(ball_center, grass_mask):
                continue
            ball_candidates.append((bbox_x, bbox_y, bbox_size, bbox_size))
    return ball_candidates


def filter_ball_by_field_position(   #검토 필요.. 너무 엄격하게 중앙에만 공이 잡히는 현상이 발생한다.
    ball_bboxes: List[Tuple[int, int, int, int]], frame_shape: Tuple[int, int]
) -> List[Tuple[int, int, int, int]]:
    if not ball_bboxes:
        return ball_bboxes
    height, width = frame_shape[:2]
    margin_x = width // 10
    margin_y = height // 10
    filtered_balls = []
    for bbox in ball_bboxes:
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        if margin_x < center_x < width - margin_x and margin_y < center_y < height - margin_y:
            filtered_balls.append(bbox)
    return filtered_balls


def draw_ball_detection(
    frame: np.ndarray, ball_bboxes: List[Tuple[int, int, int, int]], color: Tuple[int, int, int]
) -> np.ndarray:
    result = frame.copy()
    for bbox in ball_bboxes:
        x, y, w, h = bbox
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        center_x = x + w // 2
        center_y = y + h // 2
        radius = w // 2
        cv2.circle(result, (center_x, center_y), 3, color, -1)
        cv2.circle(result, (center_x, center_y), radius, color, 2)
        cv2.putText(result, "BALL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return result


# ==========================================
# 4. 트래킹 시스템 (원본의 강력한 로직 복구됨)
# ==========================================

class PlayerTracker:
    def __init__(
        self,
        team_id: int,
        initial_bbox: Tuple[int, int, int, int],
        tracker_id: int,
        grass_color: Tuple[int, int, int],
    ):
        self.team_id = team_id
        self.tracker_id = tracker_id
        self.current_bbox = initial_bbox
        self.previous_bbox = initial_bbox
        self.dx = 0
        self.dy = 0
        self.frames_lost_score = 0
        self.max_lost_frames_in_bounds = 45
        self.max_lost_frames_out_bounds = 2
        self.out_of_bounds_frames = 0
        self.grass_color = grass_color

    def update_bbox(self, new_bbox: Tuple[int, int, int, int]) -> None:
        self.previous_bbox = self.current_bbox
        self.current_bbox = new_bbox
        prev_center_x = self.previous_bbox[0] + self.previous_bbox[2] // 2
        prev_center_y = self.previous_bbox[1] + self.previous_bbox[3] // 2
        curr_center_x = new_bbox[0] + new_bbox[2] // 2
        curr_center_y = new_bbox[1] + new_bbox[3] // 2
        self.dx = curr_center_x - prev_center_x
        self.dy = curr_center_y - prev_center_y
        self.frames_lost_score = 0
        self.out_of_bounds_frames = 0

    def predict_next_position(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.current_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        pred_center_x = center_x + self.dx
        pred_center_y = center_y + self.dy
        pred_x = pred_center_x - w // 2
        pred_y = pred_center_y - h // 2
        return (int(pred_x), int(pred_y), w, h)

    def get_center(self) -> Tuple[int, int]:
        x, y, w, h = self.current_bbox
        return (x + w // 2, y + h // 2)

    def calculate_predicted_distance_to_bbox(self, bbox: Tuple[int, int, int, int]) -> float:  #모든 bbox 에 대해서 계산 중... 어떠한 방법이 없을까? 
        predicted_bbox = self.predict_next_position()
        pred_center_x = predicted_bbox[0] + predicted_bbox[2] // 2
        pred_center_y = predicted_bbox[1] + predicted_bbox[3] // 2
        bbox_center_x = bbox[0] + bbox[2] // 2
        bbox_center_y = bbox[1] + bbox[3] // 2
        dx = pred_center_x - bbox_center_x
        dy = pred_center_y - bbox_center_y
        return np.sqrt(dx * dx + dy * dy)

    def is_out_of_bounds(self, frame_width: int, frame_height: int) -> bool:
        x, y, w, h = self.current_bbox
        completely_out = x + w < 0 or x > frame_width or y + h < 0 or y > frame_height #현재 화면 밖이냐 
        if completely_out:
            return True
        center_x = x + w // 2
        center_y = y + h // 2
        next_center_x = center_x + self.dx * self.frames_lost_score
        next_center_y = center_y + self.dy * self.frames_lost_score
        margin_x = w // 2
        margin_y = h // 2
        near_left_edge = x <= margin_x and self.dx < 0  #경계에 걸쳐져 있고 나가는 중이냐
        near_right_edge = x + w >= frame_width - margin_x and self.dx > 0
        near_top_edge = y <= margin_y and self.dy < 0
        near_bottom_edge = y + h >= frame_height - margin_y and self.dy > 0
        next_out_of_bounds = (
            next_center_x - margin_x < 0
            or next_center_x + margin_x > frame_width
            or next_center_y - margin_y < 0
            or next_center_y + margin_y > frame_height
        )
        return near_left_edge or near_right_edge or near_top_edge or near_bottom_edge or next_out_of_bounds

    def should_be_removed(self, frame_width: int, frame_height: int) -> bool:
        if self.is_out_of_bounds(frame_width, frame_height):
            self.out_of_bounds_frames += 1
            return self.out_of_bounds_frames > self.max_lost_frames_out_bounds
        return self.frames_lost_score > self.max_lost_frames_in_bounds

    def is_bbox_on_grass(self, frame: np.ndarray) -> bool:
        if frame is None:
            return False
        x, y, w, h = self.current_bbox
        frame_height, frame_width = frame.shape[:2]
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        w = max(1, min(w, frame_width - x))
        h = max(1, min(h, frame_height - y))
        center_x = x + w // 2
        center_y = y + h // 2
        sample_size = 2
        x1 = max(0, center_x - sample_size)
        y1 = max(0, center_y - sample_size)
        x2 = min(frame_width, center_x + sample_size + 1)
        y2 = min(frame_height, center_y + sample_size + 1)
        sample_region = frame[y1:y2, x1:x2]
        if sample_region.size == 0:
            return False
        avg_color = np.mean(sample_region, axis=(0, 1))
        grass_color_distance = np.sqrt(np.sum((avg_color - np.array(self.grass_color)) ** 2))
        grass_threshold = 50
        return grass_color_distance < grass_threshold

    def increment_lost_score(self, frame: np.ndarray = None) -> None:
        if frame is not None and self.is_bbox_on_grass(frame):
            self.frames_lost_score += 6
        else:
            self.frames_lost_score += 1


class PlayerTrackerManager:
    def __init__(self, frame_width: int, frame_height: int, grass_color: Tuple[int, int, int]):
        self.trackers: List[PlayerTracker] = []
        self.next_tracker_id = 0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_assignment_distance = 50
        self.initialization_complete = False
        self.grass_color = grass_color

    def initialize_trackers(
        self, team1_bboxes: List[Tuple[int, int, int, int]], team2_bboxes: List[Tuple[int, int, int, int]]
    ) -> None:
        if self.initialization_complete:
            return
        for bbox in team1_bboxes:
            self.trackers.append(PlayerTracker(1, bbox, self.next_tracker_id, self.grass_color))
            self.next_tracker_id += 1
        for bbox in team2_bboxes:
            self.trackers.append(PlayerTracker(2, bbox, self.next_tracker_id, self.grass_color))
            self.next_tracker_id += 1
        self.initialization_complete = True

    def update_trackers_only(
        self,
        team1_bboxes: List[Tuple[int, int, int, int]],
        team2_bboxes: List[Tuple[int, int, int, int]],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        if not self.initialization_complete:
            return
        all_bboxes = [(bbox, 1) for bbox in team1_bboxes] + [(bbox, 2) for bbox in team2_bboxes]
        assignments = self._assign_bboxes_to_trackers(all_bboxes)
        assignments = self._resolve_conflicts(assignments, all_bboxes)
        self._update_tracker_positions(assignments, frame)
        self._cleanup_trackers()

    def update_trackers(
        self,
        team1_bboxes: List[Tuple[int, int, int, int]],
        team2_bboxes: List[Tuple[int, int, int, int]],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        all_bboxes = [(bbox, 1) for bbox in team1_bboxes] + [(bbox, 2) for bbox in team2_bboxes]
        assignments = self._assign_bboxes_to_trackers(all_bboxes)
        assignments = self._resolve_conflicts(assignments, all_bboxes)
        self._update_tracker_positions(assignments, frame)
        self._create_new_trackers(all_bboxes, assignments)
        self._cleanup_trackers()

    def _assign_bboxes_to_trackers(self, all_bboxes: List[Tuple[Tuple[int, int, int, int], int]]) -> dict:
        assignments = {}
        for tracker in self.trackers:
            best_bbox = None
            best_distance = float("inf")
            best_idx = -1
            for idx, (bbox, team_id) in enumerate(all_bboxes):
                if team_id != tracker.team_id:
                    continue
                distance = tracker.calculate_predicted_distance_to_bbox(bbox)
                if distance < best_distance and distance < self.max_assignment_distance:
                    best_distance = distance
                    best_bbox = bbox
                    best_idx = idx
            if best_bbox is not None:
                assignments[tracker.tracker_id] = (best_idx, best_bbox, best_distance)
        return assignments

    def _resolve_conflicts(self, assignments: dict, all_bboxes: List) -> dict:
        bbox_conflicts = {}
        for tracker_id, (bbox_idx, _, distance) in assignments.items():
            bbox_conflicts.setdefault(bbox_idx, []).append((tracker_id, distance))

        resolved_assignments = {}
        losing_trackers = []
        for bbox_idx, competing_trackers in bbox_conflicts.items():
            if len(competing_trackers) == 1:
                tracker_id = competing_trackers[0][0]
                resolved_assignments[tracker_id] = assignments[tracker_id]
            else:
                competing_trackers.sort(key=lambda x: x[1])
                winner_tracker_id = competing_trackers[0][0]
                resolved_assignments[winner_tracker_id] = assignments[winner_tracker_id]
                for tracker_id, _ in competing_trackers[1:]:
                    losing_trackers.append(tracker_id)

        self._assign_unassigned_bboxes_to_losing_trackers(losing_trackers, all_bboxes, resolved_assignments)
        return resolved_assignments

    def _assign_unassigned_bboxes_to_losing_trackers(
        self, losing_tracker_ids: List[int], all_bboxes: List, resolved_assignments: dict
    ) -> None:
        if not losing_tracker_ids:
            return
        assigned_bbox_indices = set()
        for _, (bbox_idx, _, _) in resolved_assignments.items():
            if bbox_idx >= 0:
                assigned_bbox_indices.add(bbox_idx)

        unassigned_bboxes = [
            (idx, bbox, team_id)
            for idx, (bbox, team_id) in enumerate(all_bboxes)
            if idx not in assigned_bbox_indices
        ]

        for tracker_id in losing_tracker_ids:
            tracker = next((t for t in self.trackers if t.tracker_id == tracker_id), None)
            if tracker is None:
                continue
            best_bbox = None
            best_distance = float("inf")
            best_idx = -1
            for idx, bbox, team_id in unassigned_bboxes:
                if team_id != tracker.team_id:
                    continue
                distance = tracker.calculate_predicted_distance_to_bbox(bbox)
                if distance < best_distance and distance < self.max_assignment_distance:
                    best_distance = distance
                    best_bbox = bbox
                    best_idx = idx
            if best_bbox is not None:
                resolved_assignments[tracker_id] = (best_idx, best_bbox, best_distance)
                unassigned_bboxes = [(i, b, t) for i, b, t in unassigned_bboxes if i != best_idx]

    def _update_tracker_positions(self, assignments: dict, frame: Optional[np.ndarray] = None) -> None:
        for tracker in self.trackers:
            if tracker.tracker_id in assignments:
                _, bbox, _ = assignments[tracker.tracker_id]
                tracker.update_bbox(bbox)
            else:
                tracker.increment_lost_score(frame)
                if tracker.is_out_of_bounds(self.frame_width, self.frame_height):
                    tracker.out_of_bounds_frames += 1

    def _create_new_trackers(self, all_bboxes: List, assignments: dict) -> None:
        assigned_indices = set()
        for _, (bbox_idx, _, _) in assignments.items():
            if bbox_idx >= 0:
                assigned_indices.add(bbox_idx)
        for idx, (bbox, team_id) in enumerate(all_bboxes):
            if idx not in assigned_indices:
                self.trackers.append(PlayerTracker(team_id, bbox, self.next_tracker_id, self.grass_color))
                self.next_tracker_id += 1

    def _cleanup_trackers(self) -> None:
        self.trackers = [
            tracker
            for tracker in self.trackers
            if not tracker.should_be_removed(self.frame_width, self.frame_height)
        ]

    def get_all_bboxes(self) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        team1_bboxes = []
        team2_bboxes = []
        for tracker in self.trackers:
            if tracker.team_id == 1:
                team1_bboxes.append(tracker.current_bbox)
            else:
                team2_bboxes.append(tracker.current_bbox)
        return team1_bboxes, team2_bboxes

    def get_all_tracks(self) -> List[Tuple[int, int, Tuple[int, int, int, int]]]:
        return [
            (tracker.tracker_id, tracker.team_id, tracker.current_bbox)
            for tracker in self.trackers
        ]


class BallTracker:
    def __init__(self, initial_position: Tuple[int, int], tracker_id: int, initial_score: int = 100):
        self.tracker_id = tracker_id
        self.position = initial_position
        self.previous_position = initial_position
        self.score = initial_score
        self.dx = 0
        self.dy = 0
        self.frames_since_last_update = 0

    def update_position(self, new_position: Tuple[int, int]) -> None:
        self.previous_position = self.position
        self.position = new_position
        frames_gap = max(1, self.frames_since_last_update + 1)
        self.dx = (new_position[0] - self.previous_position[0]) / frames_gap
        self.dy = (new_position[1] - self.previous_position[1]) / frames_gap
        self.frames_since_last_update = 0

    def update_score_success(self, distance: float) -> None:
        score_increase = max(0, 20 - (distance / 30) * 30)
        self.score = min(300, self.score + score_increase)

    def update_score_failure(self) -> None:
        self.score = max(0, self.score - 5)

    def get_center(self) -> Tuple[int, int]:
        return self.position

    def calculate_predicted_distance_to_position(self, position: Tuple[int, int]) -> float:
        predicted_pos = self.get_predicted_position()
        dx = predicted_pos[0] - position[0]
        dy = predicted_pos[1] - position[1]
        return np.sqrt(dx * dx + dy * dy)

    def get_predicted_position(self) -> Tuple[int, int]:
        prediction_frames = self.frames_since_last_update + 1
        predicted_x = self.position[0] + (self.dx * prediction_frames)
        predicted_y = self.position[1] + (self.dy * prediction_frames)
        return (int(predicted_x), int(predicted_y))

    def increment_frames_since_update(self) -> None:
        self.frames_since_last_update += 1


class BallTrackerManager:
    def __init__(self, frame_width: int, frame_height: int, grass_color: Tuple[int, int, int]):
        self.trackers: List[BallTracker] = []
        self.next_tracker_id = 0
        self.distance_threshold = 30
        self.min_distance_to_player = 25 #이거 주석 처리해서 한번 test 해보자..! 
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grass_color = grass_color
        self.current_ball_bbox = None
        self.frames_lost = 0
        self.max_lost_frames = 30
        self.last_best_position = None

    def update_trackers(self, ball_candidates: List[Tuple[int, int, int, int]], player_positions: List[Tuple[int, int]]):
        ball_centers = []
        for bbox in ball_candidates:
            x, y, w, h = bbox
            ball_centers.append((x + w // 2, y + h // 2))
        filtered_ball_centers = self._filter_candidates_near_players(ball_centers, player_positions)
        self._create_new_trackers(filtered_ball_centers)
        assignments = self._assign_candidates_to_trackers(ball_centers)
        for tracker in self.trackers:
            tracker.increment_frames_since_update()
        self._update_tracker_positions_and_scores(assignments, ball_centers)
        self._cleanup_trackers()
        self._update_last_best_position()

    def _filter_candidates_near_players(  #굳이...? 검토 필요.. 너무 엄격하게 중앙에만 공이 잡히는 현상이 발생한다.
        self, ball_centers: List[Tuple[int, int]], player_positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        filtered_centers = []
        for ball_center in ball_centers:
            too_close_to_player = False
            for player_pos in player_positions:
                distance = self._calculate_distance(ball_center, player_pos)
                if distance < self.min_distance_to_player:
                    too_close_to_player = True
                    break
            if not too_close_to_player:
                filtered_centers.append(ball_center)
        return filtered_centers

    def _create_new_trackers(self, ball_centers: List[Tuple[int, int]]) -> None:
        for center in ball_centers:
            too_close_to_existing = False
            for tracker in self.trackers:
                distance = self._calculate_distance(tracker.get_center(), center)
                if distance < self.distance_threshold:
                    too_close_to_existing = True
                break
            if not too_close_to_existing:
                initial_score = self._calculate_initial_score(center)
                self.trackers.append(BallTracker(center, self.next_tracker_id, initial_score))
                self.next_tracker_id += 1

    def _assign_candidates_to_trackers(self, ball_centers: List[Tuple[int, int]]) -> dict:
        assignments = {}
        used_candidates = set()
        tracker_candidate_distances = []
        for i, tracker in enumerate(self.trackers):
            best_candidate = None
            best_distance = float("inf")
            best_candidate_idx = -1
            for j, candidate in enumerate(ball_centers):
                if j in used_candidates:
                    continue
                distance = tracker.calculate_predicted_distance_to_position(candidate)
                if distance <= self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate
                    best_candidate_idx = j
            if best_candidate is not None:
                tracker_candidate_distances.append((i, best_candidate_idx, best_distance))
        tracker_candidate_distances.sort(key=lambda x: x[2])
        for tracker_idx, candidate_idx, distance in tracker_candidate_distances:
            if candidate_idx not in used_candidates:
                assignments[tracker_idx] = (candidate_idx, distance)
                used_candidates.add(candidate_idx)
        return assignments

    def _update_tracker_positions_and_scores(self, assignments: dict, ball_centers: List[Tuple[int, int]]) -> None:
        for i, tracker in enumerate(self.trackers):
            if i in assignments:
                candidate_idx, distance = assignments[i]
                new_position = ball_centers[candidate_idx]
                tracker.update_position(new_position)
                tracker.update_score_success(distance)
            else:
                tracker.update_score_failure()

    def _cleanup_trackers(self) -> None:
        self.trackers = [tracker for tracker in self.trackers if tracker.score > 0]

    def _update_last_best_position(self) -> None:
        if self.trackers:
            best_tracker = max(self.trackers, key=lambda t: t.score)
            self.last_best_position = best_tracker.get_center()

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx * dx + dy * dy)

    def _calculate_initial_score(self, new_position: Tuple[int, int]) -> int:
        if self.last_best_position is None:
            return 100
        distance = self._calculate_distance(new_position, self.last_best_position)
        max_distance = 150
        min_score = 50
        max_score = 300
        if distance >= max_distance:
            return min_score
        score_range = max_score - min_score
        distance_ratio = distance / max_distance
        squared_ratio = distance_ratio ** 2
        initial_score = max_score - (squared_ratio * score_range)
        return int(initial_score)

    def update_ball_tracking(
        self, ball_candidates: List[Tuple[int, int, int, int]], player_tracker_manager, frame: np.ndarray, frame_count: int
    ) -> None:
        team1_bboxes, team2_bboxes = player_tracker_manager.get_all_bboxes()
        player_positions = []
        for bbox in team1_bboxes + team2_bboxes:
            x, y, w, h = bbox
            player_positions.append((x + w // 2, y + h // 2))
        self.update_trackers(ball_candidates, player_positions)
        best_position = self.get_best_ball_position()
        if best_position is not None:
            x, y = best_position
            self.current_ball_bbox = (x - 5, y - 5, 10, 10) #box 대신 색깔로.. 추후 수정.! 
            self.frames_lost = 0
        else:
            self.current_ball_bbox = None
            self.frames_lost += 1

    def get_best_ball_position(self) -> Optional[Tuple[int, int]]:
        if not self.trackers:
            return None
        best_tracker = max(self.trackers, key=lambda t: t.score)
        return best_tracker.get_center()

    def get_ball_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self.current_ball_bbox


def add_text_to_frame(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def format_mot_line(
    frame_idx: int, track_id: int, bbox: Tuple[int, int, int, int]
) -> str:
    x, y, w, h = bbox
    # MOT format: frame,id,x,y,w,h,conf,-1,-1,-1
    return f"{int(frame_idx)},{int(track_id)},{int(x)},{int(y)},{int(w)},{int(h)},1,-1,-1,-1"


# ==========================================
# 5. 메인 처리 함수 (Grid View Visualization)
# ==========================================

def process_video(
    video_path: str,
    output_path: str,
    pred_output_path: str,
    team1_color_rgb: Optional[Tuple[int, int, int]],
    team2_color_rgb: Optional[Tuple[int, int, int]],
    ball_color_rgb: Optional[Tuple[int, int, int]],
    tracker_debug_mode: bool,
    no_player_tracking: bool,
    show_window: bool,
) -> int:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    is_image_input = Path(video_path).suffix.lower() in image_exts

    cap = None
    out = None

    if is_image_input:
        first_frame = cv2.imread(video_path)
        if first_frame is None:
            print("Error: Could not read input image", file=sys.stderr)
            return 1
        fps = 30
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file", file=sys.stderr)
            return 1

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame", file=sys.stderr)
            return 1

    # 리사이즈 및 잔디 색상 자동 추출
    first_frame = cv2.resize(first_frame, (640, 360))
    frame_height, frame_width = first_frame.shape[:2]
    
    all_mask = np.ones_like(first_frame, dtype=np.uint8) * 255
    dominant_grass = integrate_realtime_colors(first_frame, all_mask, color_space="bgr")
    if dominant_grass is None: dominant_grass = (0, 128, 0)
    print(f"Detected Grass Color (BGR): {dominant_grass}")

    # K-Means 기반 자동 팀 컬러 추출은 비활성화(HVS 고정 필터 사용)
    # if team1_color_rgb is None or team2_color_rgb is None:
    #     team1_color_bgr, team2_color_bgr = detect_team_colors_automatically(first_frame, dominant_grass)
    # else:
    #     team1_color_bgr = (team1_color_rgb[2], team1_color_rgb[1], team1_color_rgb[0])
    #     team2_color_bgr = (team2_color_rgb[2], team2_color_rgb[1], team2_color_rgb[0])

    team1_hsv_filters = [red]
    team2_hsv_filters = [white]
    print(f"Using HSV team filters -> Team1(red): {red}, Team2(white): {white}")

    if ball_color_rgb is None: ball_color_bgr = (255, 255, 255)
    else: ball_color_bgr = (ball_color_rgb[2], ball_color_rgb[1], ball_color_rgb[0])

    # 트래커 매니저 초기화
    tracker_manager = PlayerTrackerManager(frame_width, frame_height, dominant_grass)
    # ball_tracker_manager = BallTrackerManager(frame_width, frame_height, dominant_grass)

    # 출력 설정: 비디오는 VideoWriter, 이미지는 cv2.imwrite
    if not is_image_input:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height * 2))

    if not is_image_input:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    is_first_frame = True
    frame_idx = 0
    mot_pred_lines: List[str] = []
    
    while True:
        if is_image_input:
            if frame_idx > 0:
                break
            frame = first_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break
        frame_idx += 1
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # 1. 잔디 마스크
        lower_green, upper_green = bgr_range(dominant_grass[0], dominant_grass[1], dominant_grass[2], tolerance=60)
        mask_green = cv2.inRange(frame, lower_green, upper_green)
        mask_not_green = cv2.bitwise_not(mask_green)

        # 2. 팀 마스크 생성 (HSV 고정 필터: red/white)
        mask_team1 = create_hsv_mask(frame, team1_hsv_filters)
        mask_team2 = create_hsv_mask(frame, team2_hsv_filters)
        
        # 3. 잔디가 아닌 곳에서 팀 색상 추출
        mask_team1_final = cv2.bitwise_and(mask_not_green, mask_team1)
        mask_team2_final = cv2.bitwise_and(mask_not_green, mask_team2)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_team1_final = cv2.morphologyEx(mask_team1_final, cv2.MORPH_CLOSE, kernel)
        mask_team2_final = cv2.morphologyEx(mask_team2_final, cv2.MORPH_CLOSE, kernel)

        # 4. 박스 검출
        team1_bboxes = get_bounding_boxes(frame, mask_team1_final, dominant_grass)
        team2_bboxes = get_bounding_boxes(frame, mask_team2_final, dominant_grass)
        team1_bboxes = filter_bboxes_by_fill_ratio(team1_bboxes, mask_team1_final, min_fill_ratio=0.08, min_white_pixels=25)
        team2_bboxes = filter_bboxes_by_fill_ratio(team2_bboxes, mask_team2_final, min_fill_ratio=0.08, min_white_pixels=25)

        # 5. 공 검출 및 추적 (주석 처리)
        # all_bboxes = team1_bboxes + team2_bboxes
        # ball_cands = detect_ball(frame, mask_green, ball_color_bgr, player_bboxes=all_bboxes)
        # ball_cands = filter_ball_by_field_position(ball_cands, frame.shape)
        # ball_tracker_manager.update_ball_tracking(ball_cands, tracker_manager, frame, 0)

        # 6. 선수 추적 업데이트 (옵션으로 비활성화 가능)ㅔㅛ
        if no_player_tracking:
            t1_tracked, t2_tracked = team1_bboxes, team2_bboxes
        elif tracker_debug_mode:
            if is_first_frame:
                tracker_manager.initialize_trackers(team1_bboxes, team2_bboxes)
                is_first_frame = False
            else:
                tracker_manager.update_trackers_only(team1_bboxes, team2_bboxes, frame)
            t1_tracked, t2_tracked = tracker_manager.get_all_bboxes()
        else:
            tracker_manager.update_trackers(team1_bboxes, team2_bboxes, frame)
            t1_tracked, t2_tracked = tracker_manager.get_all_bboxes()

        if no_player_tracking:
            all_current = [(1, bbox) for bbox in t1_tracked] + [(2, bbox) for bbox in t2_tracked]
            for local_idx, (_, bbox) in enumerate(all_current):
                synthetic_id = frame_idx * 10000 + local_idx + 1
                mot_pred_lines.append(format_mot_line(frame_idx, synthetic_id, bbox))
        else:
            for tracker_id, _, bbox in tracker_manager.get_all_tracks():
                mot_pred_lines.append(format_mot_line(frame_idx, tracker_id, bbox))

        # 7. 결과 그리기 (Annotated Frame)
        annotated = draw_boxes_on_frame(frame, t1_tracked, (0, 0, 255)) # Red Box
        annotated = draw_boxes_on_frame(annotated, t2_tracked, (255, 0, 0)) # Blue Box
        
        # ball_bbox = ball_tracker_manager.get_ball_bbox()
        # if ball_bbox:
        #     annotated = draw_ball_detection(annotated, [ball_bbox], (0, 255, 255)) # Yellow

        # 8. [Visualization] 4분할 화면 생성
        # 흑백 마스크를 컬러로 변환해야 hstack 가능
        mask_t1_view = cv2.cvtColor(mask_team1_final, cv2.COLOR_GRAY2BGR)
        mask_t2_view = cv2.cvtColor(mask_team2_final, cv2.COLOR_GRAY2BGR)
        mask_green_view = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)
        
        # 텍스트 추가
        mask_t1_view = add_text_to_frame(mask_t1_view, "Team 1 Mask (Auto)")
        mask_t2_view = add_text_to_frame(mask_t2_view, "Team 2 Mask (Auto)")
        mask_green_view = add_text_to_frame(mask_green_view, "Grass Mask")
        annotated = add_text_to_frame(annotated, "Final Tracking")
        
        # 상단: 팀1, 팀2 마스크
        top_row = np.hstack((mask_t1_view, mask_t2_view))
        # 하단: 잔디 마스크, 최종 결과
        bottom_row = np.hstack((mask_green_view, annotated))
        # 전체 합치기
        combined_output = np.vstack((top_row, bottom_row))
        
        if is_image_input:
            cv2.imwrite(output_path, combined_output)
        else:
            out.write(combined_output)
        if show_window:
            cv2.imshow("Soccer Tracking Process", combined_output)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    if cap is not None:
        cap.release()
    if out is not None:
        out.release()
    if show_window: cv2.destroyAllWindows()
    with open(pred_output_path, "w", encoding="utf-8") as f:
        if mot_pred_lines:
            f.write("\n".join(mot_pred_lines) + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto Team Color Detection Soccer Tracker")
    parser.add_argument("input", help="Input video path.")
    parser.add_argument("output", help="Output video path.")
    parser.add_argument("--pred-output", default="pred.txt", help="Path to save MOT-format predictions.")
    # [수정됨] 색상 입력이 선택 사항(Optional)으로 변경됨
    parser.add_argument("--team1-color", nargs=3, type=int, required=False, metavar=("R", "G", "B"), help="Optional")
    parser.add_argument("--team2-color", nargs=3, type=int, required=False, metavar=("R", "G", "B"), help="Optional")
    parser.add_argument("--ball-color", nargs=3, type=int, metavar=("R", "G", "B"))
    parser.add_argument("--tracker-debug", action="store_true")
    parser.add_argument("--no-player-tracking", action="store_true")
    parser.add_argument("--show-window", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return process_video(
        args.input,
        args.output,
        args.pred_output,
        tuple(args.team1_color) if args.team1_color else None,
        tuple(args.team2_color) if args.team2_color else None,
        tuple(args.ball_color) if args.ball_color else None,
        args.tracker_debug,
        args.no_player_tracking,
        args.show_window,
    )


if __name__ == "__main__":
    raise SystemExit(main())
