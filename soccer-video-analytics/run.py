import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass


def expand_bbox_centered(
    bbox_xyxy: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    scale: float = 2.0,
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox_xyxy
    frame_h, frame_w = frame_shape[:2]

    width = max(1, xmax - xmin)
    height = max(1, ymax - ymin)
    center_x = xmin + width / 2.0
    center_y = ymin + height / 2.0

    expanded_w = max(1, int(round(width * scale)))
    expanded_h = max(1, int(round(height * scale)))

    new_xmin = int(round(center_x - expanded_w / 2.0))
    new_ymin = int(round(center_y - expanded_h / 2.0))
    new_xmax = int(round(center_x + expanded_w / 2.0))
    new_ymax = int(round(center_y + expanded_h / 2.0))

    new_xmin = max(0, min(new_xmin, frame_w - 1))
    new_ymin = max(0, min(new_ymin, frame_h - 1))
    new_xmax = max(0, min(new_xmax, frame_w - 1))
    new_ymax = max(0, min(new_ymax, frame_h - 1))

    if new_xmax <= new_xmin:
        new_xmax = min(frame_w - 1, new_xmin + 1)
    if new_ymax <= new_ymin:
        new_ymax = min(frame_h - 1, new_ymin + 1)

    return (new_xmin, new_ymin, new_xmax, new_ymax)


def is_bbox_near_frame_edge(
    bbox_xyxy: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    margin_ratio: float = 0.04,
    min_margin_px: int = 20,
) -> bool:
    xmin, ymin, xmax, ymax = bbox_xyxy
    frame_h, frame_w = frame_shape[:2]

    margin_x = max(min_margin_px, int(round(frame_w * margin_ratio)))
    margin_y = max(min_margin_px, int(round(frame_h * margin_ratio)))

    return (
        xmin <= margin_x
        or ymin <= margin_y
        or xmax >= frame_w - margin_x
        or ymax >= frame_h - margin_y
    )


def classify_team_from_roi_hsv(
    frame_bgr: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    team_filters: Dict[str, List[dict]],
) -> Tuple[Optional[str], float]:
    xmin, ymin, xmax, ymax = bbox_xyxy
    roi = frame_bgr[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        return None, 0.0

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    best_team = None
    best_score = 0.0
    roi_area = float(max(1, roi.shape[0] * roi.shape[1]))

    for team_name, colors in team_filters.items():
        merged_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
        for color in colors:
            lower = np.array(color["lower_hsv"], dtype=np.uint8)
            upper = np.array(color["upper_hsv"], dtype=np.uint8)
            merged_mask = cv2.bitwise_or(
                merged_mask, cv2.inRange(roi_hsv, lower, upper)
            )

        non_black_pixels = float(cv2.countNonZero(merged_mask))
        fill_ratio = non_black_pixels / roi_area
        if fill_ratio > best_score:
            best_score = fill_ratio
            best_team = team_name

    return best_team, best_score


def detect_team_boxes_in_roi_hsv(
    frame_bgr: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    team_filters: Dict[str, List[dict]],
    min_box_area: int = 5,
) -> Tuple[Optional[str], float, List[Tuple[int, int, int, int, str]]]:
    xmin, ymin, xmax, ymax = bbox_xyxy
    frame_h, frame_w = frame_bgr.shape[:2]
    roi = frame_bgr[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        return None, 0.0, []

    target_w, target_h = 640, 360
    scale_x = target_w / float(max(1, frame_w))
    scale_y = target_h / float(max(1, frame_h))

    sxmin = max(0, min(int(round(xmin * scale_x)), target_w - 1))
    symin = max(0, min(int(round(ymin * scale_y)), target_h - 1))
    sxmax = max(0, min(int(round(xmax * scale_x)), target_w - 1))
    symax = max(0, min(int(round(ymax * scale_y)), target_h - 1))

    if sxmax <= sxmin:
        sxmax = min(target_w - 1, sxmin + 1)
    if symax <= symin:
        symax = min(target_h - 1, symin + 1)

    scaled_roi_w = max(1, sxmax - sxmin)
    scaled_roi_h = max(1, symax - symin)
    resized_roi = cv2.resize(
        roi,
        (scaled_roi_w, scaled_roi_h),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[symin:symin + scaled_roi_h, sxmin:sxmin + scaled_roi_w] = resized_roi

    canvas_hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    scaled_roi_area = float(max(1, scaled_roi_w * scaled_roi_h))

    best_team = None
    best_score = 0.0
    all_boxes_with_team: List[Tuple[int, int, int, int, str]] = []

    for team_name, colors in team_filters.items():
        merged_mask = np.zeros(canvas_hsv.shape[:2], dtype=np.uint8)
        for color in colors:
            lower = np.array(color["lower_hsv"], dtype=np.uint8)
            upper = np.array(color["upper_hsv"], dtype=np.uint8)
            merged_mask = cv2.bitwise_or(
                merged_mask, cv2.inRange(canvas_hsv, lower, upper)
            )

        merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel)
        merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel)
        score = float(cv2.countNonZero(merged_mask)) / scaled_roi_area

        contours, _ = cv2.findContours(
            merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes_abs: List[Tuple[int, int, int, int, str]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_box_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 1 or h <= 1:
                continue

            # Keep detections in or intersecting the scaled ROI pasted region.
            cx = x + w / 2.0
            cy = y + h / 2.0
            inside_scaled_roi = (
                sxmin <= cx <= sxmin + scaled_roi_w and symin <= cy <= symin + scaled_roi_h
            )
            if not inside_scaled_roi:
                continue

            ox1 = int(round(x / scale_x))
            oy1 = int(round(y / scale_y))
            ox2 = int(round((x + w) / scale_x))
            oy2 = int(round((y + h) / scale_y))

            ox1 = max(0, min(ox1, frame_w - 1))
            oy1 = max(0, min(oy1, frame_h - 1))
            ox2 = max(0, min(ox2, frame_w - 1))
            oy2 = max(0, min(oy2, frame_h - 1))

            if ox2 <= ox1:
                ox2 = min(frame_w - 1, ox1 + 1)
            if oy2 <= oy1:
                oy2 = min(frame_h - 1, oy1 + 1)

            boxes_abs.append((ox1, oy1, ox2, oy2, team_name))

        if score > best_score:
            best_score = score
            best_team = team_name
        all_boxes_with_team.extend(boxes_abs)

    return best_team, best_score, all_boxes_with_team


parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--image",
    type=str,
    default=None,
    help="Path to a single input image to process instead of video",
)
parser.add_argument(
    "--image-output",
    type=str,
    default="../test/image_output.jpg",
    help="Path to save processed single-image output",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
parser.add_argument(
    "--dump-missing-id-frames",
    action="store_true",
    help="Save frames where tracked player IDs disappear compared to previous frame",
)
parser.add_argument(
    "--missing-id-dir",
    default="debug_v9/missing_ids",
    type=str,
    help="Directory where missing-ID frames are stored",
)
args = parser.parse_args()

if args.dump_missing_id_frames:
    os.makedirs(args.missing_id_dir, exist_ok=True)
    raw_missing_id_dir = os.path.join(args.missing_id_dir, "raw_frames")
    annotated_missing_id_dir = os.path.join(args.missing_id_dir, "annotated_frames")
    missing_id_log_path = os.path.join(args.missing_id_dir, "missing_ids_bbox.csv")
    missing_id_txt_log_path = os.path.join(args.missing_id_dir, "missing_ids_bbox.txt")
    os.makedirs(raw_missing_id_dir, exist_ok=True)
    os.makedirs(annotated_missing_id_dir, exist_ok=True)
    missing_id_log_file = open(missing_id_log_path, "w", encoding="utf-8")
    missing_id_txt_log_file = open(missing_id_txt_log_path, "w", encoding="utf-8")
    missing_id_log_file.write(
        "event_frame,last_seen_frame,missing_id,last_xmin,last_ymin,last_xmax,last_ymax,expanded_xmin,expanded_ymin,expanded_xmax,expanded_ymax,width,height,recovered_team,recovered_score,recovered_box_count,recovered_boxes,raw_frame_path,annotated_frame_path\n"
    )
else:
    raw_missing_id_dir = None
    annotated_missing_id_dir = None
    missing_id_log_file = None
    missing_id_txt_log_file = None

image_mode = args.image is not None

if image_mode:
    input_image = cv2.imread(args.image)
    if input_image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    fps = 30
else:
    video = Video(input_path=args.video)
    fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Object Detectors
player_detector = YoloV5()
ball_detector = YoloV5(model_path=args.model)

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match
chelsea = Team(
    name="red team",
    abbreviation="CHE",
    color=(255, 0, 0),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
man_city = Team(
    name="white team",
    abbreviation="MNC",
    color=(255, 255, 255),
    board_color=(235, 235, 235),
    text_color=(0, 0, 0),
)
teams = [chelsea, man_city]
match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city
valid_team_names = {team.name for team in teams}
team_filters = {
    filter_cfg["name"]: filter_cfg["colors"]
    for filter_cfg in filters
    if filter_cfg["name"] in valid_team_names
}

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=0 if image_mode else 3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=0 if image_mode else 20,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()
previous_player_ids = set()
previous_player_bboxes = {}

frames = [(0, input_image)] if image_mode else enumerate(video)

for i, frame in frames:
    original_frame = frame.copy()

    # Get Detections
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    current_player_ids = {
        int(detection.data["id"])
        for detection in player_detections
        if detection is not None and "id" in detection.data
    }
    current_player_bboxes = {}
    for detection in player_detections:
        if detection is None or "id" not in detection.data:
            continue
        player_id = int(detection.data["id"])
        x1, y1 = detection.points[0]
        x2, y2 = detection.points[1]
        xmin = int(round(min(x1, x2)))
        ymin = int(round(min(y1, y2)))
        xmax = int(round(max(x1, x2)))
        ymax = int(round(max(y1, y2)))
        current_player_bboxes[player_id] = (xmin, ymin, xmax, ymax)

    missing_player_ids = sorted(previous_player_ids - current_player_ids)
    valid_missing_player_ids = []
    missing_fallback_entries = []

    for missing_id in missing_player_ids:
        last_bbox = previous_player_bboxes.get(missing_id)
        if last_bbox is None:
            continue
        if is_bbox_near_frame_edge(last_bbox, frame.shape):
            continue

        expanded_bbox = expand_bbox_centered(
            last_bbox, frame_shape=frame.shape, scale=2.5
        )
        recovered_team, recovered_score, recovered_boxes = detect_team_boxes_in_roi_hsv(
            frame_bgr=frame,
            bbox_xyxy=expanded_bbox,
            team_filters=team_filters,
        )

        team_obj = Team.from_name(teams=teams, name=recovered_team)
        if team_obj is None:
            draw_color = (0, 255, 255)
            recovered_team_name = "unknown"
        else:
            draw_color = team_obj.color
            recovered_team_name = team_obj.name

        missing_fallback_entries.append(
            {
                "id": missing_id,
                "last_bbox": last_bbox,
                "expanded_bbox": expanded_bbox,
                "recovered_team": recovered_team_name,
                "score": recovered_score,
                "draw_color": draw_color,
                "recovered_boxes": recovered_boxes,
            }
        )
        valid_missing_player_ids.append(missing_id)

    if args.dump_missing_id_frames and valid_missing_player_ids:
        debug_frame = original_frame.copy()
        missing_ids_text = ",".join(
            str(missing_id) for missing_id in valid_missing_player_ids
        )
        cv2.putText(
            debug_frame,
            f"frame={i} missing_ids={missing_ids_text}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        output_path = os.path.join(
            raw_missing_id_dir,
            f"frame_{i:06d}_missing_{'-'.join(map(str, valid_missing_player_ids))}.jpg",
        )
        cv2.imwrite(output_path, debug_frame)
        raw_output_path = output_path
    else:
        raw_output_path = ""

    previous_player_ids = current_player_ids
    previous_player_bboxes_for_log = previous_player_bboxes
    previous_player_bboxes = current_player_bboxes

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=player_detections, teams=teams)
    match.update(players, ball)

    # Draw
    frame = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if args.possession:
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )

        frame = match.draw_possession_counter(
            frame, counter_background=possession_background, debug=False
        )

        if ball:
            frame = ball.draw(frame)
    elif image_mode:
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )
        if ball:
            frame = ball.draw(frame)

    if args.passes:
        pass_list = match.passes

        frame = Pass.draw_pass_list(
            img=frame, passes=pass_list, coord_transformations=coord_transformations
        )

        frame = match.draw_passes_counter(
            frame, counter_background=passes_background, debug=False
        )

    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    for entry in missing_fallback_entries:
        ex1, ey1, ex2, ey2 = entry["expanded_bbox"]
        color = entry["draw_color"]
        # Draw ROI box (expanded region) and per-mask detections together.
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), color, 2)
        cv2.putText(
            frame,
            f"ROI id={entry['id']}",
            (ex1, max(20, ey1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        for bx1, by1, bx2, by2, det_team in entry["recovered_boxes"]:
            det_team_obj = Team.from_name(teams=teams, name=det_team)
            det_color = det_team_obj.color if det_team_obj is not None else color
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), det_color, 2)
            cv2.putText(
                frame,
                f"DET {det_team}",
                (bx1, max(20, by1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                det_color,
                1,
                cv2.LINE_AA,
            )

    if args.dump_missing_id_frames and valid_missing_player_ids:
        annotated_debug_frame = frame.copy()
        missing_ids_text = ",".join(
            str(missing_id) for missing_id in valid_missing_player_ids
        )
        cv2.putText(
            annotated_debug_frame,
            f"frame={i} missing_ids={missing_ids_text}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        annotated_output_path = os.path.join(
            annotated_missing_id_dir,
            f"frame_{i:06d}_missing_{'-'.join(map(str, valid_missing_player_ids))}.jpg",
        )
        cv2.imwrite(annotated_output_path, annotated_debug_frame)

        for missing_id in valid_missing_player_ids:
            bbox = previous_player_bboxes_for_log.get(missing_id)
            if bbox is None:
                continue

            matching_entry = next(
                (entry for entry in missing_fallback_entries if entry["id"] == missing_id),
                None,
            )
            if matching_entry is None:
                continue

            xmin, ymin, xmax, ymax = matching_entry["last_bbox"]
            exmin, eymin, exmax, eymax = matching_entry["expanded_bbox"]
            width = xmax - xmin
            height = ymax - ymin
            recovered_boxes = matching_entry["recovered_boxes"]
            recovered_boxes_str = "|".join(
                [
                    f"{bx1}:{by1}:{bx2}:{by2}:{det_team}"
                    for bx1, by1, bx2, by2, det_team in recovered_boxes
                ]
            )
            missing_id_log_file.write(
                f"{i},{i-1},{missing_id},{xmin},{ymin},{xmax},{ymax},{exmin},{eymin},{exmax},{eymax},{width},{height},{matching_entry['recovered_team']},{matching_entry['score']:.6f},{len(recovered_boxes)},{recovered_boxes_str},{raw_output_path},{annotated_output_path}\n"
            )
            missing_id_txt_log_file.write(
                " ".join(
                    [
                        f"event_frame={i}",
                        f"last_seen_frame={i-1}",
                        f"id={missing_id}",
                        f"last_bbox=({xmin},{ymin},{xmax},{ymax})",
                        f"expanded_bbox=({exmin},{eymin},{exmax},{eymax})",
                        f"team={matching_entry['recovered_team']}",
                        f"score={matching_entry['score']:.6f}",
                        f"boxes={recovered_boxes_str if recovered_boxes_str else 'none'}",
                    ]
                )
                + "\n"
            )

    if image_mode:
        image_output_dir = os.path.dirname(args.image_output)
        if image_output_dir:
            os.makedirs(image_output_dir, exist_ok=True)
        cv2.imwrite(args.image_output, frame)
    else:
        # Write video
        video.write(frame)

if missing_id_log_file is not None:
    missing_id_log_file.close()
if missing_id_txt_log_file is not None:
    missing_id_txt_log_file.close()
