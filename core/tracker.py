# Ultralytics track() with ByteTrack; separate Kalman on the ball in pixel space.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import cv2
from ultralytics import YOLO

from core.detector import COCO_PERSON, COCO_SPORTS_BALL

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

class ByteTrackRunner:
    def __init__(
        self,
        weights: str = "models/best.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        imgsz: int = 480,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        self.device = device or _device()
        self.imgsz = imgsz
        self.tracker_cfg = tracker_cfg
        
        import os
        custom_cfg = os.path.join(
            os.path.dirname(__file__), "..", "configs", "bytetrack_custom.yaml"
        )
        if os.path.exists(custom_cfg):
            self.tracker_cfg = custom_cfg
        
        try:
            self.model = YOLO(weights)
        except Exception as e:
            print(f"[Tracker] Failed to load {weights}, falling back to yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            
        self.model.to(self.device)
        
        self.court_top_ratio = 0.20 # Added 20% hard limit to prevent ceiling triangle
        self.court_center_x_min = 0.25
        self.court_center_x_max = 0.75
        
        self.ball_kalman = cv2.KalmanFilter(4, 2)
        self.ball_kalman.measurementMatrix = np.array(
            [[1,0,0,0],[0,1,0,0]], np.float32)
        self.ball_kalman.transitionMatrix = np.array(
            [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.ball_kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1 # Increased for fast ball
        self.ball_kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.ball_kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.ball_kalman.errorCovPre = np.eye(4, dtype=np.float32)
        self.ball_kalman_initialized = False
        self.ball_lost_frames = 0
        self.BALL_MAX_LOST = 10 # Re-detection window

    def reset_ball_kalman(self) -> None:
        self.ball_kalman_initialized = False
        self.ball_lost_frames = 0

    def _get_orange_ball(self, frame_bgr, player_boxes=None) -> Optional[np.ndarray]:
        # Apply CLAHE to L channel of LAB space for consistent lighting
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (5, 50, 50), (25, 255, 255))
        mask2 = cv2.inRange(hsv, (0, 60, 100), (10, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        
        valid_cnts = []
        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if 80 <= area <= 8000 and perimeter > 0:
                circ = 4 * np.pi * area / (perimeter ** 2)
                if circ > 0.4: # Relaxed from 0.6 to catch blurred/fast movement
                    valid_cnts.append((circ, cnt))
                    
        if not valid_cnts:
            return None
            
        # Pick the most circular contour
        _, best_cnt = max(valid_cnts, key=lambda x: x[0])
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.4 or aspect > 2.5:
            return None  # not round enough
            
        # Ball must be smaller than player boxes
        ball_area = bw * bh
        if ball_area > 15000:
            return None  # too large — it's a person not a ball
            
        ball_cx = x + bw // 2
        ball_cy = y + bh // 2
        
        # Dropped rigid player box intersection logic because it blocks dribbling/shooting frames
        # The area and circularity filters are strict enough to reject jerseys without this check.

        cx, cy, side = ball_cx, ball_cy, max(bw, bh)//2
        fh, fw = frame_bgr.shape[:2]
        
        return np.array([
            max(0, cx - side), 
            max(0, cy - side), 
            min(fw, cx + side), 
            min(fh, cy + side)
        ], dtype=np.float32)

    def process_frame(
        self, frame_bgr: np.ndarray, persist: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, float]], bool]:
        
        # SINGLE YOLO EXECUTION
        results = self.model.track(
            frame_bgr, persist=persist, tracker=self.tracker_cfg,
            imgsz=self.imgsz, conf=0.10, iou=0.45, verbose=False,
            device=self.device, agnostic_nms=True, classes=[COCO_PERSON, COCO_SPORTS_BALL]
        )[0]

        p_xy, p_ids = [], []
        raw_ball_box = None
        ball_conf = 0.0

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            if results.boxes.id is not None:
                ids = results.boxes.id.cpu().numpy().astype(int)
            else:
                ids = [-1] * len(boxes)
                
            for box, cls, cid, conf in zip(boxes, clss, ids, confs):
                if int(cls) == COCO_PERSON and conf >= 0.10:
                    p_xy.append(box)
                    p_ids.append(cid)
        # Step 1: YOLO Track -> check if ball class found
        raw_ball_box = None
        ball_conf = 0.0
        confirmed_by_detector = False

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, clss, confs):
                if int(cls) == COCO_SPORTS_BALL:
                    if conf > ball_conf:
                        raw_ball_box = box.astype(np.float32)
                        ball_conf = float(conf)
                        confirmed_by_detector = True

        # Step 2: HSV Fallback (if YOLO failed)
        if not confirmed_by_detector:
            hsv_box = self._get_orange_ball(frame_bgr, player_boxes=p_xy)
            if hsv_box is not None:
                raw_ball_box = hsv_box
                confirmed_by_detector = True

        # Step 3: Kalman Prediction (if still None)
        ball_is_predicted = False
        if not confirmed_by_detector:
            self.ball_lost_frames += 1
            if self.ball_kalman_initialized and self.ball_lost_frames <= self.BALL_MAX_LOST:
                # Damped prediction
                self.ball_kalman.statePre[2:] *= 0.5 
                pred = self.ball_kalman.predict()
                px, py = float(pred[0, 0]), float(pred[1, 0])
                
                fh, fw = frame_bgr.shape[:2]
                if py < fh * self.court_top_ratio:  # reject anything above 20% height
                    self.ball_kalman_initialized = False
                    raw_ball_box = None
                    ball_is_predicted = False
                else:
                    raw_ball_box = np.array([px - 10, py - 10, px + 10, py + 10], dtype=np.float32)
                    ball_is_predicted = True
            else:
                self.ball_kalman_initialized = False

        # Step 4: Boundary Filter
        if raw_ball_box is not None:
            _cy_check = (raw_ball_box[1] + raw_ball_box[3]) / 2
            if _cy_check < frame_bgr.shape[0] * 0.15:
                raw_ball_box = None

        ball_box = None
        if raw_ball_box is not None:
            fh, fw = frame_bgr.shape[:2]
            cx, cy = (raw_ball_box[0] + raw_ball_box[2]) / 2.0, (raw_ball_box[1] + raw_ball_box[3]) / 2.0
            
            # Change 3: Specialized 15% height reject logic
            if cy < fh * 0.15:
                ball_box = None
                self.ball_kalman_initialized = False
            else:
                x1, y1, x2, y2 = raw_ball_box
                reject = False
                if cy < fh * self.court_top_ratio and not (fw * self.court_center_x_min <= cx <= fw * self.court_center_x_max):
                    reject = True
                if cy > fh * 0.95 or cx < fw * 0.02 or cx > fw * 0.98:
                    reject = True
                    
                if not reject:
                    if ball_is_predicted and cy < fh * 0.25: # Prevent Kalman drift to top
                        ball_box = None
                        ball_is_predicted = False
                        self.ball_kalman_initialized = False
                    else:
                        ball_box = raw_ball_box

        # Step 5: Kalman Update
        smooth_center = None
        if ball_box is not None:
            cx, cy = (ball_box[0] + ball_box[2]) / 2.0, (ball_box[1] + ball_box[3]) / 2.0
            if confirmed_by_detector:
                measured = np.array([[np.float32(cx)], [np.float32(cy)]])
                if not self.ball_kalman_initialized:
                    self.ball_kalman.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                    self.ball_kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                    self.ball_kalman_initialized = True
                self.ball_kalman.correct(measured)
                self.ball_lost_frames = 0
            
            # Extract smoothed position
            if self.ball_kalman_initialized:
                state = self.ball_kalman.statePost
                smooth_center = (float(state[0, 0]), float(state[1, 0]))
            else:
                smooth_center = (float(cx), float(cy))

        p_xy_arr = np.stack(p_xy) if p_xy else None
        p_ids_arr = np.array(p_ids, dtype=np.int32) if p_ids else None

        return p_xy_arr, p_ids_arr, ball_box, smooth_center, ball_is_predicted
