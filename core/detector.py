# YOLO for boxes, KMeans for Jersey classification (Blue vs White).

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

COCO_PERSON = 2
COCO_SPORTS_BALL = 0

def load_court_settings(path: str | Path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)

class JerseyClassifier:
    """Classifies players into Blue vs White teams using KMeans on the torso region."""
    def __init__(self):
        # Reference colors in BGR
        self.ref_white = np.array([210, 210, 210], dtype=np.float32)
        self.ref_blue = np.array([180, 100, 40], dtype=np.float32) # Standard dark blue

    def classify_crop(self, crop_bgr: np.ndarray) -> str:
        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 10: return "unknown"
        
        # Focus on Torso (middle 30-70% height, 20-80% width)
        y1, y2 = int(h * 0.3), int(h * 0.7)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        torso = crop_bgr[y1:y2, x1:x2]
        if torso.size == 0: return "unknown"

        # KMeans to find dominant colors (k=3 to separate jersey, skin, floor)
        pixels = torso.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        
        best_team = "unknown"
        min_dist = 1000.0
        
        for center in centers:
            dist_w = np.linalg.norm(center - self.ref_white)
            dist_b = np.linalg.norm(center - self.ref_blue)
            
            # Check distance to White
            if dist_w < min_dist and dist_w < 90:
                min_dist = dist_w
                best_team = "Team White"
            # Check distance to Blue
            if dist_b < min_dist and dist_b < 130:
                min_dist = dist_b
                best_team = "Team Blue"
                
        return best_team

    def classify_track_crops(
        self,
        frame_bgr: np.ndarray,
        player_xyxy: np.ndarray,
        track_ids: np.ndarray,
        cache: Dict[int, str],
    ) -> Dict[int, str]:
        """Runs KMeans classification once per ID and locks it."""
        for xy, tid in zip(player_xyxy, track_ids):
            tid = int(tid)
            # LOCK LOGIC: Only classify if not already in cache or if unknown
            if tid in cache and cache[tid] != "unknown":
                continue
                
            x1, y1, x2, y2 = (int(v) for v in xy)
            h, w = frame_bgr.shape[:2]
            crop = frame_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            
            team = self.classify_crop(crop)
            if team != "unknown":
                cache[tid] = team
            elif tid not in cache:
                cache[tid] = "unknown"
        return cache
