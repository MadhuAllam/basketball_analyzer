# Four court corners -> homography into meters. Auto path is cheap OpenCV; swap in a keypoint net if you have one.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CourtConfig:
    length_m: float
    width_m: float
    manual_corners_pixel: Optional[list] = None

    @classmethod
    def from_json(cls, path: str | Path) -> CourtConfig:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        mc = data.get("manual_corners_pixel")
        return cls(
            length_m=float(data["length_m"]),
            width_m=float(data["width_m"]),
            manual_corners_pixel=mc if mc else None,
        )


class CourtCornerDetector:
    def detect(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Wood-ish floor; tweak if your broadcast looks nothing like this.
        mask = cv2.inRange(hsv, np.array([5, 30, 50], np.uint8), np.array([45, 255, 255], np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < (h * w) * 0.05:
            return None
        peri = cv2.arcLength(cnt, True)
        approx = None
        for eps_frac in (0.02, 0.03, 0.04, 0.05, 0.07):
            a = cv2.approxPolyDP(cnt, eps_frac * peri, True)
            if len(a) == 4:
                approx = a
                break
        if approx is None:
            hull = cv2.convexHull(cnt)
            peri_h = cv2.arcLength(hull, True)
            a = cv2.approxPolyDP(hull, 0.02 * peri_h, True)
            if len(a) >= 4:
                approx = a[:4]
            else:
                return None
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) != 4:
            return None
        return self._order_corners(pts)

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


class CourtHomography:
    def __init__(self, court: CourtConfig):
        self._court = court
        self._H: Optional[np.ndarray] = None
        self._H_inv: Optional[np.ndarray] = None

    @property
    def is_ready(self) -> bool:
        return self._H is not None

    def compute_from_corners(
        self, corners_px: np.ndarray, frame_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        L, W = self._court.length_m, self._court.width_m
        dst = np.array([[0.0, 0.0], [L, 0.0], [L, W], [0.0, W]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(corners_px.astype(np.float32), dst)
        self._H = H
        self._H_inv = np.linalg.inv(H)
        return H

    def try_init(self, frame_bgr: np.ndarray, detector: CourtCornerDetector) -> bool:
        if self._court.manual_corners_pixel:
            corners = np.array(self._court.manual_corners_pixel, dtype=np.float32)
            if corners.shape == (4, 2):
                self.compute_from_corners(corners, frame_bgr.shape)
                return True
        corners = detector.detect(frame_bgr)
        if corners is None:
            return False
        self.compute_from_corners(corners, frame_bgr.shape)
        return True

    def pixel_to_meters(self, xy: np.ndarray) -> np.ndarray:
        if self._H is None:
            raise RuntimeError("Homography not initialized")
        xy = np.atleast_2d(xy).astype(np.float64)
        ones = np.ones((xy.shape[0], 1), dtype=np.float64)
        hom = np.hstack([xy, ones]).T
        wp = self._H @ hom
        wp /= wp[2] + 1e-9
        return np.stack([wp[0], wp[1]], axis=1)

    def meters_to_pixel(self, XY: np.ndarray) -> np.ndarray:
        if self._H_inv is None:
            raise RuntimeError("Homography not initialized")
        XY = np.atleast_2d(XY).astype(np.float64)
        ones = np.ones((XY.shape[0], 1), dtype=np.float64)
        hom = np.hstack([XY, ones]).T
        pp = self._H_inv @ hom
        pp /= pp[2] + 1e-9
        return np.stack([pp[0], pp[1]], axis=1)
