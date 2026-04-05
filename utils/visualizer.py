# Mini-map in the corner + chunky HUD text.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_court_minimap(
    canvas_hw: Tuple[int, int],
    length_m: float,
    width_m: float,
    player_positions_m: List[Tuple[float, float, str]],
    ball_m: Optional[Tuple[float, float]],
    team_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    h, w = canvas_hw[0], canvas_hw[1]
    img = np.ones((h, w, 3), dtype=np.uint8) * 40

    margin = 8
    scale_x = (w - 2 * margin) / length_m
    scale_y = (h - 2 * margin) / width_m
    scale = min(scale_x, scale_y)

    def to_px(X: float, Y: float) -> Tuple[int, int]:
        px = int(margin + X * scale)
        py = int(margin + Y * scale)
        return px, py

    p00 = to_px(0, 0)
    pL0 = to_px(length_m, 0)
    pLL = to_px(length_m, width_m)
    p0W = to_px(0, width_m)
    cv2.rectangle(img, p00, pLL, (200, 200, 200), 1)
    cv2.polylines(img, [np.array([p00, pL0, pLL, p0W])], True, (180, 180, 180), 1)
    cv2.line(img, to_px(length_m / 2, 0), to_px(length_m / 2, width_m), (120, 120, 120), 1)

    tc = team_colors or {
        "Team A": (80, 160, 240),
        "Team B": (240, 120, 80),
        "unknown": (180, 180, 180),
    }

    for Xm, Ym, team in player_positions_m:
        color = tc.get(team, tc["unknown"])
        c = to_px(Xm, Ym)
        cv2.circle(img, c, 5, color, -1)

    if ball_m is not None:
        bx, by = to_px(ball_m[0], ball_m[1])
        cv2.circle(img, (bx, by), 4, (0, 255, 255), -1)

    cv2.putText(
        img,
        "Top-down",
        (6, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return img


def overlay_minimap(
    frame_bgr: np.ndarray,
    minimap: np.ndarray,
    corner: str = "br",
    margin: int = 12,
) -> np.ndarray:
    fh, fw = frame_bgr.shape[:2]
    mh, mw = minimap.shape[:2]
    scale = min(0.28 * fw / mw, 0.28 * fh / mh, 1.0)
    tw, th = int(mw * scale), int(mh * scale)
    small = cv2.resize(minimap, (tw, th), interpolation=cv2.INTER_AREA)
    out = frame_bgr.copy()
    y1 = fh - th - margin if corner.startswith("b") else margin
    x1 = fw - tw - margin if corner.endswith("r") else margin
    y2, x2 = y1 + th, x1 + tw
    roi = out[y1:y2, x1:x2]
    if roi.shape[:2] == small.shape[:2]:
        blended = cv2.addWeighted(roi, 0.25, small, 0.75, 0)
        out[y1:y2, x1:x2] = blended
    return out


def draw_hud_text(
    frame_bgr: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int] = (16, 32),
) -> None:
    y = origin[1]
    for line in lines[:6]:
        cv2.putText(
            frame_bgr,
            line[:120],
            (origin[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line[:120],
            (origin[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 22
