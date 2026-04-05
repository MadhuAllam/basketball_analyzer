import argparse, sys, cv2
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core.detector import JerseyClassifier, load_court_settings
from core.homography import CourtCornerDetector, CourtHomography, CourtConfig
from core.physics import PhysicsConfig, PhysicsEngine, ball_speed_mps
from core.tracker import ByteTrackRunner
from utils.commentary_engine import CommentaryEngine
from utils.visualizer import draw_court_minimap, draw_hud_text, overlay_minimap

def is_on_court(box, h, w):
    x1, y1, x2, y2 = box
    # Relaxed filtering to detect EVERY player requested by user
    # Keeping only a minimal 2% border to avoid camera-edge artifacts
    # Removed the 25% top filter to catch background players
    if (y1 + y2) / 2 > h * 0.98: return False
    if (y1 + y2) / 2 < h * 0.02: return False
    if (x1 + x2) / 2 < w * 0.01 or (x1 + x2) / 2 > w * 0.99: return False
    return True

def get_pro_box(box):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    # Force 1:2 aspect ratio (Height = 2.0 * Width)
    new_w = bh * 0.5
    return [cx - new_w/2, y1, cx + new_w/2, y2]

def draw_commentary(frame, lines):
    if not lines or not lines[0]: return
    h, w = frame.shape[:2]
    font_scale = 0.8
    thickness = 2
    
    y0_pos = h - 60
    y1_pos = h - 110
    
    # Box for line 1 (Stats/Possession) - Only draw if not empty
    if len(lines) > 1 and lines[1]:
        (w1, h1), _ = cv2.getTextSize(lines[1], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x1 = (w - w1) // 2
        cv2.rectangle(frame, (x1 - 15, y0_pos - h1 - 10), (x1 + w1 + 15, y0_pos + 10), (0, 0, 0), -1)
        cv2.putText(frame, lines[1], (x1, y0_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Box for line 0 (Commentary)
    max_w = int(w * 0.8)
    msg = lines[0]
    (w0, h0), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if w0 > max_w:
        msg = msg[:45] + "..."
        (w0, h0), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    x0 = (w - w0) // 2
    cv2.rectangle(frame, (x0 - 15, y1_pos - h0 - 10), (x0 + w0 + 15, y1_pos + 10), (0, 0, 0), -1)
    cv2.putText(frame, msg, (x0, y1_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

def run_pipeline(
    video_path: str,
    output_path: str,
    config_path: str,
    weights: str = "models/best.pt",
    imgsz: int = 480,
    clip_every: int = 20,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> None:
    cfg = load_court_settings(config_path)
    ct_cfg = CourtConfig.from_json(config_path)
    ph_cfg = PhysicsConfig(
        possession_radius_m=float(cfg["possession_radius_m"]),
        ball_lost_max_frames=int(cfg["ball_lost_max_frames"]),
        ball_history_frames=int(cfg["ball_history_frames"]),
        release_velocity_mps=float(cfg["release_velocity_mps"]),
        hand_attach_radius_m=float(cfg["hand_attach_radius_m"]),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames: total_frames = min(total_frames, max_frames)
    w, h = int(cap.get(3)), int(cap.get(4))
    
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    homography = CourtHomography(ct_cfg)
    detector = CourtCornerDetector()
    tracker = ByteTrackRunner(weights=weights, imgsz=imgsz)
    classifier = JerseyClassifier() # KMeans
    physics = PhysicsEngine(homography, ph_cfg, fps=fps)
    commentary = CommentaryEngine(
        fps=fps,
        gemini_api_key="AIzaSyC-fj9FCbsxPbb0CjcbsqFcn_UM099xZUo",
    )

    cache, ids_map = {}, {}
    prev_ball_m, frame_i = None, 0
    colors_map = {"Team Blue": (240, 120, 80), "Team White": (225, 225, 225), "unknown": (180, 180, 180)}

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or (max_frames and frame_i >= max_frames): break

            if frame_i == 0 and not homography.try_init(frame, detector):
                homography._H = np.array([[0.02, 0, 0], [0, 0.02, 0], [0, 0, 1]], dtype=np.float32)

            # 1. Detect & Track (conf=0.5 handled in tracker results loop)
            p_xy, p_ids, b_xy, b_sm, b_is_pred = tracker.process_frame(frame)
            b_px = (float(b_xy[0]+b_xy[2])/2, float(b_xy[1]+b_xy[3])/2) if b_xy is not None else None

            # 2. Team Classification (Jersey KMeans)
            if p_xy is not None:
                cache = classifier.classify_track_crops(frame, p_xy, p_ids, cache)

            # 3. Physics
            plist, p_metrics = [], []
            if homography.is_ready and p_xy is not None:
                centers = np.stack([(p_xy[:,0]+p_xy[:,2])/2, (p_xy[:,1]+p_xy[:,3])/2], axis=1)
                pm = homography.pixel_to_meters(centers)
                plist = physics.update_players_speed(p_ids, cache, pm)
                p_metrics = [(p.position_m[0], p.position_m[1], p.team) for p in plist]

            b_m = None
            if homography.is_ready:
                b_m, _ = physics.resolve_ball_meters(b_px, b_xy is not None, plist)

            b_vel = ball_speed_mps(homography, prev_ball_m, b_m, fps)
            prev_ball_m = b_m
            _, poss, evs = physics.possession_and_events(b_m, plist, b_vel)
            
            # 4. SMAAAART Commentary
            max_spd = max([p.velocity_mps for p in plist]) if plist else 0.0
            hud_lines = commentary.on_frame(poss, max_spd, evs, b_m=b_m, frame_bgr=frame)

            # 5. Draw
            vis = frame.copy()
            
            # Professional Header Label
            cv2.putText(vis, "AUTO-ANALYZER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vis, "AUTO-ANALYZER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 1, cv2.LINE_AA)
            
            if p_xy is not None:
                # REMOVED redundant NMS - ByteTrack already handles this
                # This ensures EVERY player gets their own box as requested.
                for i in range(len(p_xy)):
                    box, pid = p_xy[i], p_ids[i]
                    if not is_on_court(box, h, w): continue # Spatial Filter
                    
                    # Tighten to 1:2 Aspect Ratio
                    x1, y1, x2, y2 = map(int, get_pro_box(box))
                    
                    team = cache.get(pid, "unknown")
                    c = colors_map.get(team, colors_map["unknown"])
                    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
                    if pid not in ids_map: ids_map[pid] = len(ids_map) + 1
                    cv2.putText(vis, f"{ids_map[pid]}", (x1, max(0, y1-5)), 1, 0.7, c, 1)

            # Ball drawing logic (New Green Triangle)
            if b_xy is not None and not b_is_pred:
                bx1, by1, bx2, by2 = map(int, b_xy)
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                tri_size = 18
                pts = np.array([
                    [bcx,              bcy + tri_size],
                    [bcx - tri_size,   bcy - tri_size],
                    [bcx + tri_size,   bcy - tri_size]
                ], np.int32)
                cv2.fillPoly(vis, [pts], (0, 220, 0))
                cv2.polylines(vis, [pts], True, (255, 255, 255), 2)
            
            elif b_xy is not None and b_is_pred:
                bx1, by1, bx2, by2 = map(int, b_xy)
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                tri_size = 14
                pts = np.array([
                    [bcx,              bcy + tri_size],
                    [bcx - tri_size,   bcy - tri_size],
                    [bcx + tri_size,   bcy - tri_size]
                ], np.int32)
                cv2.polylines(vis, [pts], True, (180, 180, 180), 2)

            # 4. Commentary (Drawn absolute last for HUD clarity)
            draw_commentary(vis, hud_lines)
            draw_hud_text(vis, [f"Frame {frame_i}"])
            
            if homography.is_ready:
                mm = draw_court_minimap((200, 120), ct_cfg.length_m, ct_cfg.width_m, p_metrics, b_m)
                vis = overlay_minimap(vis, mm, "br")

            writer.write(vis)
            frame_i += 1
            if progress_callback and frame_i % 20 == 0:
                progress_callback(0.05 + 0.90 * (frame_i / total_frames), f"{frame_i}/{total_frames}")

    finally:
        cap.release()
        writer.release()
        if progress_callback: progress_callback(1.0, "Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="result.mp4")
    p.add_argument("--config", default="configs/court_config.json")
    p.add_argument("--max_frames", type=int, default=None)
    a = p.parse_args()
    run_pipeline(a.video, a.out, a.config, max_frames=a.max_frames)
