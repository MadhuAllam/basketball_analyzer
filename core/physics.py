# Metric court: who has the ball, passes vs turnovers, speeds, and short "glue" when the detector drops the ball.

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from core.homography import CourtHomography


@dataclass
class PhysicsConfig:
    possession_radius_m: float = 1.5
    ball_lost_max_frames: int = 10
    ball_history_frames: int = 5
    release_velocity_mps: float = 12.0
    hand_attach_radius_m: float = 1.2


@dataclass
class PlayerState:
    track_id: int
    team: str
    position_m: Tuple[float, float]
    velocity_mps: float = 0.0


@dataclass
class PossessionEvent:
    kind: str  # pass | interception
    team_from: str
    team_to: str
    track_from: Optional[int] = None
    track_to: Optional[int] = None


class PhysicsEngine:
    def __init__(
        self,
        homography: "CourtHomography",
        cfg: PhysicsConfig,
        fps: float = 30.0,
    ):
        self.H = homography
        self.cfg = cfg
        self.fps = max(fps, 1e-3)
        self._prev_pos: Dict[int, Tuple[float, float]] = {}
        self._possessor_tid: Optional[int] = None
        self._possessor_team: str = "unknown"
        self._ball_lost_frames: int = 0
        self._anchored_tid: Optional[int] = None
        self._ball_history_m: Deque[Tuple[float, float]] = deque(
            maxlen=max(cfg.ball_history_frames, 2)
        )
        self._last_ball_m: Optional[Tuple[float, float]] = None
        self._frame_idx: int = 0

    def reset(self) -> None:
        self._prev_pos.clear()
        self._possessor_tid = None
        self._possessor_team = "unknown"
        self._ball_lost_frames = 0
        self._anchored_tid = None
        self._ball_history_m.clear()
        self._last_ball_m = None
        self._frame_idx = 0

    def _dist_m(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _nearest_player_to_ball(
        self,
        ball_m: Tuple[float, float],
        players: List[PlayerState],
    ) -> Optional[Tuple[PlayerState, float]]:
        best: Optional[Tuple[PlayerState, float]] = None
        for p in players:
            d = self._dist_m(ball_m, p.position_m)
            if best is None or d < best[1]:
                best = (p, d)
        return best

    def update_players_speed(
        self,
        track_ids: np.ndarray,
        team_by_id: Dict[int, str],
        positions_m: np.ndarray,
    ) -> List[PlayerState]:
        dt = 1.0 / self.fps
        out: List[PlayerState] = []
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            pos = (float(positions_m[i, 0]), float(positions_m[i, 1]))
            team = team_by_id.get(tid, "unknown")
            v = 0.0
            if tid in self._prev_pos:
                p0 = self._prev_pos[tid]
                d = self._dist_m(p0, pos)
                v = d / dt
            self._prev_pos[tid] = pos
            out.append(PlayerState(track_id=tid, team=team, position_m=pos, velocity_mps=v))
        self._frame_idx += 1
        return out

    def resolve_ball_meters(
        self,
        ball_center_px: Optional[Tuple[float, float]],
        ball_detected: bool,
        players: List[PlayerState],
    ) -> Tuple[Optional[Tuple[float, float]], bool]:
        used_anchor = False
        if not self.H.is_ready:
            return None, False

        if ball_detected and ball_center_px is not None:
            bm = self.H.pixel_to_meters(np.array([[ball_center_px[0], ball_center_px[1]]]))[0]
            bmt = (float(bm[0]), float(bm[1]))
            self._last_ball_m = bmt
            self._ball_history_m.append(bmt)
            self._ball_lost_frames = 0
            self._anchored_tid = None
            return bmt, False

        self._ball_lost_frames += 1

        if self._ball_lost_frames >= self.cfg.ball_lost_max_frames:
            self._anchored_tid = None
            return self._last_ball_m, used_anchor

        if self._ball_lost_frames < self.cfg.ball_lost_max_frames and players:
            hist = list(self._ball_history_m)
            if len(hist) >= 2:
                vx = hist[-1][0] - hist[-2][0]
                vy = hist[-1][1] - hist[-2][1]
                pred = (hist[-1][0] + vx, hist[-1][1] + vy)
            elif self._last_ball_m is not None:
                pred = self._last_ball_m
            else:
                pred = None
            ref = pred if pred is not None else self._last_ball_m
            if ref is None:
                return None, False

            nearest = self._nearest_player_to_ball(ref, players)
            if nearest and nearest[1] <= self.cfg.hand_attach_radius_m:
                p = nearest[0]
                self._anchored_tid = p.track_id
                used_anchor = True
                return p.position_m, True

        if self._last_ball_m is not None:
            return self._last_ball_m, False
        return None, False

    def possession_and_events(
        self,
        ball_m: Optional[Tuple[float, float]],
        players: List[PlayerState],
        ball_velocity_mps: float,
    ) -> Tuple[Optional[int], str, List[PossessionEvent]]:
        events: List[PossessionEvent] = []
        if ball_m is None or not players:
            return self._possessor_tid, self._possessor_team, events

        possessor: Optional[Tuple[int, str, float]] = None
        for p in players:
            d = self._dist_m(ball_m, p.position_m)
            if d <= self.cfg.possession_radius_m:
                if possessor is None or d < possessor[2]:
                    possessor = (p.track_id, p.team, d)

        if self._anchored_tid is not None:
            for p in players:
                if p.track_id == self._anchored_tid:
                    possessor = (p.track_id, p.team, 0.0)
                    break

        if possessor is None:
            if ball_velocity_mps > self.cfg.release_velocity_mps:
                if self._possessor_tid is not None:
                    events.append(
                        PossessionEvent(
                            kind="shot",
                            team_from=self._possessor_team,
                            team_to="unknown",
                            track_from=self._possessor_tid,
                        )
                    )
                self._possessor_tid = None
                self._possessor_team = "unknown"
            return self._possessor_tid, self._possessor_team, events

        tid, team, _ = possessor
        prev_tid = self._possessor_tid
        prev_team = self._possessor_team

        if prev_tid is not None and prev_tid != tid:
            if prev_team in ("Team White", "Team Blue") and team in ("Team White", "Team Blue"):
                if prev_team == team:
                    events.append(
                        PossessionEvent(
                            kind="pass",
                            team_from=prev_team,
                            team_to=team,
                            track_from=prev_tid,
                            track_to=tid,
                        )
                    )
                else:
                    events.append(
                        PossessionEvent(
                            kind="interception",
                            team_from=prev_team,
                            team_to=team,
                            track_from=prev_tid,
                            track_to=tid,
                        )
                    )

        self._possessor_tid = tid
        self._possessor_team = team if team in ("Team White", "Team Blue") else self._possessor_team
        return tid, self._possessor_team, events


def ball_speed_mps(
    homography: "CourtHomography",
    prev_m: Optional[Tuple[float, float]],
    curr_m: Optional[Tuple[float, float]],
    fps: float,
) -> float:
    if prev_m is None or curr_m is None or not homography.is_ready:
        return 0.0
    d = float(np.hypot(curr_m[0] - prev_m[0], curr_m[1] - prev_m[1]))
    return d * fps
