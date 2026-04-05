import google.generativeai as genai
import threading
import cv2
import random
import numpy as np
from PIL import Image

class CommentaryEngine:
    def __init__(
        self,
        fast_break_speed_mps=6.0,
        rotation_pass_threshold=3,
        fps=30.0,
        display_seconds=1.5, # 45 frames expiry at 30fps
        gemini_api_key=None,
        gemini_interval_seconds=1.5,
    ):
        self.fps = max(fps, 1.0)
        self.display_frames = 45 # User requested 45 frame cycle (~1.5s)
        self.fast_break_speed = fast_break_speed_mps
        
        # State tracking for triggers
        self._last_poss = "unknown"
        self._last_zone = "unknown"
        self._last_shot_ev = False
        self._frame_count = 0
        
        # Pass counting
        self._pa = 0
        self._pb = 0

        self.gemini_interval_frames = 45 # Continuous 45-frame (1.5s) injection
        
        # Gemini
        self.gemini_enabled = False
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            self.gemini_enabled = True

        self._gemini_latest = ""
        self._gemini_running = False
        self._gemini_timer = 0
        self._gemini_history = [] # History of last 20 responses
        self._gemini_call_count = 0 # Track first 3 for synchronous start
        
        self._fallback_phrases = [] # Cleared for AI-only mode
        self._last_fallback = ""
        self._frames_since_gemini = 0
        
        # Display
        self._current_text = "Game analysis starting..."
        self._current_ttl = 999 
        self._queue = []
        self._last_shown_text = ""
        self._cooldown_frames = int(fps * 6.0) # 6 second silence between same trigger

    def _is_too_similar(self, text):
        if not text: return False
        new_words = set(text.lower().replace(".", "").replace("!", "").split())
        if not new_words: return False
        for old_text in self._gemini_history[-20:]:
            old_words = set(old_text.lower().replace(".", "").replace("!", "").split())
            overlap = new_words.intersection(old_words)
            if len(overlap) > len(new_words) * 0.7:
                return True
        return False

    def _generate_gemini_text(self, frame_bgr, max_spd, events):
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ev_str = ", ".join([str(e.kind) for e in events if e.kind is not None]) or "None"

            # Attempt up to 3 times for a non-similar response
            for _ in range(3):
                prompt = f"""You are a live basketball TV commentator. 
Watch this frame and describe concisely what is happening - ball handler, player movement, and defensive stance. 
Maximum 10 words. Be specific to what you SEE. 
DO NOT repeat previous commentary: '{self._last_shown_text}'
Telemetry: Speed {max_spd:.1f} m/s, Events: {ev_str}, Zone: {self._last_zone}
"""
                print(f"[Gemini] Calling API at frame {self._frame_count}")
                response = self.gemini_model.generate_content([prompt, pil_img])
                print(f"[Gemini] Response: {response.text[:50]}")
                text = response.text.strip().replace('"','').replace("'","")
                if text and not self._is_too_similar(text):
                    # Ensure strict 10-word limit if Gemini overshoots
                    words = text.split()
                    if len(words) > 10:
                        text = " ".join(words[:10]) + "."
                    self._gemini_history.append(text)
                    if len(self._gemini_history) > 20: self._gemini_history.pop(0)
                    return text
            return "" # fallback to empty if still too similar
        except Exception:
            return ""

    def _call_gemini_sync(self, frame_bgr, max_spd, events):
        text = self._generate_gemini_text(frame_bgr, max_spd, events)
        if text: self._gemini_latest = text

    def _call_gemini_async(self, frame_bgr, max_spd, events):
        def _run():
            try:
                self._gemini_running = True
                text = self._generate_gemini_text(frame_bgr, max_spd, events)
                if text: self._gemini_latest = text
            finally: self._gemini_running = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _get_zone(self, b_m):
        if not b_m: return "unknown"
        x, y = b_m
        if abs(x) < 2.5 and y < 5.0: return "paint"
        if y > 14.0: return "backcourt"
        return "wing"

    def on_frame(self, possessor_team, max_player_speed_mps, events, b_m=None, frame_bgr=None):
        self._last_zone = self._get_zone(b_m)
        self._frame_count += 1
        
        # Bootstrap Trigger: Call Gemini at very first frame
        if self._frame_count == 1 and self.gemini_enabled:
             self._call_gemini_async(frame_bgr, max_player_speed_mps, events)

        # CONTINUOUS 30-FRAME INJECTION
        if self.gemini_enabled:
            self._gemini_timer += 1
            if self._gemini_timer >= self.gemini_interval_frames:
                self._gemini_timer = 0
                self._gemini_call_count += 1
                
                # First 3 calls are synchronous for immediate HUD population
                if self._gemini_call_count <= 3:
                    self._call_gemini_sync(frame_bgr, max_player_speed_mps, events)
                elif not self._gemini_running:
                    self._call_gemini_async(frame_bgr, max_player_speed_mps, events)

        # Action: Update display if fresh Gemini insights arrive
        if self._gemini_latest:
            if self._gemini_latest != self._last_shown_text:
                self._current_text = self._gemini_latest
                self._last_shown_text = self._gemini_latest
                self._current_ttl = 60 # Set display duration
            self._gemini_latest = ""
            self._frames_since_gemini = 0
        else:
            self._frames_since_gemini += 1

        # Action: Advance display time
        if self._current_ttl > 0:
            self._current_ttl -= 1
        
        # Action: Infinite display persistence guard
        if self.gemini_enabled and self._current_ttl <= 0:
            self._current_ttl = 45 # Renew for another 1.5 seconds

        # Action: Tracking stats (Passes)
        for e in events:
            if e.kind == "pass":
                if e.team_to == "Team White": self._pa += 1
                if e.team_to == "Team Blue": self._pb += 1

        line0 = self._current_text if self._current_text else ""
        line1 = "" # Restricted stats line
        
        return [line0, line1]
