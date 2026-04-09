# src/tracking/player_manager.py
import numpy as np

class PlayerManager:
    """
    Wraps raw ByteTrack output and enforces stable 2-player identity.
    
    - Limits to max 2 players at all times
    - Assigns permanent labels "A" and "B" based on court half
    - Re-identifies players after brief tracking loss using position
    """

    def __init__(self, court_roi, max_players=2, reidentify_dist=120):
        self.max_players      = max_players
        self.reidentify_dist  = reidentify_dist  # px radius to re-identify same player
        self.court_roi        = court_roi

        # Stable players: label → {last_bbox, last_center, track_id, color}
        self.players = {}   # "A" or "B" → player dict
        self.colors  = {
            "A": (0, 255, 100),    # green
            "B": (255, 100, 0),    # blue
        }

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _foot_point(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    def _dist(self, c1, c2):
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def _court_half(self, bbox):
        """Returns 'top' or 'bottom' based on average Y of court ROI."""
        ys = [p[1] for p in self.court_roi]
        mid_y = (min(ys) + max(ys)) / 2
        _, y1, _, y2 = bbox
        center_y = (y1 + y2) / 2
        return "top" if center_y < mid_y else "bottom"

    def _assign_initial_labels(self, track_a, track_b):
        """
        On first detection of 2 players, assign A=top-half, B=bottom-half.
        This keeps labels geographically stable.
        """
        half_a = self._court_half(track_a["bbox"])
        half_b = self._court_half(track_b["bbox"])

        if half_a == "top":
            self.players["A"] = self._make_player(track_a, "A")
            self.players["B"] = self._make_player(track_b, "B")
        else:
            self.players["A"] = self._make_player(track_b, "A")
            self.players["B"] = self._make_player(track_a, "B")

    def _make_player(self, track, label):
        return {
            "label":       label,
            "track_id":    track["track_id"],
            "bbox":        track["bbox"],
            "center":      self._bbox_center(track["bbox"]),
            "foot":        self._foot_point(track["bbox"]),
            "confidence":  track["confidence"],
            "color":       self.colors[label],
            "lost_frames": 0,
        }

    def _try_reidentify(self, track):
        """
        Match an incoming ByteTrack track to an existing stable player
        using proximity. Returns label "A"/"B" or None.
        """
        best_label = None
        best_dist  = self.reidentify_dist  # threshold

        center = self._bbox_center(track["bbox"])
        for label, player in self.players.items():
            d = self._dist(center, player["center"])
            if d < best_dist:
                best_dist  = d
                best_label = label

        return best_label

    def update(self, raw_tracks):
        """
        Takes raw ByteTrack output, returns stable 2-player list.

        Each returned player: {label, bbox, center, foot, confidence, color}
        """
        if not raw_tracks:
            # Mark all players as lost
            for label in self.players:
                self.players[label]["lost_frames"] += 1
            return list(self.players.values())

        # Step 1: Keep only top-N by confidence
        candidates = sorted(raw_tracks, key=lambda t: t["confidence"], reverse=True)
        candidates = candidates[:self.max_players]

        # Step 2: Initial assignment when we first see 2 players
        if not self.players and len(candidates) >= 2:
            self._assign_initial_labels(candidates[0], candidates[1])
            return list(self.players.values())

        if not self.players and len(candidates) == 1:
            # Only 1 player visible at start, assign as A temporarily
            self.players["A"] = self._make_player(candidates[0], "A")
            return list(self.players.values())

        # Step 3: Re-identify each candidate against known players
        used_labels = set()
        for track in candidates:
            label = self._try_reidentify(track)

            if label and label not in used_labels:
                # Update existing stable player
                self.players[label].update({
                    "track_id":   track["track_id"],
                    "bbox":       track["bbox"],
                    "center":     self._bbox_center(track["bbox"]),
                    "foot":       self._foot_point(track["bbox"]),
                    "confidence": track["confidence"],
                    "lost_frames": 0,
                })
                used_labels.add(label)

            elif label is None:
                # New player appeared — assign to whichever slot is empty/lost most
                available = [l for l in ("A", "B") if l not in used_labels]
                if available:
                    assign_to = available[0]
                    # Prefer the slot whose player has been lost longest
                    if len(available) == 2:
                        assign_to = max(available,
                            key=lambda l: self.players.get(l, {}).get("lost_frames", 999))
                    self.players[assign_to] = self._make_player(track, assign_to)
                    used_labels.add(assign_to)

        # Step 4: Increment lost counter for unmatched players
        for label in self.players:
            if label not in used_labels:
                self.players[label]["lost_frames"] += 1

        return list(self.players.values())

    def get_player(self, label):
        return self.players.get(label)