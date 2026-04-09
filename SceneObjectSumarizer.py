import math


class SceneObjectsSummarizer:
    def __init__(
            self,
            min_confidence=0.3,
            min_track_length=5,
            max_center_distance=220.0,
            area_change_threshold=0.15,
            movement_threshold=500.0,
            max_missing_frames=100,
            interaction_iou_threshold=0.05,
            interaction_center_threshold=80.0,
            min_shared_frames_for_interaction=3,
            min_close_ratio_for_interaction=0.5,
            strong_interaction_min_shared_frames=5,
            strong_interaction_min_close_ratio=0.65,
            strong_interaction_min_same_direction=0.5,
            strong_interaction_min_iou_ratio=0.2,
    ):
        self.min_confidence = min_confidence
        self.min_track_length = min_track_length
        self.max_center_distance = max_center_distance
        self.area_change_threshold = area_change_threshold
        self.movement_threshold = movement_threshold
        self.max_missing_frames = max_missing_frames

        self.interaction_iou_threshold = interaction_iou_threshold
        self.interaction_center_threshold = interaction_center_threshold
        self.min_shared_frames_for_interaction = min_shared_frames_for_interaction
        self.min_close_ratio_for_interaction = min_close_ratio_for_interaction

        self.strong_interaction_min_shared_frames = strong_interaction_min_shared_frames
        self.strong_interaction_min_close_ratio = strong_interaction_min_close_ratio
        self.strong_interaction_min_same_direction = strong_interaction_min_same_direction
        self.strong_interaction_min_iou_ratio = strong_interaction_min_iou_ratio

    def summarize_scene_objects(self, scene_objects):
        filtered = self._filter_low_confidence(scene_objects)
        tracks = self._build_tracks(filtered)
        tracks = self._remove_short_tracks(tracks)
        tracks = self._merge_fragmented_tracks(tracks)
        interactions = self._detect_interactions(tracks)

        if not tracks:
            return "Scene summary: no reliable objects detected."

        lines = ["Scene summary:"]

        for track in tracks:
            line = self._track_to_text(track)
            if line:
                lines.append(f"- {line}")

        if interactions:
            lines.append("Possible interactions:")
            for interaction in interactions:
                lines.append(f"- {self._interaction_to_text(interaction)}")

        return "\n".join(lines)

    def build_scene_analysis(self, scene_objects):
        """
        Если нужен не только текст, но и структура:
        {
            "tracks": [...],
            "interactions": [...]
        }
        """
        filtered = self._filter_low_confidence(scene_objects)
        tracks = self._build_tracks(filtered)
        tracks = self._remove_short_tracks(tracks)
        tracks = self._merge_fragmented_tracks(tracks)
        interactions = self._detect_interactions(tracks)

        return {
            "tracks": tracks,
            "interactions": interactions,
        }

    def _filter_low_confidence(self, scene_objects):
        result = {}

        for frame_id, detections in scene_objects.items():
            valid = []
            for det in detections:
                confidence = det.get("confidence", 0.0)
                if confidence >= self.min_confidence:
                    valid.append(det)
            result[frame_id] = valid

        return result

    def _build_tracks(self, scene_objects):
        frames = sorted(scene_objects.keys())
        active_tracks = []
        finished_tracks = []
        next_track_id = 0

        for frame_id in frames:
            detections = scene_objects[frame_id]
            used_detection_indices = set()
            new_active_tracks = []

            for track in active_tracks:
                best_idx = None
                best_distance = float("inf")

                last_det = track["detections"][-1]

                for i, det in enumerate(detections):
                    if i in used_detection_indices:
                        continue
                    if det["object"] != last_det["object"]:
                        continue

                    dist = self._bbox_center_distance(last_det["bbox"], det["bbox"])
                    if dist < best_distance and dist <= self.max_center_distance:
                        best_distance = dist
                        best_idx = i

                if best_idx is not None:
                    matched_det = detections[best_idx]
                    track["detections"].append(matched_det)
                    track["frames"].append(frame_id)
                    track["missing_count"] = 0
                    used_detection_indices.add(best_idx)
                    new_active_tracks.append(track)
                else:
                    track["missing_count"] += 1
                    if track["missing_count"] <= self.max_missing_frames:
                        new_active_tracks.append(track)
                    else:
                        finished_tracks.append(track)

            for i, det in enumerate(detections):
                if i in used_detection_indices:
                    continue

                track = {
                    "id": next_track_id,
                    "object": det["object"],
                    "frames": [frame_id],
                    "detections": [det],
                    "missing_count": 0,
                }
                next_track_id += 1
                new_active_tracks.append(track)

            active_tracks = new_active_tracks

        finished_tracks.extend(active_tracks)

        for track in finished_tracks:
            track.pop("missing_count", None)

        return finished_tracks

    def _remove_short_tracks(self, tracks):
        return [t for t in tracks if len(t["frames"]) >= self.min_track_length]

    def _merge_fragmented_tracks(self, tracks):
        if not tracks:
            return []

        tracks = sorted(tracks, key=lambda t: (t["object"], t["frames"][0]))
        merged = []
        used = [False] * len(tracks)

        for i in range(len(tracks)):
            if used[i]:
                continue

            current = {
                "id": tracks[i]["id"],
                "object": tracks[i]["object"],
                "frames": list(tracks[i]["frames"]),
                "detections": list(tracks[i]["detections"]),
            }
            used[i] = True

            changed = True
            while changed:
                changed = False

                for j in range(len(tracks)):
                    if used[j]:
                        continue

                    candidate = tracks[j]

                    if current["object"] != candidate["object"]:
                        continue

                    frame_gap = candidate["frames"][0] - current["frames"][-1]
                    if frame_gap < 1 or frame_gap > (self.max_missing_frames + 1):
                        continue

                    last_bbox = current["detections"][-1]["bbox"]
                    next_bbox = candidate["detections"][0]["bbox"]
                    dist = self._bbox_center_distance(last_bbox, next_bbox)

                    if dist <= self.max_center_distance:
                        current["frames"].extend(candidate["frames"])
                        current["detections"].extend(candidate["detections"])
                        used[j] = True
                        changed = True

            merged.append(current)

        return merged

    def _detect_interactions(self, tracks):
        interactions = []

        for track in tracks:
            track["frame_to_detection"] = {
                frame_id: det
                for frame_id, det in zip(track["frames"], track["detections"])
            }

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                t1 = tracks[i]
                t2 = tracks[j]

                shared_frames = sorted(set(t1["frames"]) & set(t2["frames"]))
                if len(shared_frames) < self.min_shared_frames_for_interaction:
                    continue

                stats = self._compute_pair_interaction_stats(t1, t2, shared_frames)
                if stats is None:
                    continue

                interaction_type = self._infer_interaction_type(t1, t2, stats, shared_frames)
                if interaction_type is None:
                    continue

                interactions.append({
                    "type": interaction_type,
                    "track_1_id": t1["id"],
                    "track_1_object": t1["object"],
                    "track_2_id": t2["id"],
                    "track_2_object": t2["object"],
                    "shared_frames": shared_frames,
                    "shared_frame_ranges": self._format_frame_ranges(shared_frames),
                    "shared_frame_count": len(shared_frames),
                    "iou_hits": stats["iou_hits"],
                    "close_hits": stats["close_hits"],
                    "close_ratio": stats["close_hits"] / max(1, len(shared_frames)),
                    "avg_iou": stats["avg_iou"],
                    "avg_center_distance": stats["avg_center_distance"],
                    "same_direction_ratio": stats["same_direction_ratio"],
                    "confidence": self._interaction_confidence(stats, interaction_type, len(shared_frames)),
                })

        for track in tracks:
            track.pop("frame_to_detection", None)

        return interactions

    def _compute_pair_interaction_stats(self, t1, t2, shared_frames):
        iou_hits = 0
        close_hits = 0
        iou_values = []
        center_distances = []

        for frame_id in shared_frames:
            det1 = t1["frame_to_detection"][frame_id]
            det2 = t2["frame_to_detection"][frame_id]

            bbox1 = det1["bbox"]
            bbox2 = det2["bbox"]

            iou = self._bbox_iou(bbox1, bbox2)
            dist = self._bbox_center_distance(bbox1, bbox2)

            iou_values.append(iou)
            center_distances.append(dist)

            if iou >= self.interaction_iou_threshold:
                iou_hits += 1
            if dist <= self.interaction_center_threshold:
                close_hits += 1

        if not center_distances:
            return None

        same_direction_ratio = self._pair_same_direction_ratio(t1, t2)

        return {
            "iou_hits": iou_hits,
            "close_hits": close_hits,
            "avg_iou": sum(iou_values) / len(iou_values),
            "avg_center_distance": sum(center_distances) / len(center_distances),
            "same_direction_ratio": same_direction_ratio,
        }

    def _infer_interaction_type(self, t1, t2, stats, shared_frames):
        shared_count = len(shared_frames)
        close_ratio = stats["close_hits"] / max(1, shared_count)
        iou_ratio = stats["iou_hits"] / max(1, shared_count)
        same_direction = stats["same_direction_ratio"]

        # сильная интеракция: долго рядом, часто пересекаются/сильно близки, двигаются похоже
        if (
                shared_count >= self.strong_interaction_min_shared_frames
                and close_ratio >= self.strong_interaction_min_close_ratio
                and same_direction >= self.strong_interaction_min_same_direction
                and (iou_ratio >= self.strong_interaction_min_iou_ratio or stats[
            "avg_center_distance"] <= self.interaction_center_threshold * 0.7)
        ):
            return "strong joint interaction"

        # обычная интеракция: есть устойчивое пространственное соседство
        if (
                shared_count >= self.min_shared_frames_for_interaction
                and (
                stats["iou_hits"] > 0
                or close_ratio >= self.min_close_ratio_for_interaction
        )
        ):
            return "possible interaction"

        return None

    def _interaction_confidence(self, stats, interaction_type, shared_count):
        score = 0.0

        score += min(shared_count / 10.0, 1.0)
        score += min(stats["avg_iou"] * 3.0, 1.0)
        score += min(stats["same_direction_ratio"], 1.0)

        if stats["avg_center_distance"] <= self.interaction_center_threshold:
            score += 0.7

        if interaction_type == "strong joint interaction":
            score += 0.5

        if score >= 2.4:
            return "high"
        if score >= 1.5:
            return "medium"
        return "low"

    def _pair_same_direction_ratio(self, t1, t2):
        dir1 = self._track_motion_vector(t1)
        dir2 = self._track_motion_vector(t2)

        mag1 = math.sqrt(dir1[0] ** 2 + dir1[1] ** 2)
        mag2 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        dot = dir1[0] * dir2[0] + dir1[1] * dir2[1]
        cos_sim = dot / (mag1 * mag2)

        return max(0.0, cos_sim)

    def _track_motion_vector(self, track):
        detections = track["detections"]
        if len(detections) < 2:
            return (0.0, 0.0)

        first_bbox = detections[0]["bbox"]
        last_bbox = detections[-1]["bbox"]

        dx = self._bbox_center_x(last_bbox) - self._bbox_center_x(first_bbox)
        dy = self._bbox_center_y(last_bbox) - self._bbox_center_y(first_bbox)

        return (dx, dy)

    def _interaction_to_text(self, interaction):
        t = interaction["type"]
        obj1 = interaction["track_1_object"]
        id1 = interaction["track_1_id"]
        obj2 = interaction["track_2_object"]
        id2 = interaction["track_2_id"]
        frames = interaction["shared_frame_ranges"]
        conf = interaction["confidence"]

        return (
            f"{t} between {obj1} track {id1} and {obj2} track {id2}; "
            f"shared frames {frames}; "
            f"close in {interaction['close_hits']} frames; "
            f"overlap in {interaction['iou_hits']} frames; "
            f"confidence {conf}"
        )

    def _track_to_text(self, track):
        obj_name = track["object"]
        detections = track["detections"]
        frames = sorted(track["frames"])

        if not detections:
            return None

        first_bbox = detections[0]["bbox"]
        last_bbox = detections[-1]["bbox"]

        dx = self._bbox_center_x(last_bbox) - self._bbox_center_x(first_bbox)
        dy = self._bbox_center_y(last_bbox) - self._bbox_center_y(first_bbox)

        first_area = self._bbox_area(first_bbox)
        last_area = self._bbox_area(last_bbox)

        movement_x = self._describe_horizontal_motion(dx)
        movement_y = self._describe_vertical_motion(dy)
        depth_motion = self._describe_depth_motion(first_area, last_area)
        position = self._describe_position(first_bbox, last_bbox)

        frame_ranges_text = self._format_frame_ranges(frames)

        parts = [
            f"{obj_name} track {track['id']}: appears in frames {frame_ranges_text}",
            f"detected in {len(frames)} frames"
        ]

        if position:
            parts.append(position)

        motion_parts = []
        if movement_x:
            motion_parts.append(movement_x)
        if movement_y:
            motion_parts.append(movement_y)

        if motion_parts:
            parts.append("moves " + " and ".join(motion_parts))
        else:
            parts.append("remains mostly stationary")

        if depth_motion:
            parts.append(depth_motion)

        return "; ".join(parts)

    def _format_frame_ranges(self, frames):
        if not frames:
            return ""

        ranges = []
        start = frames[0]
        end = frames[0]

        for frame in frames[1:]:
            if frame == end + 1:
                end = frame
            else:
                ranges.append((start, end))
                start = frame
                end = frame

        ranges.append((start, end))

        parts = []
        for start, end in ranges:
            if start == end:
                parts.append(str(start))
            else:
                parts.append(f"{start}-{end}")

        return ", ".join(parts)

    def _describe_horizontal_motion(self, dx):
        if dx > self.movement_threshold:
            return "from left to right"
        if dx < -self.movement_threshold:
            return "from right to left"
        return None

    def _describe_vertical_motion(self, dy):
        if dy > self.movement_threshold:
            return "downward"
        if dy < -self.movement_threshold:
            return "upward"
        return None

    def _describe_depth_motion(self, first_area, last_area):
        if first_area <= 0:
            return None

        ratio = (last_area - first_area) / first_area

        if ratio > self.area_change_threshold:
            return "gets larger, likely moving closer"
        if ratio < -self.area_change_threshold:
            return "gets smaller, likely moving farther away"
        return None

    def _describe_position(self, first_bbox, last_bbox):
        first_pos = self._bbox_horizontal_zone(first_bbox)
        last_pos = self._bbox_horizontal_zone(last_bbox)

        if first_pos == last_pos:
            return f"stays mostly in the {first_pos} part of the scene"

        return f"shifts from the {first_pos} part of the scene to the {last_pos} part"

    def _bbox_center_distance(self, bbox1, bbox2):
        x1 = self._bbox_center_x(bbox1)
        y1 = self._bbox_center_y(bbox1)
        x2 = self._bbox_center_x(bbox2)
        y2 = self._bbox_center_y(bbox2)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _bbox_center_x(self, bbox):
        return (bbox["x1"] + bbox["x2"]) / 2.0

    def _bbox_center_y(self, bbox):
        return (bbox["y1"] + bbox["y2"]) / 2.0

    def _bbox_area(self, bbox):
        width = max(0.0, bbox["x2"] - bbox["x1"])
        height = max(0.0, bbox["y2"] - bbox["y1"])
        return width * height

    def _bbox_horizontal_zone(self, bbox):
        center_x = self._bbox_center_x(bbox)

        if center_x < 106:
            return "left"
        if center_x > 213:
            return "right"
        return "center"

    def _bbox_iou(self, bbox1, bbox2):
        x_left = max(bbox1["x1"], bbox2["x1"])
        y_top = max(bbox1["y1"], bbox2["y1"])
        x_right = min(bbox1["x2"], bbox2["x2"])
        y_bottom = min(bbox1["y2"], bbox2["y2"])

        inter_w = max(0.0, x_right - x_left)
        inter_h = max(0.0, y_bottom - y_top)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0

        area1 = self._bbox_area(bbox1)
        area2 = self._bbox_area(bbox2)
        union = area1 + area2 - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union