import math
from collections import Counter


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
        self.frame_width = None
        self.frame_height = None
        self.grid_columns = ("left", "center", "right")
        self.grid_rows = ("top", "middle", "bottom")
        self.depth_layers = ("foreground", "midground", "background")

    def summarize_scene_objects(self, scene_objects):
        self._update_frame_size(scene_objects)
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

        self._update_frame_size(scene_objects)
        filtered = self._filter_low_confidence(scene_objects)
        tracks = self._build_tracks(filtered)
        tracks = self._remove_short_tracks(tracks)
        tracks = self._merge_fragmented_tracks(tracks)
        interactions = self._detect_interactions(tracks)

        return {
            "tracks": tracks,
            "interactions": interactions,
        }

    def summarize_for_llm(self, scene_objects):
        self._update_frame_size(scene_objects)
        filtered = self._filter_low_confidence(scene_objects)
        tracks = self._build_tracks(filtered)
        tracks = self._remove_short_tracks(tracks)
        tracks = self._merge_fragmented_tracks(tracks)
        interactions = self._detect_interactions(tracks)
        total_frames = self._total_frame_count(scene_objects)

        lines = ["Video analysis summary:"]

        lines.append("")
        lines.append("Main objects:")
        if tracks:
            lines.append(f"- {self._stable_object_count_summary(tracks)}")
            for track in self._representative_tracks(tracks, limit=3):
                obj_name = self._readable_object_name(track["object"])
                visibility = self._visibility_summary(track, total_frames)
                lines.append(f"- {self._sentence_start_object(obj_name)} {visibility}.")
        else:
            lines.append("- No stable objects were detected.")

        lines.append("")
        lines.append("Spatial layout:")
        if tracks:
            for track in self._representative_tracks(tracks, limit=4):
                obj_name = self._readable_object_name(track["object"])
                lines.append(f"- {self._sentence_start_object(obj_name)} {self._simplified_position_summary(track)}.")
        else:
            lines.append("- Spatial layout is unclear because there are no reliable object tracks.")

        lines.append("")
        lines.append("Movement and changes:")
        if tracks:
            for track in self._representative_tracks(tracks, limit=4):
                obj_name = self._readable_object_name(track["object"])
                lines.append(f"- {self._movement_change_summary(track, obj_name)}")
        else:
            lines.append("- No reliable movement can be summarized.")

        lines.append("")
        lines.append("Possible interactions:")
        selected_interactions = self._select_llm_interactions(interactions)
        if selected_interactions:
            for interaction in selected_interactions:
                lines.append(f"- {self._interaction_semantic_hint(interaction)}")
        else:
            lines.append("- No reliable interactions were detected.")

        lines.append("")
        lines.append("Notes:")
        lines.append("- Short or low-confidence detections are ignored.")
        lines.append("- Spatial regions are estimated from object bounding boxes over a 3x3x3 frame grid, including approximate depth.")

        return "\n".join(lines)

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

    def _update_frame_size(self, scene_objects):
        width = None
        height = None
        max_x = 0.0
        max_y = 0.0

        for detections in scene_objects.values():
            for det in detections:
                width = width or det.get("frame_width")
                height = height or det.get("frame_height")

                bbox = det.get("bbox")
                if not bbox:
                    continue

                max_x = max(max_x, bbox.get("x2", 0.0))
                max_y = max(max_y, bbox.get("y2", 0.0))

        self.frame_width = float(width) if width else max_x or None
        self.frame_height = float(height) if height else max_y or None

    def _total_frame_count(self, scene_objects):
        if not scene_objects:
            return 0

        frame_ids = sorted(scene_objects.keys())
        return max(1, frame_ids[-1] - frame_ids[0] + 1)

    def _representative_tracks(self, tracks, limit=4):
        return sorted(
            tracks,
            key=lambda track: (len(track["frames"]), self._bbox_area(track["detections"][-1]["bbox"])),
            reverse=True,
        )[:limit]

    def _stable_object_count_summary(self, tracks):
        counts = Counter(track["object"] for track in tracks)
        parts = []

        for obj_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            readable = self._readable_object_name(obj_name, count)
            parts.append(f"{count} {readable}")

        return f"Detected stable objects: {self._join_words(parts)}."

    def _readable_object_name(self, class_name, count=1):
        name = str(class_name).replace("_", " ").strip().lower()
        singular_specials = {
            "person": "person",
        }
        plural_specials = {
            "person": "people",
            "bicycle": "bicycles",
            "car": "cars",
            "dog": "dogs",
        }

        if count == 1:
            return singular_specials.get(name, name)

        if name in plural_specials:
            return plural_specials[name]
        if name.endswith(("s", "x", "z", "ch", "sh")):
            return f"{name}es"
        if len(name) > 1 and name.endswith("y") and name[-2] not in "aeiou":
            return f"{name[:-1]}ies"
        return f"{name}s"

    def _sentence_start_object(self, obj_name):
        article = "an" if obj_name[:1].lower() in "aeiou" else "a"
        return f"{article.capitalize()} {obj_name}"

    def _visibility_summary(self, track, total_frames):
        if total_frames <= 0 or not track["frames"]:
            return "is visible briefly"

        frames = sorted(track["frames"])
        detection_count = len(frames)
        first_frame = frames[0]
        last_frame = frames[-1]
        span = max(1, last_frame - first_frame + 1)
        coverage = detection_count / total_frames
        span_ratio = span / total_frames
        density = detection_count / span

        if coverage >= 0.6 or (span_ratio >= 0.8 and density >= 0.45):
            return "is visible throughout most of the video"
        if detection_count <= max(2, total_frames * 0.08) or coverage < 0.08:
            return "is visible briefly"
        if span_ratio >= 0.35 and density < 0.5:
            return "appears intermittently"

        midpoint = (first_frame + last_frame) / 2.0
        relative_midpoint = midpoint / max(1, total_frames)

        if relative_midpoint < 1.0 / 3.0:
            return "is visible in the first part of the video"
        if relative_midpoint > 2.0 / 3.0:
            return "is visible in the last part of the video"
        return "is visible in the middle part of the video"

    def _simplified_position_summary(self, track):
        first_region = self._simplified_bbox_region(track["detections"][0]["bbox"])
        last_region = self._simplified_bbox_region(track["detections"][-1]["bbox"])

        if first_region == last_region:
            if first_region == "large area of the frame":
                return "covers a large area of the frame"
            return f"stays mostly {first_region}"

        return f"moves from the {first_region} to the {last_region}"

    def _simplified_bbox_region(self, bbox):
        layer = self._bbox_depth_layer(bbox)
        layer_phrase = self._depth_layer_phrase(layer)
        rows, columns = self._bbox_grid_parts(bbox)
        cell_count = len(rows) * len(columns)

        if cell_count >= 6:
            return f"{layer_phrase} across a large area of the frame"

        if "left" in columns and "right" not in columns:
            horizontal = "on the left side of the frame"
        elif "right" in columns and "left" not in columns:
            horizontal = "on the right side of the frame"
        else:
            horizontal = "near the center of the frame"

        if len(rows) == 1 and rows[0] != "middle":
            vertical = "upper" if rows[0] == "top" else "lower"
            return f"{layer_phrase} in the {vertical} area {horizontal}"

        return f"{layer_phrase} {horizontal}"

    def _movement_change_summary(self, track, obj_name):
        detections = track["detections"]
        first_bbox = detections[0]["bbox"]
        last_bbox = detections[-1]["bbox"]

        dx = self._bbox_center_x(last_bbox) - self._bbox_center_x(first_bbox)
        dy = self._bbox_center_y(last_bbox) - self._bbox_center_y(first_bbox)
        first_area = self._bbox_area(first_bbox)
        last_area = self._bbox_area(last_bbox)

        screen_movement = self._screen_movement_summary(dx, dy)
        depth_change = self._depth_and_scale_change_summary(first_bbox, last_bbox, first_area, last_area)
        subject = self._sentence_start_object(obj_name)

        if depth_change and screen_movement == "stays mostly stable on screen":
            return f"{subject} {screen_movement}, but {depth_change}."
        if depth_change:
            return f"{subject} {screen_movement} and {depth_change}."
        return f"{subject} {screen_movement}."

    def _screen_movement_summary(self, dx, dy):
        movement_parts = []

        if dx > self.movement_threshold:
            movement_parts.append("left to right")
        elif dx < -self.movement_threshold:
            movement_parts.append("right to left")

        if dy > self.movement_threshold:
            movement_parts.append("downward")
        elif dy < -self.movement_threshold:
            movement_parts.append("upward")

        if not movement_parts:
            return "stays mostly stable on screen"

        return f"moves {self._join_words(movement_parts)}"

    def _scale_change_summary(self, first_area, last_area):
        if first_area <= 0:
            return None, None

        ratio = (last_area - first_area) / first_area

        if ratio > self.area_change_threshold:
            return "gets larger", "moving closer"
        if ratio < -self.area_change_threshold:
            return "gets smaller", "moving farther away"
        return None, None

    def _depth_and_scale_change_summary(self, first_bbox, last_bbox, first_area=None, last_area=None):
        if first_area is None:
            first_area = self._bbox_area(first_bbox)
        if last_area is None:
            last_area = self._bbox_area(last_bbox)

        first_layer = self._bbox_depth_layer(first_bbox)
        last_layer = self._bbox_depth_layer(last_bbox)
        scale_change, scale_interpretation = self._scale_change_summary(first_area, last_area)

        if first_layer == last_layer:
            if not scale_change:
                return None
            return f"{scale_change} while staying mostly in the {first_layer}, suggesting it may be {scale_interpretation}"

        first_index = self._depth_layer_index(first_layer)
        last_index = self._depth_layer_index(last_layer)
        layer_interpretation = "moving closer" if last_index < first_index else "moving farther away"
        layer_change = f"shifts from the {first_layer} toward the {last_layer}"

        if scale_change:
            if scale_interpretation == layer_interpretation:
                return f"{layer_change} and {scale_change}, suggesting it may be {layer_interpretation}"
            return (
                f"{layer_change} while it {scale_change}; "
                f"the depth and scale cues are mixed, so its distance change is uncertain"
            )

        return f"{layer_change}, suggesting it may be {layer_interpretation}"

    def _select_llm_interactions(self, interactions, limit=4):
        high = [item for item in interactions if item.get("confidence") == "high"]
        medium = [item for item in interactions if item.get("confidence") == "medium"]
        low = [item for item in interactions if item.get("confidence") == "low"]

        selected = list(high)
        if len(selected) < 2:
            selected.extend(medium[:2 - len(selected)])
        if not selected:
            selected.extend(low[:1])

        return selected[:limit]

    def _interaction_semantic_hint(self, interaction):
        obj1 = self._readable_object_name(interaction["track_1_object"])
        obj2 = self._readable_object_name(interaction["track_2_object"])
        pair = {interaction["track_1_object"], interaction["track_2_object"]}
        overlap_ratio = interaction["iou_hits"] / max(1, interaction["shared_frame_count"])
        relation = self._stable_relative_position_sentence(interaction, obj1, obj2)
        depth_relation = self._interaction_depth_sentence(interaction, obj1, obj2)

        if pair == {"person", "bicycle"}:
            if overlap_ratio >= 0.35 or self._relative_position_share(interaction, "overlapping with") >= 0.35:
                base = (
                    "A person and a bicycle overlap for much of the video, "
                    "which may indicate someone riding or standing very close to a bicycle."
                )
            else:
                base = "A person stays close to a bicycle."
        elif interaction["track_1_object"] == "person" and interaction["track_2_object"] == "person":
            base = "Two people appear close to each other."
        else:
            base = (
                f"{self._sentence_start_object(obj1)} appears close to or overlapping with "
                f"{self._object_with_article(obj2)}."
            )

        details = [detail for detail in (relation, depth_relation) if detail]
        if details:
            return f"{base} {' '.join(details)}"
        return base

    def _stable_relative_position_sentence(self, interaction, obj1, obj2):
        relative_position = interaction.get("relative_position")
        ratio = interaction.get("relative_position_ratio", 0.0)

        if self._relative_position_share(interaction, "overlapping with") >= 0.35:
            return "The objects frequently overlap."
        if not relative_position or ratio < 0.6 or relative_position == "near":
            return None
        if relative_position == "overlapping with":
            return "The objects frequently overlap."

        return f"The {obj1} is usually {relative_position} the {obj2}."

    def _relative_position_share(self, interaction, relation):
        counts = interaction.get("relative_position_counts") or {}
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        return counts.get(relation, 0) / total

    def _interaction_depth_sentence(self, interaction, obj1, obj2):
        layer_1 = interaction.get("track_1_depth_layer")
        layer_2 = interaction.get("track_2_depth_layer")
        same_depth_ratio = interaction.get("same_depth_ratio", 0.0)

        if not layer_1 or not layer_2:
            return None

        if layer_1 == layer_2 and same_depth_ratio >= 0.6:
            return f"Both objects are mostly in the {layer_1}."

        if same_depth_ratio < 0.4:
            return f"The {obj1} is mostly in the {layer_1}, while the {obj2} is mostly in the {layer_2}."

        return None

    def _object_with_article(self, obj_name):
        article = "an" if obj_name[:1].lower() in "aeiou" else "a"
        return f"{article} {obj_name}"

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
        min_track_length = self._effective_min_track_length(tracks)
        return [t for t in tracks if len(t["frames"]) >= min_track_length]

    def _effective_min_track_length(self, tracks):
        if not tracks:
            return self.min_track_length

        sampled_frames = {
            frame_id
            for track in tracks
            for frame_id in track["frames"]
        }

        if not sampled_frames:
            return self.min_track_length

        # Scene-aware object detection samples only a subset of frames, so the
        # reliability threshold must scale with the number of analyzed frames.
        adaptive_length = max(1, math.ceil(len(sampled_frames) * 0.35))
        return min(self.min_track_length, adaptive_length)

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
                    "relative_position": stats["relative_position"],
                    "relative_position_ratio": stats["relative_position_ratio"],
                    "relative_position_counts": stats["relative_position_counts"],
                    "track_1_depth_layer": stats["track_1_depth_layer"],
                    "track_2_depth_layer": stats["track_2_depth_layer"],
                    "same_depth_ratio": stats["same_depth_ratio"],
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
        relative_positions = []
        depth_pairs = []

        for frame_id in shared_frames:
            det1 = t1["frame_to_detection"][frame_id]
            det2 = t2["frame_to_detection"][frame_id]

            bbox1 = det1["bbox"]
            bbox2 = det2["bbox"]

            iou = self._bbox_iou(bbox1, bbox2)
            dist = self._bbox_center_distance(bbox1, bbox2)

            iou_values.append(iou)
            center_distances.append(dist)
            relative_positions.append(self._bbox_relative_position(bbox1, bbox2))
            depth_pairs.append((self._bbox_depth_layer(bbox1), self._bbox_depth_layer(bbox2)))

            if iou >= self.interaction_iou_threshold:
                iou_hits += 1
            if dist <= self.interaction_center_threshold:
                close_hits += 1

        if not center_distances:
            return None

        same_direction_ratio = self._pair_same_direction_ratio(t1, t2)
        relative_counter = Counter(relative_positions)
        relative_position, relative_hits = relative_counter.most_common(1)[0]
        depth_1_counter = Counter(pair[0] for pair in depth_pairs)
        depth_2_counter = Counter(pair[1] for pair in depth_pairs)
        same_depth_hits = sum(1 for pair in depth_pairs if pair[0] == pair[1])

        return {
            "iou_hits": iou_hits,
            "close_hits": close_hits,
            "avg_iou": sum(iou_values) / len(iou_values),
            "avg_center_distance": sum(center_distances) / len(center_distances),
            "same_direction_ratio": same_direction_ratio,
            "relative_position": relative_position,
            "relative_position_ratio": relative_hits / len(relative_positions),
            "relative_position_counts": dict(relative_counter),
            "track_1_depth_layer": depth_1_counter.most_common(1)[0][0],
            "track_2_depth_layer": depth_2_counter.most_common(1)[0][0],
            "same_depth_ratio": same_depth_hits / len(depth_pairs),
        }

    def _infer_interaction_type(self, t1, t2, stats, shared_frames):
        shared_count = len(shared_frames)
        close_ratio = stats["close_hits"] / max(1, shared_count)
        iou_ratio = stats["iou_hits"] / max(1, shared_count)
        same_direction = stats["same_direction_ratio"]

        if (
                shared_count >= self.strong_interaction_min_shared_frames
                and close_ratio >= self.strong_interaction_min_close_ratio
                and same_direction >= self.strong_interaction_min_same_direction
                and (iou_ratio >= self.strong_interaction_min_iou_ratio or stats[
            "avg_center_distance"] <= self.interaction_center_threshold * 0.7)
        ):
            return "strong joint interaction"

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
            f"{obj1} is usually {interaction['relative_position']} {obj2}; "
            f"depth: {obj1} mostly {interaction.get('track_1_depth_layer', 'unknown')}, "
            f"{obj2} mostly {interaction.get('track_2_depth_layer', 'unknown')}; "
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
        depth_motion = self._depth_and_scale_change_summary(first_bbox, last_bbox, first_area, last_area)
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
            parts.append("stays mostly stable on screen")

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
        first_pos = self._bbox_grid_position_text(first_bbox)
        last_pos = self._bbox_grid_position_text(last_bbox)

        if first_pos == last_pos:
            return f"stays mostly {self._position_phrase(first_pos)}"

        return (
            f"shifts from {self._position_phrase_short(first_pos)} "
            f"to {self._position_phrase_short(last_pos)}"
        )

    def _position_phrase_short(self, position):
        if "whole frame" in position:
            return "the whole frame"
        if "," in position or " and " in position:
            return f"the {position} parts of the scene"
        return f"the {position} part of the scene"

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

    def _bbox_grid_position_text(self, bbox):
        return self._format_grid_cells(self._bbox_grid_cells(bbox))

    def _bbox_grid_cells(self, bbox):
        rows, columns = self._bbox_grid_parts(bbox)
        layer = self._bbox_depth_layer(bbox)

        cells = []
        for row in self.grid_rows:
            if row not in rows:
                continue
            for column in self.grid_columns:
                if column not in columns:
                    continue
                cells.append(self._grid_cell_name(row, column, layer))
        return cells

    def _bbox_grid_parts(self, bbox):
        frame_width = self._frame_width_for_bbox(bbox)
        frame_height = self._frame_height_for_bbox(bbox)
        columns = self._axis_grid_zones(
            bbox["x1"],
            bbox["x2"],
            frame_width,
            self.grid_columns,
        )
        rows = self._axis_grid_zones(
            bbox["y1"],
            bbox["y2"],
            frame_height,
            self.grid_rows,
        )

        return rows, columns

    def _bbox_horizontal_zone(self, bbox):
        center_x = self._bbox_center_x(bbox)
        frame_width = self._frame_width_for_bbox(bbox)

        return self._third_zone(center_x, frame_width, "left", "center", "right")

    def _bbox_vertical_zone(self, bbox):
        center_y = self._bbox_center_y(bbox)
        frame_height = self._frame_height_for_bbox(bbox)

        return self._third_zone(center_y, frame_height, "top", "middle", "bottom")

    def _axis_grid_zones(self, start, end, size, labels):
        size = max(size, 1.0)
        start = max(0.0, min(start, size))
        end = max(0.0, min(end, size))

        if end < start:
            start, end = end, start

        if start == end:
            return [self._third_zone(start, size, labels[0], labels[1], labels[2])]

        boundaries = (0.0, size / 3.0, size * 2.0 / 3.0, size)
        zones = []

        for i, label in enumerate(labels):
            zone_start = boundaries[i]
            zone_end = boundaries[i + 1]

            if max(start, zone_start) < min(end, zone_end):
                zones.append(label)

        return zones or [self._third_zone((start + end) / 2.0, size, labels[0], labels[1], labels[2])]

    def _format_grid_cells(self, cells):
        if len(cells) == 1:
            return cells[0]

        if len(cells) == 9:
            return f"{self._cell_depth_layer(cells[0])} whole frame"

        return self._join_words(cells)

    def _cell_depth_layer(self, cell):
        return cell.split(" ", 1)[0]

    def _position_phrase(self, position):
        if "whole frame" in position:
            return f"across the {position}"
        if "," in position or " and " in position:
            return f"in the {position} parts of the scene"
        return f"in the {position} part of the scene"

    def _grid_cell_name(self, row, column, layer):
        cell = "center" if row == "middle" and column == "center" else f"{row}-{column}"
        return f"{layer} {cell}"

    def _bbox_depth_layer(self, bbox):
        frame_width = self._frame_width_for_bbox(bbox)
        frame_height = self._frame_height_for_bbox(bbox)
        frame_area = max(frame_width * frame_height, 1.0)
        area_ratio = self._bbox_area(bbox) / frame_area
        height_ratio = max(0.0, bbox["y2"] - bbox["y1"]) / max(frame_height, 1.0)
        center_y_ratio = self._bbox_center_y(bbox) / max(frame_height, 1.0)

        if area_ratio > 0.18 or height_ratio > 0.55 or (center_y_ratio > 0.70 and area_ratio > 0.05):
            return "foreground"
        if area_ratio < 0.03 or height_ratio < 0.18:
            return "background"
        return "midground"

    def _depth_layer_phrase(self, layer):
        if layer == "foreground":
            return "in the foreground"
        if layer == "background":
            return "in the background"
        return "in the midground"

    def _depth_layer_index(self, layer):
        order = {
            "foreground": 0,
            "midground": 1,
            "background": 2,
        }
        return order.get(layer, 1)

    def _bbox_relative_position(self, bbox1, bbox2):
        if self._bbox_iou(bbox1, bbox2) >= self.interaction_iou_threshold:
            return "overlapping with"

        dx = self._bbox_center_x(bbox1) - self._bbox_center_x(bbox2)
        dy = self._bbox_center_y(bbox1) - self._bbox_center_y(bbox2)
        deadzone_x = self._relative_position_deadzone(self._frame_width_for_bbox(bbox1))
        deadzone_y = self._relative_position_deadzone(self._frame_height_for_bbox(bbox1))

        horizontal = None
        vertical = None

        if dx < -deadzone_x:
            horizontal = "left of"
        elif dx > deadzone_x:
            horizontal = "right of"

        if dy < -deadzone_y:
            vertical = "above"
        elif dy > deadzone_y:
            vertical = "below"

        if vertical and horizontal:
            return f"{vertical}-{horizontal}"
        if vertical:
            return vertical
        if horizontal:
            return horizontal
        return "near"

    def _relative_position_deadzone(self, size):
        return max(20.0, size * 0.04)

    def _frame_width_for_bbox(self, bbox):
        return self.frame_width or max(bbox["x2"], 1.0)

    def _frame_height_for_bbox(self, bbox):
        return self.frame_height or max(bbox["y2"], 1.0)

    def _join_words(self, words):
        if len(words) == 1:
            return words[0]
        if len(words) == 2:
            return f"{words[0]} and {words[1]}"
        return f"{', '.join(words[:-1])}, and {words[-1]}"

    def _third_zone(self, value, size, first_label, middle_label, last_label):
        first_boundary = size / 3.0
        second_boundary = size * 2.0 / 3.0

        if value < first_boundary:
            return first_label
        if value > second_boundary:
            return last_label
        return middle_label

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
