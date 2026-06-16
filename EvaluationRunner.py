import argparse
import csv
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_VIDEOS_DIR = r"D:\DB_AutomaticVideoDetection\videos"
DEFAULT_MODEL = "qwen2.5:14b-instruct"


class EvaluationRunner:
    def __init__(
            self,
            videos_dir,
            output_dir,
            limit=100,
            model=DEFAULT_MODEL,
            threshold=0.15,
            image_size=1920,
            scene_mode="transnet",
            force_detections=False,
            force_captions=False,
    ):
        self.root_dir = Path(__file__).parent.absolute()
        self.videos_dir = Path(videos_dir)
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.model = model
        self.threshold = threshold
        self.image_size = image_size
        self.scene_mode = scene_mode
        self.force_detections = force_detections
        self.force_captions = force_captions

        self.scene_dir = self.output_dir / "scene_detections"
        self.object_dir = self.output_dir / "object_detections"
        self.caption_dir = self.output_dir / "captions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.object_dir.mkdir(parents=True, exist_ok=True)
        self.caption_dir.mkdir(parents=True, exist_ok=True)
        self.video_processor = _get_video_processor_tools()
        self.description_generator = _get_sentence_maker()
        self.description_generator.ollama_model = self.model

    def run(self):
        videos = sorted(self.videos_dir.glob("*.mp4"))[:self.limit]
        results = []

        for index, video_path in enumerate(videos, start=1):
            print(f"[{index}/{len(videos)}] Processing {video_path.name}")
            try:
                results.extend(self._process_video(video_path))
            except Exception as exc:
                print(f"[ERROR] {video_path.name}: {exc}")
                results.append({
                    "id": f"{video_path.stem}::error",
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "error": str(exc),
                })

            self._write_json(self.output_dir / "results.json", {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "videos_dir": str(self.videos_dir),
                "model": self.model,
                "items": results,
            })

        self._write_initial_ratings(results)
        self._write_review_html()
        self._write_manifest(results)
        print(f"Evaluation run ready: {self.output_dir}")
        print(f"Start review server with: python EvaluationServer.py --run-dir \"{self.output_dir}\"")

    def _process_video(self, video_path):
        video_meta = self._get_video_metadata(video_path)
        scenes = self._get_scenes(video_path, video_meta)
        object_xml = self._get_object_detections(video_path, scenes)
        scene_objects = self._load_objects(object_xml)
        items = []

        for scene_index, (start_frame, end_frame) in enumerate(scenes, start=1):
            scene_id = f"{video_path.stem}::scene_{scene_index:03d}"
            scene_data = self._filter_objects_by_frame_range(scene_objects, start_frame, end_frame)
            captions_path = self.caption_dir / f"{self._safe_name(scene_id)}.json"

            if captions_path.exists() and not self.force_captions:
                captions = self._read_json(captions_path)
            else:
                captions = self._generate_scene_captions(scene_data)
                self._write_json(captions_path, captions)

            display_order = ["summary_text", "objects_only"]
            random.Random(scene_id).shuffle(display_order)

            items.append({
                "id": scene_id,
                "video_name": video_path.name,
                "video_path": str(video_path),
                "scene_index": scene_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": video_meta["fps"],
                "duration_sec": video_meta["duration_sec"],
                "display_order": display_order,
                "summary_text_input": captions["summary_text_input"],
                "objects_only_input": captions["objects_only_input"],
                "descriptions": {
                    "summary_text": captions["summary_text_description"],
                    "objects_only": captions["objects_only_description"],
                },
                "object_counts": captions["object_counts"],
                "sampled_frame_count": len(scene_data),
            })

        return items

    def _get_scenes(self, video_path, video_meta):
        scene_xml = self.scene_dir / f"{video_path.stem}.xml"

        if scene_xml.exists() and self.scene_mode != "single":
            return self.video_processor.load_scenes(str(scene_xml))

        existing = self._find_existing_scene_xml(video_path)
        if existing and self.scene_mode in ("existing", "transnet"):
            scenes = self.video_processor.load_scenes(str(existing))
            self._write_scenes_xml(scene_xml, scenes, video_meta)
            return scenes

        if self.scene_mode == "transnet":
            generated_scene_xml = self.video_processor.detect_shot_boundary(str(video_path), str(self.scene_dir))
            if generated_scene_xml:
                scenes = self.video_processor.load_scenes(str(generated_scene_xml))
            else:
                scenes = [(0, max(0, video_meta["total_frames"] - 1))]
                self._write_scenes_xml(scene_xml, scenes, video_meta)
            return scenes

        scenes = [(0, max(0, video_meta["total_frames"] - 1))]
        self._write_scenes_xml(scene_xml, scenes, video_meta)
        return scenes

    def _get_object_detections(self, video_path, scenes):
        object_xml = self.object_dir / f"{video_path.stem}_detect.xml"
        if object_xml.exists() and not self.force_detections:
            return object_xml

        result_dict = {"object": ""}
        self.video_processor.run_object_detections(
            str(video_path),
            self.threshold,
            self.image_size,
            result_dict,
            scenes=scenes,
            output_dir=self.object_dir,
        )
        return result_dict["object"]

    def _generate_scene_captions(self, scene_objects):
        summary_text_input = self.description_generator.summarizer.summarize_for_llm(scene_objects)
        objects_only_input = self.description_generator.objects_only_summary(scene_objects)

        return {
            "summary_text_input": summary_text_input,
            "objects_only_input": objects_only_input,
            "summary_text_description": self.description_generator.generate_scene_description(scene_objects),
            "objects_only_description": self.description_generator.generate_scene_description_objects_only(scene_objects),
            "object_counts": self._object_counts(scene_objects),
        }

    def _object_counts(self, scene_objects):
        counts = Counter()
        for detections in scene_objects.values():
            for detection in detections:
                counts[detection["object"]] += 1
        return dict(counts)

    def _load_objects(self, object_xml):
        return self.video_processor.load_objects(str(object_xml))

    def _filter_objects_by_frame_range(self, scene_objects, start_frame, end_frame):
        return self.video_processor.filter_objects_by_frame_range(scene_objects, start_frame, end_frame)

    def _find_existing_scene_xml(self, video_path):
        candidates = [
            self.root_dir / "Runs" / f"{video_path.stem}.xml",
            self.root_dir / f"{video_path.stem}.xml",
            video_path.with_suffix(".xml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _write_scenes_xml(self, path, scenes, video_meta):
        root = ET.Element("shot_boundaries")
        ET.SubElement(root, "resolution").text = f"{video_meta['width']}x{video_meta['height']}"
        ET.SubElement(root, "fps").text = f"{video_meta['fps']:.3f}"
        ET.SubElement(root, "frames").text = str(video_meta["total_frames"])
        for start_frame, end_frame in scenes:
            scene_el = ET.SubElement(root, "scene")
            ET.SubElement(scene_el, "startFrame").text = str(start_frame)
            ET.SubElement(scene_el, "endFrame").text = str(end_frame)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _get_video_metadata(video_path):
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": total_frames / fps if fps else 0.0,
        }

    def _write_initial_ratings(self, results):
        ratings_path = self.output_dir / "ratings.json"
        if ratings_path.exists():
            return
        self._write_json(ratings_path, {
            "updated_at": None,
            "ratings": {},
        })

    def _write_manifest(self, results):
        rows = []
        for item in results:
            rows.append({
                "id": item.get("id"),
                "video_name": item.get("video_name"),
                "scene_index": item.get("scene_index"),
                "start_frame": item.get("start_frame"),
                "end_frame": item.get("end_frame"),
                "error": item.get("error"),
            })

        with (self.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "video_name", "scene_index", "start_frame", "end_frame", "error"])
            writer.writeheader()
            writer.writerows(rows)

    def _write_review_html(self):
        html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Description Evaluation</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; color: #222; }
    header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd; padding: 12px 20px; z-index: 2; }
    main { max-width: 1180px; margin: 0 auto; padding: 20px; }
    .scene { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 18px; }
    .meta { color: #555; font-size: 14px; margin-bottom: 10px; }
    video { width: 100%; max-height: 420px; background: #000; border-radius: 6px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa; }
    .desc { min-height: 84px; line-height: 1.45; }
    .hint { color: #666; font-size: 13px; margin: 8px 0 0; }
    label { display: block; margin: 0 0 4px; font-size: 13px; color: #444; }
    select, textarea, input { width: 100%; box-sizing: border-box; }
    textarea { min-height: 72px; }
    .row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .field { display: block; margin: 8px 0; }
    button { padding: 8px 12px; border: 1px solid #bbb; border-radius: 6px; background: #fff; cursor: pointer; }
    button.primary { background: #222; color: #fff; border-color: #222; }
    .status { margin-left: 12px; color: #555; }
    @media (max-width: 800px) { .grid, .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <button class="primary" onclick="saveRatings()">Save ratings</button>
  <button onclick="exportRatings()">Export JSON</button>
  <span id="progress" class="status"></span>
  <span id="saveStatus" class="status"></span>
</header>
<main id="app"></main>
<script>
let results = null;
let ratings = {};
const scoreFields = ["objects", "action", "interaction", "spatial", "fluency", "hallucination", "overall"];
const scoreLabels = {
  objects: "Main objects",
  action: "Action",
  interaction: "Interactions",
  spatial: "Spatial/movement",
  fluency: "Fluency",
  hallucination: "No hallucination",
  overall: "Overall"
};

async function loadData() {
  results = await (await fetch("results.json")).json();
  const ratingResponse = await fetch("ratings.json").catch(() => null);
  if (ratingResponse && ratingResponse.ok) {
    const data = await ratingResponse.json();
    ratings = data.ratings || {};
  }
  render();
}

function modeLabel(mode, orderIndex) {
  return orderIndex === 0 ? "Description A" : "Description B";
}

function renderScoreSelect(itemId, mode, field) {
  const key = `${itemId}.${mode}.${field}`;
  const value = ratings[itemId]?.scores?.[mode]?.[field] || "";
  let html = `<div class="field"><label>${scoreLabels[field] || field}</label><select data-key="${key}" onchange="setRating(this)">`;
  html += `<option value="">-</option>`;
  for (let i = 1; i <= 5; i++) {
    html += `<option value="${i}" ${String(i) === String(value) ? "selected" : ""}>${i}</option>`;
  }
  html += `</select></div>`;
  return html;
}

function render() {
  const app = document.getElementById("app");
  const items = results.items.filter(item => !item.error);
  document.getElementById("progress").textContent = `${Object.keys(ratings).length}/${items.length} scenes rated`;
  app.innerHTML = items.map((item, index) => {
    const startSec = item.fps ? item.start_frame / item.fps : 0;
    const endSec = item.fps ? item.end_frame / item.fps : 0;
    const modes = item.display_order || ["summary_text", "objects_only"];
    const cards = modes.map((mode, modeIndex) => `
      <section class="card">
        <h3>${modeLabel(mode, modeIndex)}</h3>
        <p class="desc">${escapeHtml(item.descriptions[mode] || "")}</p>
        <div class="row">
          ${scoreFields.map(field => renderScoreSelect(item.id, mode, field)).join("")}
        </div>
      </section>
    `).join("");
    const preference = ratings[item.id]?.preference || "";
    const notes = ratings[item.id]?.notes || "";
    return `
      <article class="scene">
        <h2>${index + 1}. ${escapeHtml(item.video_name)} - scene ${item.scene_index}</h2>
        <div class="meta">Frames ${item.start_frame}-${item.end_frame}, approx. ${startSec.toFixed(1)}s-${endSec.toFixed(1)}s, sampled frames: ${item.sampled_frame_count}</div>
        <video id="video_${index}" src="/media/${encodeURIComponent(item.id)}" controls preload="metadata"></video>
        <p><button onclick="playScene('video_${index}', ${startSec}, ${endSec})">Play scene segment</button></p>
        <div class="grid">${cards}</div>
        <p class="hint">Hallucination score: 1 = invents unsupported details, 5 = no unsupported details.</p>
        <div class="field">
          <label>Preference</label>
          <select data-item="${item.id}" onchange="setPreference(this)">
            <option value="">-</option>
            <option value="A" ${preference === "A" ? "selected" : ""}>A better</option>
            <option value="B" ${preference === "B" ? "selected" : ""}>B better</option>
            <option value="tie" ${preference === "tie" ? "selected" : ""}>Tie</option>
            <option value="both_bad" ${preference === "both_bad" ? "selected" : ""}>Both bad</option>
          </select>
        </div>
        <div class="field">
          <label>Notes</label>
          <textarea data-item="${item.id}" oninput="setNotes(this)">${escapeHtml(notes)}</textarea>
        </div>
      </article>
    `;
  }).join("");
}

function ensureItem(itemId) {
  ratings[itemId] = ratings[itemId] || { scores: {}, preference: "", notes: "" };
  return ratings[itemId];
}

function setRating(element) {
  const [itemId, mode, field] = element.dataset.key.split(".");
  const item = ensureItem(itemId);
  item.scores[mode] = item.scores[mode] || {};
  item.scores[mode][field] = element.value;
}

function setPreference(element) {
  ensureItem(element.dataset.item).preference = element.value;
}

function setNotes(element) {
  ensureItem(element.dataset.item).notes = element.value;
}

async function saveRatings() {
  const response = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ updated_at: new Date().toISOString(), ratings })
  });
  document.getElementById("saveStatus").textContent = response.ok ? "Saved" : "Save failed";
  render();
}

function exportRatings() {
  const blob = new Blob([JSON.stringify({ ratings }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ratings_export.json";
  a.click();
  URL.revokeObjectURL(url);
}

function playScene(videoId, startSec, endSec) {
  const video = document.getElementById(videoId);
  video.currentTime = startSec;
  video.play();
  const timer = setInterval(() => {
    if (video.currentTime >= endSec || video.paused) clearInterval(timer);
    if (video.currentTime >= endSec) video.pause();
  }, 250);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[char]));
}

loadData();
</script>
</body>
</html>
"""
        (self.output_dir / "review.html").write_text(html, encoding="utf-8")

    @staticmethod
    def _safe_name(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    @staticmethod
    def _write_json(path, data):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))


def default_output_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("EvaluationRuns") / f"run_{stamp}"


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for evaluation. Install project requirements in the active environment."
        ) from exc


def _get_video_processor_tools():
    try:
        from VideoProcessor import VideoProcessor
        return VideoProcessor.__new__(VideoProcessor)
    except ImportError as exc:
        raise RuntimeError(
            "VideoProcessor dependencies are missing. Install project requirements in the active environment."
        ) from exc


def _get_sentence_maker():
    try:
        from SentenceMaker import SentenceMaker
        return SentenceMaker()
    except ImportError as exc:
        raise RuntimeError(
            "SentenceMaker dependencies are missing. Install project requirements in the active environment."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Run side-by-side video description evaluation.")
    parser.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--scene-mode", choices=["transnet", "existing", "single"], default="transnet")
    parser.add_argument("--force-detections", action="store_true")
    parser.add_argument("--force-captions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    started = time.time()
    runner = EvaluationRunner(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        model=args.model,
        threshold=args.threshold,
        image_size=args.image_size,
        scene_mode=args.scene_mode,
        force_detections=args.force_detections,
        force_captions=args.force_captions,
    )
    runner.run()
    print(f"Finished in {(time.time() - started) / 60.0:.1f} minutes")
