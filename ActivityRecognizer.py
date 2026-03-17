import cv2
from pathlib import Path
import xml.etree.ElementTree as ET

import pathlib

import matplotlib as mpl
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

mpl.rcParams.update({
    'font.size': 10,
})


class ActivityRecognizer:
    KINETICS_600_LABELS = []
    def __init__(self):
        self.model = self.load_model()
        self.labels = self.loadLables()

    def load_model(self):
        print("Loading model movinet starts...")
        id = 'a2'
        mode = 'base'
        version = '3'
        hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'
        model = hub.load(hub_url)
        sig = model.signatures['serving_default']
        print(sig.pretty_printed_signature())
        print("Loading model movinet end...")
        return model

    def read_video_frames(self, video_path, max_frames=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if max_frames is not None and len(frames) >= max_frames:
                break

        cap.release()
        return np.array(frames), fps

    def make_clips(self, frames, clip_len=16, stride=8):
        clips = []
        starts = []

        for start in range(0, len(frames) - clip_len + 1, stride):
            clip = frames[start:start + clip_len]
            clips.append(clip)
            starts.append(start)

        return clips, starts

    def preprocess_clip(self, clip):
        clip = np.array([
            cv2.resize(frame, (224, 224)) for frame in clip
        ], dtype=np.float32) / 255.0
        return clip

    def predict_clip(self, clip):
        print("Activity recognition starts...")
        probs = []
        top_k = self.get_top_k(probs, k=5)
        print(top_k)

        best_label, best_conf = top_k[0]
        print("Activity recognition ends...")
        return best_label, float(best_conf)

    def merge_predictions(self, predictions, fps, stride):
        if not predictions:
            return []

        merged = [predictions[0]]

        for start_f, end_f, label, conf in predictions[1:]:
            last_start, last_end, last_label, last_conf = merged[-1]

            if label == last_label and start_f <= last_end:
                merged[-1] = (
                    last_start,
                    end_f,
                    label,
                    max(last_conf, conf)
                )
            else:
                merged.append((start_f, end_f, label, conf))

        result = []
        for start_f, end_f, label, conf in merged:
            result.append({
                "start_sec": start_f / fps,
                "end_sec": end_f / fps,
                "label": label,
                "confidence": conf
            })

        return result

    def save_to_xml(self, segments, output_path):
        root = ET.Element("activity_detection")

        for seg in segments:
            activity_el = ET.SubElement(root, "activity")
            ET.SubElement(activity_el, "label").text = seg["label"]
            ET.SubElement(activity_el, "start_sec").text = f'{seg["start_sec"]:.2f}'
            ET.SubElement(activity_el, "end_sec").text = f'{seg["end_sec"]:.2f}'
            ET.SubElement(activity_el, "confidence").text = f'{seg["confidence"]:.4f}'

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def process_video(self, video_path):
        frames, fps = self.read_video_frames(video_path)

        if len(frames) == 0:
            raise ValueError("No frames read from video")

        clip_len = 16
        stride = 8
        threshold = 0.3

        clips, starts = self.make_clips(frames, clip_len=clip_len, stride=stride)

        raw_predictions = []
        for clip, start in zip(clips, starts):
            clip_input = self.preprocess_clip(clip)
            label, conf = self.predict_clip(clip_input)

            print(f"Predicted: {label}, conf={conf:.4f}, start_frame={start}")

            if conf >= threshold:
                raw_predictions.append((start, start + clip_len, label, conf))

        segments = self.merge_predictions(raw_predictions, fps=fps, stride=stride)

        main_dir = Path(__file__).parent.absolute()
        detections_dir = main_dir / "ActionRecognition" / "Detections"
        detections_dir.mkdir(parents=True, exist_ok=True)

        video_stem = Path(video_path).stem
        output_path = detections_dir / f"{video_stem}_activity.xml"

        self.save_to_xml(segments, output_path)
        return output_path

    def loadLables(self):
        labels_path = tf.keras.utils.get_file(
            fname='labels.txt',
            origin='https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
        )
        labels_path = pathlib.Path(labels_path)

        lines = labels_path.read_text().splitlines()
        KINETICS_600_LABELS = np.array([line.strip() for line in lines])
        return KINETICS_600_LABELS

    # Get top_k labels and probabilities
    def get_top_k(self, probs, k=5, label_map=None):

        if label_map is None:
            label_map = self.KINETICS_600_LABELS

        probs = tf.convert_to_tensor(probs)

        if len(probs.shape) == 2 and probs.shape[0] == 1:
            probs = probs[0]

        num_classes = probs.shape[-1]
        k = min(k, num_classes)

        # индексы top-k
        top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]

        top_labels = tf.gather(label_map, top_predictions)

        top_labels = [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in top_labels.numpy()
        ]

        # вероятности top-k
        top_probs = tf.gather(probs, top_predictions).numpy()

        print("Top predictions:")
        print(top_labels)
        print("Top probabilities:")
        print(top_probs)

        return list(zip(top_labels, top_probs))