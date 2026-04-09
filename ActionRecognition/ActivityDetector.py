import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import mediapy as media
import cv2
import pathlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

import tqdm

import tqdm

class ActivityDetector(object):

    URL_TO_LABELS = 'https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
    REQUIRED_WIDTH = 160
    REQUIRED_HEIGHT = 160

    def __init__(self):

        model_id = "a0"
        mode = "stream"
        version = "3"

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth failed: {e}")

        try:

            model_path = f"https://tfhub.dev/tensorflow/movinet/{model_id}/{mode}/kinetics-600/classification/{version}"
            self.model = hub.load(model_path)
        except Exception as e:
            raise Exception(f"Problem while loading MoViNets model {e}")

        try:
            labels_path = tf.keras.utils.get_file(
                fname='labels.txt',
                origin=self.URL_TO_LABELS
            )
            labels_path = pathlib.Path(labels_path)

            lines = labels_path.read_text().splitlines()
            self.KINETICS_600_LABELS = np.array([line.strip() for line in lines])
            self.LABELS = tf.constant([line.strip() for line in lines])
        except Exception as e:
            raise Exception(f"Problem loading MoViNets labels: {e}")

    def get_top_k(self, probs):
        label_map = self.LABELS

        top_prediction = tf.argsort(probs, axis=-1, direction='DESCENDING')[0]
        top_label = tf.gather(label_map, top_prediction, axis=-1).numpy().decode('utf8')

        return top_label

    def detect_activity(self, video_path):
        print("ACTIVITY DETECTION START")
        video = cv2.VideoCapture(video_path)
        num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)


        duration = num_of_frames / fps
        print("Duration: ", duration)
        if duration > 20:
            print("ACTIVITY DETECTION END WITH CODE 2")
            sys.exit(2)
        else:
            frame_count = 0
            start_frame = 0

            video_name_whole = os.path.basename(video_path)
            video_name = Path(video_name_whole).stem

            video_segment = int(fps * 2)
            overlap = int(fps*0.25)

            step = video_segment - overlap

            num_of_segments = 8
            input_segments = []
            frame_info = []
            clip_id = 0

            os.makedirs('Detections', exist_ok=True)
            root = ET.Element('activity_detection')

            frames = []
            while video.isOpened():
                read, frame = video.read()

                if not read:
                    break

                if frame_count % 2 == 0:
                    frame = cv2.resize(frame, (self.REQUIRED_WIDTH, self.REQUIRED_HEIGHT))
                    frame = frame / 255.0
                    frames.append(frame)
                frame_count += 1

                if len(frames) >= video_segment:
                    clip_id += 1

                    if clip_id % 3 != 0:
                        start_frame += step
                        frames = frames[-overlap:]
                        continue
                    normal_frame = np.array(frames, dtype=np.float32)
                    input_segments.append(normal_frame)
                    frame_info.append((start_frame, start_frame + len(frames) - 1))

                    if len(input_segments) >= num_of_segments:
                        self.process_segment(input_segments, frame_info, root)
                        input_segments = []
                        frame_info = []

                    start_frame += (video_segment - overlap)
                    frames = frames[-overlap:]

            if input_segments:
                self.process_segment(input_segments, frame_info, root)

            video.release()
            xml_path = f"Detections/{video_name}_activity.xml"
            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

            print("ACTIVITY DETECTION END WITH CODE 0")
            sys.exit(0)

    def process_segment(self, segments, frame_info, root):

        try:
            input_frame = tf.convert_to_tensor(segments)
            output = self.model.signatures["serving_default"](input_frame)["classifier_head"]

            probabilities = tf.nn.softmax(output, axis=-1)

            for i in range(len(segments)):
                prediction_label = self.get_top_k(probabilities[i:i+1])
                print(f"Detected activity: {prediction_label}")
                start_id, end_id = frame_info[i]
                frame_element = ET.SubElement(root, 'frame')
                frame_start = ET.SubElement(frame_element, 'start_id')
                frame_start.text = str(start_id)
                frame_end = ET.SubElement(frame_element, 'end_id')
                frame_end.text = str(end_id)
                frame_activity = ET.SubElement(frame_element, 'activity')
                frame_activity.text = prediction_label

        except Exception as ex:
            print(f"Error while trying to detect activity: {ex}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ActivityDetector.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    detector = ActivityDetector()
    detector.detect_activity(video_path)
