import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import xml.etree.ElementTree as ET


class ObjectDetector:
    @staticmethod
    def get_object_detections(
            image_path,
            treshold=0.15,
            image_size=1920,
            scene_ranges=None,
            sample_every_seconds=1.0,
            max_frames_per_scene=30,
            output_dir=None,
    ):
        norm_path = image_path.replace("\\", "/")
        video_p = Path(norm_path)
        stem = video_p.stem

        model = YOLO("yolo26x.pt")
        root = ET.Element("video_object_detection")

        try:
            device = 0 if torch.cuda.is_available() else "cpu"
            metadata = ObjectDetector._get_video_metadata(image_path)
            ObjectDetector._write_video_metadata(root, metadata)

            if scene_ranges:
                frame_ids = ObjectDetector._select_frame_ids(
                    scene_ranges,
                    metadata["total_frames"],
                    metadata["fps"],
                    sample_every_seconds,
                    max_frames_per_scene,
                )
                ObjectDetector._write_sampling_metadata(root, "scene_aware", frame_ids)

                for frame_id, frame in ObjectDetector._iter_selected_frames(image_path, frame_ids):
                    results = model.predict(
                        source=frame,
                        conf=treshold,
                        imgsz=image_size,
                        stream=False,
                        device=device,
                        verbose=False,
                    )
                    result = results[0] if results else None
                    ObjectDetector._write_frame_detections(root, frame_id, result, model)
            else:
                ObjectDetector._write_sampling_metadata(root, "all_frames", None)
                frame_id = 0
                results = model.predict(
                    source=image_path,
                    conf=treshold,
                    imgsz=image_size,
                    stream=True,
                    device=device,
                )
                for result in results:
                    ObjectDetector._write_frame_detections(root, frame_id, result, model)
                    frame_id += 1

        except Exception as e:
            raise Exception(f"YOLO detection failed: {e}")

        if output_dir is None:
            main_dir = Path(__file__).parent.absolute()
            xml_out = main_dir / "YOLOv26l" / "runs" / "detect" / f"{stem}_detect.xml"
        else:
            xml_out = Path(output_dir) / f"{stem}_detect.xml"
        xml_out.parent.mkdir(parents=True, exist_ok=True)

        tree = ET.ElementTree(root)
        tree.write(xml_out, encoding="utf-8", xml_declaration=True)
        print("XML output", xml_out)

        return xml_out

    @staticmethod
    def _get_video_metadata(video_path):
        cap = cv2.VideoCapture(video_path)
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
        }

    @staticmethod
    def _write_video_metadata(root, metadata):
        resolution_el = ET.SubElement(root, "resolution")
        ET.SubElement(resolution_el, "width").text = str(metadata["width"])
        ET.SubElement(resolution_el, "height").text = str(metadata["height"])
        ET.SubElement(resolution_el, "fps").text = f"{metadata['fps']:.3f}"
        ET.SubElement(resolution_el, "total_frames").text = str(metadata["total_frames"])

    @staticmethod
    def _write_sampling_metadata(root, mode, frame_ids):
        sampling_el = ET.SubElement(root, "sampling")
        ET.SubElement(sampling_el, "mode").text = mode
        if frame_ids is not None:
            ET.SubElement(sampling_el, "sampled_frames").text = str(len(frame_ids))

    @staticmethod
    def _select_frame_ids(scene_ranges, total_frames, fps, sample_every_seconds, max_frames_per_scene):
        frame_ids = set()
        step = max(1, int(round((fps or 25.0) * sample_every_seconds)))

        for start_frame, end_frame in scene_ranges:
            start = int(start_frame)
            end = int(end_frame)

            if total_frames > 0:
                start = max(0, min(start, total_frames - 1))
                end = max(0, min(end, total_frames - 1))
            if end < start:
                start, end = end, start

            scene_ids = {start, (start + end) // 2, end}
            scene_ids.update(range(start, end + 1, step))

            if max_frames_per_scene and len(scene_ids) > max_frames_per_scene:
                scene_ids = set(ObjectDetector._spread_frame_ids(start, end, max_frames_per_scene))
                scene_ids.update({start, (start + end) // 2, end})

            frame_ids.update(scene_ids)

        return sorted(frame_ids)

    @staticmethod
    def _spread_frame_ids(start, end, count):
        if count <= 1 or start == end:
            return [start]

        return [
            int(round(start + i * (end - start) / (count - 1)))
            for i in range(count)
        ]

    @staticmethod
    def _iter_selected_frames(video_path, frame_ids):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            for frame_id in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if ok:
                    yield frame_id, frame
                else:
                    print(f"Could not read frame {frame_id}")
        finally:
            cap.release()

    @staticmethod
    def _write_frame_detections(root, frame_id, result, model):
        frame_el = ET.SubElement(root, "frame")
        ET.SubElement(frame_el, "id").text = str(frame_id)

        if result is None or result.boxes is None or result.boxes.cls is None:
            return

        cls_ids = result.boxes.cls.tolist()
        confs = result.boxes.conf.tolist() if result.boxes.conf is not None else []
        boxes_xyxy = result.boxes.xyxy.tolist() if result.boxes.xyxy is not None else []

        for i, cid in enumerate(cls_ids):
            cid_int = int(cid)
            name = model.names.get(cid_int, str(cid_int))
            conf = confs[i] if i < len(confs) else 0.0

            x1, y1, x2, y2 = boxes_xyxy[i] if i < len(boxes_xyxy) else (0, 0, 0, 0)
            width = x2 - x1
            height = y2 - y1

            print(
                f"Frame {frame_id}: {name}, conf={conf:.4f}, "
                f"x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, "
                f"w={width:.1f}, h={height:.1f}"
            )

            obj_el = ET.SubElement(frame_el, "object_detection")
            ET.SubElement(obj_el, "object").text = str(name)
            ET.SubElement(obj_el, "confidence").text = f"{conf:.4f}"

            bbox_el = ET.SubElement(obj_el, "bbox")
            ET.SubElement(bbox_el, "x1").text = f"{x1:.1f}"
            ET.SubElement(bbox_el, "y1").text = f"{y1:.1f}"
            ET.SubElement(bbox_el, "x2").text = f"{x2:.1f}"
            ET.SubElement(bbox_el, "y2").text = f"{y2:.1f}"
            ET.SubElement(bbox_el, "width").text = f"{width:.1f}"
            ET.SubElement(bbox_el, "height").text = f"{height:.1f}"


# if __name__ == "__main__":
#     ObjectDetector.get_object_detections(r'C:\Users\Remote_student\Documents\testIMG.png')
