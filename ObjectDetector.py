from ultralytics import YOLO
from pathlib import Path
import os
import xml.etree.ElementTree as ET

class ObjectDetector:

    @staticmethod
    def get_object_detections(video_path, tresshold=0.2, image_size=1280):
        norm_path = video_path.replace("\\", "/")
        video_p = Path(norm_path)
        stem = video_p.stem

        model = YOLO("yoloe-26s-seg.pt")
        root = ET.Element("video_object_detection")

        frame_id = 1
        try:
            for r in model.predict(
                source=norm_path,
                conf=tresshold,
                imgsz=image_size,
                device="cpu",
                stream=True,
                verbose=False
            ):
                frame_el = ET.SubElement(root, "frame")
                ET.SubElement(frame_el, "id").text = str(frame_id)

                if r.boxes is not None and r.boxes.cls is not None:
                    cls_ids = r.boxes.cls.tolist()  # list of floats
                    for cid in cls_ids:
                        cid_int = int(cid)
                        name = model.names.get(cid_int, str(cid_int))
                        ET.SubElement(frame_el, "object").text = str(name)

                frame_id += 1

        except Exception as e:
            raise Exception(f"YOLO detection failed: {e}")

        main_dir = Path(__file__).parent.absolute()
        xml_out = main_dir / "YOLOv26s" / "runs" / "detect" / f"{stem}_detect.xml"
        xml_out.parent.mkdir(parents=True, exist_ok=True)

        tree = ET.ElementTree(root)
        tree.write(xml_out, encoding="utf-8", xml_declaration=True)

        return xml_out