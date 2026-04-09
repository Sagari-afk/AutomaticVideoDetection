from ultralytics import YOLO
from pathlib import Path
import xml.etree.ElementTree as ET


class ObjectDetector:
    @staticmethod
    def get_object_detections(image_path, treshold=0.2, image_size=1920):
        norm_path = image_path.replace("\\", "/")
        video_p = Path(norm_path)
        stem = video_p.stem

        model = YOLO("yolo26x.pt")
        root = ET.Element("video_object_detection")

        frame_id = 1
        try:
            # TODO: Implement running YOLO on GPU
            results = model.predict(
                source=image_path,
                conf=treshold,
                imgsz=image_size,
                stream=True,
                device=0,
            )
            # print(results[0].show())
            for r in results:
                frame_el = ET.SubElement(root, "frame")
                ET.SubElement(frame_el, "id").text = str(frame_id)

                if r.boxes is not None:
                    for cls, conf in zip(r.boxes.cls, r.boxes.conf):
                        print("Detected:", model.names[int(cls)], "conf:", float(conf))

                if r.boxes is not None and r.boxes.cls is not None:
                    cls_ids = r.boxes.cls.tolist()
                    confs = r.boxes.conf.tolist() if r.boxes.conf is not None else []
                    boxes_xyxy = r.boxes.xyxy.tolist() if r.boxes.xyxy is not None else []

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

                frame_id += 1

        except Exception as e:
            raise Exception(f"YOLO detection failed: {e}")

        main_dir = Path(__file__).parent.absolute()
        xml_out = main_dir / "YOLOv26l" / "runs" / "detect" / f"{stem}_detect.xml"
        xml_out.parent.mkdir(parents=True, exist_ok=True)

        tree = ET.ElementTree(root)
        tree.write(xml_out, encoding="utf-8", xml_declaration=True)
        print("XML output", xml_out)

        return xml_out


if __name__ == "__main__":
    ObjectDetector.get_object_detections(r'C:\Users\Remote_student\Documents\testIMG.png')