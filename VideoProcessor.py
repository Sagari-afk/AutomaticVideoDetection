import sys
import xml.etree.ElementTree as ET
import mysql.connector

from ObjectDetector import ObjectDetector
import config
import subprocess
import OperatingSystemCheck
import os
from pathlib import Path
from enum import Enum
import multiprocessing
from lxml import etree
import SentenceMaker as sen
from itertools import chain

class Detection_type(Enum):
    SHOT = "SHOT"
    OBJECT = "OBJECT"
    ACTIVITY = "ACTIVITY"


class VideoProcessor:

    db = None
    cursor = None

    def __init__(self):
        self.load_database()
        self.description_generator = sen.SentenceMaker()
        warm_up = self.description_generator.connect_sentence(["person", "dog", "walking"])
        warm_up_sentence = self.description_generator.translate_sentence(warm_up) #to warm up models
        self.description_generator.translate_sentence(warm_up_sentence)

    def process_video(self, path, saving_path, threshold=0.8, image_size=416):

        video_name = os.path.basename(path)
        name, end = os.path.splitext(video_name)

        if saving_path is None:
            main_dir = Path(__file__).parent.absolute()
            path_to_desc = main_dir / 'Descriptions' / f"{name}_desc_EN.xml"
            save_path = main_dir / 'Runs'
            saving_path = str(save_path)
        else:
            path_to_desc = Path(
                saving_path).resolve() / f"{name}_desc_EN.xml"
        path_to_desc = str(path_to_desc)
        path_to_desc_sk = path_to_desc.replace("_desc_EN.xml", "_desc_SK.xml")
        if self.desc_file_exists(path_to_desc):
            print("Desc file already exists")
            if not self.desc_file_exists(path_to_desc_sk):
                self.save_scene_objects_n_activities_sk(path)
            return
        if not self.desc_file_exists(path_to_desc):
            video_id = self.find_video_id_by_path(path)
            try:
                if not self.video_exists(path):
                    self.add_video_to_database(path)
                    video_id = self.find_video_id_by_path(path)
                shot_detections_path = self.find_detections(path, Detection_type.SHOT.value)
                object_detections_path = self.find_detections(path, Detection_type.OBJECT.value)
                activity_detections_path = self.find_detections(path, Detection_type.ACTIVITY.value)
                if shot_detections_path is None:
                    shot_detections_path = self.get_scenes(path, saving_path)
                    self.save_detections(video_id, shot_detections_path, Detection_type.SHOT.value)

                obj_detect_need = True if object_detections_path is None else False
                act_detect_need = True if activity_detections_path is None else False
                self.close_db()
                object_detections_path, activity_detections_path = (
                    self.process_objects_n_activities(path, threshold, image_size,
                                                      object_detections_path, activity_detections_path))
                self.load_database()
                object_detections_path = str(object_detections_path)
                activity_detections_path = str(activity_detections_path)
                self.activities_detected = os.path.exists(activity_detections_path)
                if obj_detect_need:
                    self.save_detections(video_id, object_detections_path, Detection_type.OBJECT.value)
                if act_detect_need and self.activities_detected:
                    self.save_detections(video_id, activity_detections_path, Detection_type.ACTIVITY.value)
            except Exception as e:
                print("Exception during detections")
                raise Exception(f"Detections failed: {e}")

            scenes = self.load_scenes(shot_detections_path)
            scene_objects = self.load_objects(object_detections_path)
            if self.activities_detected:
                scene_activities = self.load_activities(activity_detections_path)
            else:
                scene_activities = None

            scene_contents = []

            for scene in scenes:
                start_frame = scene[0]
                end_frame = scene[1]
                uniq_objects = set()
                uniq_activities = set()
                unique_desc_data = set()
                sentences_data = {}

                if scene_activities:
                    unique_obj_frames = set(scene_objects) | set(scene_activities)
                else:
                    unique_obj_frames = set(scene_objects)
                sentences = []
                for frame in unique_obj_frames:
                    if start_frame <= frame <= end_frame:
                        objects = tuple(scene_objects.get(frame, []))
                        if self.activities_detected:
                            activity = scene_activities.get(frame, None)
                        else:
                            activity = None

                        uniq_objects.update(objects)

                        if activity and activity not in uniq_activities:
                            uniq_activities.add(activity)

                        combined_contents = (objects, activity)
                        if combined_contents not in unique_desc_data:
                            unique_desc_data.add(combined_contents)
                            sentences_data[frame] = combined_contents
                            # print(f"added UNIQUE COMBINATION in scene {frame} combination {combined_contents}")
                            if combined_contents is not None and len(combined_contents) > 0:
                                if activity is None:
                                    if objects is not None and len(objects) > 0:
                                        generated_sentence = self.description_generator.connect_sentence(objects)
                                    else:
                                        generated_sentence = None
                                else:
                                    combined_contents_clear = list(chain.from_iterable(combined_contents))
                                    generated_sentence = (self.description_generator
                                                          .connect_sentence(combined_contents_clear))
                                if generated_sentence is not None:
                                    sentences.append(generated_sentence)
                if sentences and len(sentences) > 0:
                    generated_desc2 = self.description_generator.connect_desc(sentences)
                    generated_desc3 = self.description_generator.connect_desc(sentences)
                else:
                    generated_desc2 = None
                    generated_desc3 = None
                scene_contents.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "objects": list(uniq_objects),
                    "activities": list(uniq_activities),
                    "description1": " ".join(sentences) if sentences and len(sentences) > 0 else None,
                    "description2": generated_desc2,
                    "description3": generated_desc3
                })

            self.save_scene_objects_n_activities(scene_contents, path_to_desc, video_id)
            if not self.desc_file_exists(path_to_desc_sk):
                self.save_scene_objects_n_activities_sk(path)
            return

    @staticmethod
    def process_objects_n_activities(path, threshold, image_size, object_detections_path, activity_detections_path):
        print("STARTED PROCESSING ACTIVITY AND OBJECT DETECTION")
        processes = []

        result_manager = multiprocessing.Manager()
        result_dict = result_manager.dict()
        if object_detections_path is None:
            process_yolo = multiprocessing.Process(target=VideoProcessor.run_object_detections, args=(path, threshold,
                                                                                            image_size,
                                                                                            result_dict))
            processes.append(process_yolo)

        if activity_detections_path is None:
            check_duration_command = ["ffprobe", "-v", "error",
                                      "-show_entries", "format=duration", "-of",
                                      "default=noprint_wrappers=1:nokey=1", path]
            vid_duration = subprocess.run(check_duration_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True)
            video_duration = float(vid_duration.stdout.strip())
            print(f"video is : {video_duration} seconds long")
            if video_duration <= 20:
                process_activity = multiprocessing.Process(target=VideoProcessor.run_activity_detections,
                                                           args=(path, result_dict))
                processes.append(process_activity)

        for process in processes:
            process.start()
            print("Process started: ", process.name)

        for process in processes:
            process.join()
            print("Process join: ", process.name)

        if object_detections_path is None:
            object_detections_path = result_dict.get("object")

        if activity_detections_path is None:
            activity_detections_path = result_dict.get("activity")

        print("PROCESSING ACTIVITY AND OBJECT DETECTION END")

        return object_detections_path, activity_detections_path

    def desc_file_exists(self, path_to_desc):
        db_command = "SELECT * FROM video_handler.des_files where path = %s "
        self.cursor.execute(db_command, (path_to_desc,))
        result = self.cursor.fetchone()
        if result is None:
            return False
        else:
            return True

    def save_scene_objects_n_activities_sk(self, video_path):
        db_command = (
            "SELECT d.path, v.id FROM video_handler.des_files d join video_handler.video v ON d.video_id = v.id "
            "where v.path = %s AND d.language = 'SK'")
        self.cursor.execute(db_command, (video_path,))
        sk_path = self.cursor.fetchone()
        if sk_path is None:
            db_command = ("SELECT d.path, v.id FROM video_handler.des_files d join video_handler.video v ON d.video_id = v.id "
                          "where v.path = %s AND d.language = 'EN'")
            self.cursor.execute(db_command, (video_path,))
            en_desc_path = self.cursor.fetchone()
            if en_desc_path is not None:
                en_desc_path, video_id = en_desc_path
                saving_path = en_desc_path.replace("EN.xml", "SK.xml")

                tree = etree.iterparse(en_desc_path, events=("end",), tag="scene")
                new_root = ET.Element("descriptions")
                for event, elem in tree:
                    scene_element = ET.SubElement(new_root, "scene")
                    id_element = ET.SubElement(scene_element, "id")
                    id_element.text = elem.find("id").text
                    start_elem = ET.SubElement(scene_element, "startFrame")
                    start_elem.text = elem.find("startFrame").text
                    end_elem = ET.SubElement(scene_element, "endFrame")
                    end_elem.text = elem.find("endFrame").text
                    for obj in elem.findall("object"):
                        obj_element = ET.SubElement(scene_element, "object")
                        obj_element.text = self.description_generator.translate_sentence(obj.text)
                    for desc in elem.findall("description"):
                        desc_element = ET.SubElement(scene_element, "description")
                        desc_element.text = self.description_generator.translate_sentence(desc.text)
                    elem.clear()
                tree = ET.ElementTree(new_root)
                tree.write(saving_path, encoding="utf-8", xml_declaration=True)
                self.save_desc_file(saving_path, video_id, "SK")

    def save_scene_objects_n_activities(self, scene_content, saving_path, video_id):

        root = ET.Element("descriptions")
        scene_id = 1
        for scene in scene_content:
            scene_element = ET.SubElement(root, "scene")
            id_element = ET.SubElement(scene_element, "id")
            id_element.text = str(scene_id)
            start_element = ET.SubElement(scene_element, "startFrame")
            start_element.text = str(scene["start_frame"])
            end_element = ET.SubElement(scene_element, "endFrame")
            end_element.text = str(scene["end_frame"])
            for obj in scene["objects"]:
                object_element = ET.SubElement(scene_element, "object")
                object_element.text = obj
            for act in scene["activities"]:
                object_element = ET.SubElement(scene_element, "object")
                object_element.text = act
            if scene.get("description1"):
                desc1_element = ET.SubElement(scene_element, "description")
                desc1_element.text = scene["description1"]
            if scene.get("description2"):
                desc2_element = ET.SubElement(scene_element, "description")
                desc2_element.text = scene["description2"]
            if scene.get("description3"):
                desc3_element = ET.SubElement(scene_element, "description")
                desc3_element.text = scene["description3"]
            scene_id += 1

        tree = ET.ElementTree(root)
        tree.write(saving_path, encoding="utf-8", xml_declaration=True)
        self.save_desc_file(saving_path, video_id, "EN")

    def save_desc_file(self, saving_path, video_id, language):
        db_command = "INSERT INTO video_handler.des_files (video_id, path, language) VALUES (%s, %s, %s)"
        self.cursor.execute(db_command, (video_id, saving_path, language))
        self.db.commit()

    def load_scenes(self, shot_detections_path):

        scenes = []
        content = etree.iterparse(shot_detections_path, events=("end",), tag="scene")

        for event, elem in content:
            start_frame = int(elem.find("startFrame").text)
            end_frame = int(elem.find("endFrame").text)
            scenes.append([start_frame, end_frame])
            elem.clear()
        return scenes

    def load_objects(self, object_detections_path):

        scene_objects = {}
        content = etree.iterparse(object_detections_path, events=("end",), tag="frame")

        for event, elem in content:
            frame_id = int(elem.find("id").text)
            objects = [obj.text for obj in elem.findall("object")]
            scene_objects[frame_id] = objects
            elem.clear()

        return scene_objects

    def load_activities(self, activity_detections_path):
        scene_activities = {}
        content = etree.iterparse(activity_detections_path, events=("end",), tag="frame")

        for event, elem in content:
            start_id = int(elem.find("start_id").text)
            end_id = int(elem.find("end_id").text)
            activity = elem.find("activity").text
            for frame_id in range(start_id, end_id + 1):
                scene_activities[frame_id] = activity

        return scene_activities

    @staticmethod
    def run_object_detections(path, threshold, image_size, result_dict):
        object_detections_path = ObjectDetector.get_object_detections(path, threshold, image_size)
        result_dict["object"] = object_detections_path

    @staticmethod
    def run_activity_detections( path, result_dict):
        activity_detections_path = VideoProcessor.get_activity_detections(path)
        result_dict["activity"] = activity_detections_path

    def find_detections(self, video_path, detection_type):
        command = ("SELECT path FROM detect_files WHERE detect_type = %s AND video_id in ("
                   " SELECT v.id from video v where v.path = %s)")

        self.cursor.execute(command, (detection_type, video_path,))
        shot_detections = self.cursor.fetchone()
        if shot_detections:
            found_shots = shot_detections[0]
            print(f"Shot detection files of type {detection_type} exist for path {video_path} on path: {found_shots}")
            return found_shots
        else:
            print(f"No shot detections of type {detection_type} found for path: {video_path}")
            return None

    def get_scenes(self, video_path, saving_path):
        shot_detections_path = self.detect_shot_boundary(video_path, saving_path)
        if shot_detections_path is None:
            print("No detections found")
            raise Exception("No detections found")
        return shot_detections_path

    def save_detections(self, video_id, xml_path, detection_type):
        db_command = "INSERT INTO detect_files (video_id, path, detect_type) values (%s, %s, %s)"
        self.cursor.execute(db_command, (video_id, xml_path, detection_type))
        self.db.commit()

    def find_video_id_by_path(self, video_path):
        video_id_command = "SELECT id from video_handler.video WHERE path = %s"
        self.cursor.execute(video_id_command, (video_path,))
        video_id = self.cursor.fetchone()
        if video_id is not None:
            video_id = video_id[0]
        return video_id

    def detect_shot_boundary(self, video_path, saving_path):
        print(__file__)
        norm_path = video_path.replace("\\", "/")
        video_name = os.path.basename(norm_path).replace("\\", "/")
        dir_name = os.path.dirname(norm_path).replace("\\", "/")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        transnet_path = os.path.join(script_dir, "TransNetV2-master")
        found_os = OperatingSystemCheck.operating_system_check()
        detection_complete = False
        if found_os == "Windows":
            transnet_call = ["docker", "run", "--rm", "--gpus", "1",
                             "-v", f"{dir_name}:/tmp", "transnet",
                             "transnetv2_predict",
                             f"/tmp/{video_name}"
                             ]
            try:
                process = subprocess.Popen(transnet_call, cwd=transnet_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                if process.returncode == 0:
                    detection_complete = True
                    print("TransNet detection complete, saving xml...")
                else:
                    print(f"TransNetV2 shot boundary detection failde: {stderr.decode()}")
            except subprocess.CalledProcessError as e:
                print(f"TransNetV2 failed: {e.stderr}")
            if detection_complete:
                str_vid_name = str(video_name)
                ext = str(Path(video_path).suffix)
                detection_path = (dir_name.replace("/", "\\") + "\\"
                                  + str_vid_name.replace(ext, f"{ext}.scenes.txt")) #format transnet output
                root = ET.Element("shot_boundaries")
                with open(detection_path, "r") as txt_file:
                    for line in txt_file:
                        scene = ET.SubElement(root, "scene")
                        start_frame, end_frame = line.split(" ")
                        scene_start = ET.SubElement(scene, "startFrame")
                        scene_start.text = start_frame
                        scene_end = ET.SubElement(scene, "endFrame")
                        scene_end.text = end_frame
                tree = ET.ElementTree(root)
                xml_path = saving_path + "\\" + Path(str_vid_name).stem + ".xml"
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)

                return xml_path
            else:
                print("No detection found, process failed")
                return None

    @staticmethod
    def get_activity_detections(video_path):
        main_dir = Path(__file__).parent.absolute()
        detections_path = os.path.abspath(os.path.join(main_dir, 'ActionRecognition', 'Detections')).replace("\\", "/")

        norm_path = video_path.replace("\\", "/")
        video_name = os.path.basename(norm_path)
        dir_name = os.path.dirname(norm_path)
        dock_path = os.path.abspath(dir_name).replace("\\", "/")
        activity_recognition_call = ["docker", "run", "--rm",
                                     "-v", f"{dock_path}:/TestVideos",
                                     "-v", f"{detections_path}:/ActionRecognition/Detections",
                                     "-w", "/ActionRecognition",
                                     "action-recognition", "python", "ActivityDetector.py",
                                     f"/TestVideos/{video_name}"]
        detection_complete = False
        try:
            process = subprocess.Popen(activity_recognition_call, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, bufsize=1)

            for line in process.stdout:
                print("DOCKER CALL ACTIVITY DT (STD_OUT): ", line.strip())
            for line in process.stderr:
                print("DOCKER CALL ACTIVITY DT (STD_ERR): ", line.strip())

            process.wait()

            if process.returncode == 0:
                detection_complete = True
            else:
                print(f"Activity detection failed")
        except subprocess.CalledProcessError as e:
            print(f"Activity recognition failed: {e.stderr}")

        if detection_complete:
            main_dir = Path(__file__).parent.absolute()
            video_name, suffix = os.path.splitext(video_name)
            activity_output = main_dir / 'ActionRecognition/Detections' / f"{video_name}_activity.xml"
            return activity_output

    def add_video_to_database(self, video_path):
        video_path = Path(video_path)
        video = os.path.basename(video_path)
        video_name = os.path.splitext(video)[0]
        video_path = str(video_path)

        db_command = "INSERT INTO video_handler.video (path, video_name) VALUES (%s,%s)"
        self.cursor.execute(db_command, (video_path, video_name))
        self.db.commit()

    def video_exists(self, video_path):
        db_command = "SELECT * FROM video_handler.video WHERE path = %s"
        self.cursor.execute(db_command, (video_path,))
        result = self.cursor.fetchone()
        if result is None:
            print("No video record found")
            return False
        else:
            print("Video record found")
            return True

    def edit_desc_objects(self, desc_path, scene_id, new_objects):
        objects = new_objects.split("\n#\n")
        tree = etree.parse(desc_path)
        root = tree.getroot()
        scene = root.find(f".//scene[id='{scene_id}']")
        if scene is not None:
            etree.strip_elements(scene,"object", with_tail=False)
            for obj in objects:
                new_obj_elem = etree.SubElement(scene, "object")
                new_obj_elem.text = obj
            tree.write(desc_path, encoding="utf-8", xml_declaration=True)
        else:
            print("Scene with id {} not found".format(scene_id))

    def edit_desc_descriptions(self, desc_path, scene_id, new_descriptions):
        descriptions = new_descriptions.split("\n#\n")
        tree = etree.parse(desc_path)
        root = tree.getroot()
        scene = root.find(f".//scene[id='{scene_id}']")
        if scene is not None:
            etree.strip_elements(scene, "description", with_tail=False)
            for desc in descriptions:
                new_desc_elem = etree.SubElement(scene, "description")
                new_desc_elem.text = desc
            tree.write(desc_path, encoding="utf-8", xml_declaration=True)

    def load_database(self):
        self.db = mysql.connector.connect(
            host=config.HOST,
            user=config.USER,
            password=config.PASSWORD,
            database=config.DATABASE
        )
        self.cursor = self.db.cursor()

    def close_db(self):
        print("----------------CLOSE DB CALLED-----------------")

        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'db') and self.db:
            self.db.close()
        return

    def close_des_models(self):
        if hasattr(self, 'description_generator') and self.description_generator:
            self.description_generator.delete_models()
        return

    def __del__(self):
        self.close_db()
        self.close_des_models()

# when calling separatly, always use this ↓ and don't forget to call close_db at the end
# if __name__ == "__main__":
# if __name__ == "__main__":
#     videoProc = VideoProcessor()
#     # videoProc.add_video_to_database("add_absolute_path.mp4")
#
#     try:
#         videoProc.detect_shot_boundary("add_absolute_path.mp4",
#                                        "add_absolute_path")
#         # videoProc.process_video("add_absolute_path.mp4",None)
#     #                         "add_absolute_path")
#     # videoProc.get_object_detections("add_absolute_path.mp4")
#     # print(torch.cuda.is_available())
#     # print(torch.__version__)
#     # print(torch.version.cuda)
#     # print(torch.backends.cudnn.version())
#     # print(torch.cuda.is_available())
#     # print(torch.cuda.device_count())
#     # print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
#     # videoProc.get_activity_detections("add_absolute_path.mp4")
#         videoProc.description_generator.delete_models()
#     finally:
#         videoProc.close_db()
