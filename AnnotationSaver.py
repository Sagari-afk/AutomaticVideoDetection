import mysql.connector
import io
import config
import xml.etree.ElementTree as ET
import XmlHandler


class AnnotationHandler:

    db = None
    cursor = None

    def add_anotation(self, hours, minutes, seconds, milliseconds, annotation_type, video_name, tag=""):
        self.load_database()
        has_transitions = False
        command = "SELECT id from video_files WHERE path = %s"
        self.cursor.execute(command, (video_name,))
        video_id = self.cursor.fetchone()
        if video_id is None:
            has_transitions = True
            command = "SELECT id from transition_videos WHERE path = %s"
            self.cursor.execute(command, (video_name,))
            video_id = self.cursor.fetchone()
            if video_id is None:
                self.closeDb()
                return

        new_annotation = self.format_time_inc_milis(hours, minutes, seconds, milliseconds) + ";" + annotation_type + ";" + tag + "\n"

        annotation_file_name = video_name + "_annotation.xml"

        command = "SELECT id, path FROM information_files WHERE file_name = %s"
        param = (annotation_file_name,)
        self.cursor.execute(command, param)

        found_id, found_path = self.cursor.fetchone() or (None, None)
        if found_id is None:
            if not has_transitions:
                command = "SELECT id from video_files WHERE path = %s"
            else:
                command = "SELECT id from transition_videos WHERE path = %s"

            self.cursor.execute(command, (video_name,))
            video_id = self.cursor.fetchone()
            if video_id:
                found_video_id = video_id[0]

                root = ET.Element('annotation_data')
                annotation = ET.SubElement(root, 'annotation')
                a_time = ET.SubElement(annotation, 'time')
                a_time.text = self.format_time_inc_milis(hours, minutes, seconds, milliseconds)
                a_type = ET.SubElement(annotation, 'type')
                a_type.text = annotation_type
                a_tag = ET.SubElement(annotation, 'tag')
                a_tag.text = tag
                prettified_xml = XmlHandler.prettify(root)

                with open(annotation_file_name, 'w', encoding='utf-8') as new_file:
                    new_file.write(prettified_xml)

                if not has_transitions:
                    command = "INSERT INTO information_files (path, video_id, file_name) VALUES (%s, %s, %s)"
                else:
                    command = "INSERT INTO information_files (path, trans_video_id, file_name) VALUES (%s, %s, %s)"
                self.cursor.execute(command, (annotation_file_name, found_video_id, annotation_file_name))
                self.db.commit()
            else:
                print("No video id found under used name")
        else:
            tree = ET.parse(found_path)
            root = tree.getroot()
            annotation = ET.SubElement(root, 'annotation')
            a_time = ET.SubElement(annotation, 'time')
            a_time.text = self.format_time_inc_milis(hours, minutes, seconds, milliseconds)
            a_type = ET.SubElement(annotation, 'type')
            a_type.text = annotation_type
            a_tag = ET.SubElement(annotation, 'tag')
            a_tag.text = tag
            prettified_xml = XmlHandler.prettify(root)
            with open(found_path, 'w', encoding='utf-8') as existing_file:
                existing_file.write(prettified_xml)
        self.closeDb()

    def add_annotation_from_existing(self, hours, minutes, seconds, milliseconds, annotation_type, old_video_name,
                                     new_video_name, tag=""):
        self.load_database()
        has_transitions = False
        is_old_transitioned = False
        command = "SELECT id from video_files WHERE path = %s"
        self.cursor.execute(command, (old_video_name,))
        old_video_id = self.cursor.fetchone()
        if old_video_id is None:
            command = "SELECT id from transition_videos WHERE path = %s"
            self.cursor.execute(command, (old_video_name,))
            old_video_id = self.cursor.fetchone()
            is_old_transitioned = True
            if old_video_id is None:
                self.closeDb()
                self.add_anotation(hours, minutes, seconds, milliseconds, annotation_type, new_video_name, tag)
                return
        old_video_id = old_video_id[0]
        new_annotation = self.format_time(hours, minutes, seconds) + ";" + annotation_type + ";" + tag + "\n"
        annotation_file_name = new_video_name + "_annotation.xml"

        if not is_old_transitioned:
            command = "SELECT id, path FROM information_files WHERE video_id = %s"
        else:
            command = "SELECT id, path FROM information_files WHERE trans_video_id = %s"

        param = (old_video_id,)
        self.cursor.execute(command, param)

        found_id, found_path = self.cursor.fetchone() or (None, None)
        if (found_id is None) or (found_path is None):
            self.closeDb()
            self.add_anotation(hours, minutes, seconds, milliseconds, annotation_type, new_video_name, tag)
            return
        else:
            old_path = found_path

            command = "SELECT id from video_files WHERE path = %s"
            self.cursor.execute(command, (new_video_name,))
            new_video_id = self.cursor.fetchone()
            if new_video_id is None:
                has_transitions = True
                command = "SELECT id from transition_videos WHERE path = %s"
                self.cursor.execute(command, (new_video_name,))
                new_video_id = self.cursor.fetchone()
                if new_video_id is None:
                    self.closeDb()
                    return

            command = "SELECT id, path FROM information_files WHERE file_name = %s"
            param = (annotation_file_name,)
            self.cursor.execute(command, param)

            new_found_id, new_found_path = self.cursor.fetchone() or (None, None)
            if new_found_id is None:
                if not has_transitions:
                    command = "SELECT id from video_files WHERE path = %s"
                else:
                    command = "SELECT id from transition_videos WHERE path = %s"

                self.cursor.execute(command, (new_video_name,))
                new_video_id = self.cursor.fetchone()
                if new_video_id:
                    found_video_id = new_video_id[0]

                    tree = ET.parse(old_path)
                    root = tree.getroot()
                    annotation = ET.SubElement(root, 'annotation')
                    a_time = ET.SubElement(annotation, 'time')
                    a_time.text = self.format_time_inc_milis(hours, minutes, seconds, milliseconds)
                    a_type = ET.SubElement(annotation, 'type')
                    a_type.text = annotation_type
                    a_tag = ET.SubElement(annotation, 'tag')
                    a_tag.text = tag
                    prettified_xml = XmlHandler.prettify(root)

                    with open(annotation_file_name, 'w', encoding='utf-8') as new_file:
                        new_file.write(prettified_xml)

                    if not has_transitions:
                        command = "INSERT INTO information_files (path, video_id, file_name) VALUES (%s, %s, %s)"
                    else:
                        command = "INSERT INTO information_files (path, trans_video_id, file_name) VALUES (%s, %s, %s)"

                    self.cursor.execute(command, (annotation_file_name, found_video_id, annotation_file_name))
                    self.db.commit()
            self.closeDb()



    def format_time(self, hours, minutes, seconds):
        formatted_time = str(hours) + ":" + str(minutes) + ":" + str(seconds)
        return formatted_time


    def format_time_inc_milis(self, hours, minutes, seconds, milliseconds):
        formatted_time = str(hours) + ":" + str(minutes) + ":" + str(seconds) + ":" + str(milliseconds)
        return formatted_time


    def load_database(self):
        self.db = mysql.connector.connect(
            host=config.HOST,
            user=config.USER,
            password=config.PASSWORD,
            database=config.DATABASE
        )
        self.cursor = self.db.cursor()

    def closeDb(self):
        self.cursor.close()
        self.db.close()

# handler = AnnotationHandler()
# handler.add_annotation_from_existing(0,0,8, "WIPE_LEFT", "../generatedVideos/2024-05-24_12-22-39.mp4", "../generatedVideos/2024-05-24_13-07-11.mp4","test")
# handler.closeDb()