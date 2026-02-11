import sys
from moviepy.editor import VideoClip
from lxml import etree
from moviepy.video.io.VideoFileClip import VideoFileClip

from PySide6.QtWidgets import QApplication, QMainWindow, QColorDialog, QFileDialog, QProgressDialog, QMessageBox, \
    QDialog, QWidget, QInputDialog, QDialogButtonBox, QPushButton
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QUrl, QThread, QObject, Signal, Qt, Slot, QTime
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtGui import QImage, QPixmap, QAction, QIcon, QIntValidator

import VideoProcessor as VP
import VideoSearcher as VS
import DockerStarter as DS
import VideoGenerator
from VideoGenerator import Transition
from GUIFiles import GuiDatabaseHandler
import tempfile
import shutil
import os
import math

class DockerStarterWorker(QObject):
    finished = Signal(bool)

    def __init__(self):
        super().__init__()

    @Slot()
    def run(self):
        try:
            DS.start_docker()
            self.finished.emit(True)
        except Exception as e:
            print(f"Exception during building docker images: {e}")
            self.finished.emit(False)
class GUIThreadWorker(QObject):
    finished = Signal(bool)
    # progress = Signal(int)
    db_closed = Signal(bool)

    def __init__(self, video_path, saving_path, threshold=0.8, image_size=416):
        super().__init__()
        self.video_path = video_path
        self.saving_path = saving_path
        self.threshold = threshold
        self.image_size = int(image_size)
        self.video_processor = VP.VideoProcessor()

    @Slot()
    def run_video_proc(self):
        print("Found path: " + self.video_path)
        video_path = self.video_path.replace("/", "\\")
        if self.saving_path is not None:
            save_path = self.saving_path.replace("/", "\\")
        else:
            save_path = None
        self.video_processor.process_video(video_path, save_path, self.threshold, self.image_size)
        self.finished.emit(True)

    @Slot()
    def close_database(self):
        print("Closing database for video processor")
        self.video_processor.close_db()
        self.video_processor.close_des_models()
        self.db_closed.emit(True)



class VideoGeneratorWorker(QObject):
    finished_task_1 = Signal(bool)
    progress_task_1 = Signal(int)

    finished_task_2 = Signal(bool)
    progress_task_2 = Signal(int)

    finished_task_3 = Signal(bool)
    progress_task_3 = Signal(int)

    def __init__(self):
        super().__init__()
        self.video_generator = VideoGenerator.VideoGenerator()

    @Slot(str, object)
    def add_transitions_to_one(self, video_path, part_size=5):
        if video_path is not None:
            video_path = video_path.replace("/", "\\")
            if part_size is None:
                part_size = 5
            else:
                part_size = int(part_size)
            print(f"Adding transitions to {video_path} with part size {part_size}")
            self.video_generator.add_transitions_to_one(video_path, part_size)
            self.finished_task_1.emit(True)
        else:
            print(f"No video path. Ending add transitions to one")
            self.finished_task_1.emit(True)

    @Slot(str, object, int, int, float, int, str, object, object)
    def add_transition_by_choice(self, video_path, transition, hour, minute, second, millisecond, second_half_tag="",
                                 transition_duration=1, fade_color=None):
        formatted_time = f"{hour}:{minute}:{second}.{millisecond}"
        if transition_duration is None:
            transition_duration = 1
        if video_path is not None:
            video_path = video_path.replace("/", "\\")
            print(f"Adding transition by choice to {video_path}. Transition {transition} at time {formatted_time}"
                  f" with duration {transition_duration}."
                  f"\nfade_color: {fade_color}")
            if transition_duration > 0:
                self.video_generator.add_transition_by_choice(video_path, transition, hour, minute, second, millisecond,
                                                          second_half_tag, transition_duration, fade_color)
            else:
                print("Transition duration must be at least 1 second")
            self.finished_task_2.emit(True)
        else:
            print(f"Video path missing. Ending add transition by choice")
            self.finished_task_2.emit(True)

    @Slot(object)
    def add_transitions_random(self, number_of_videos=2):
        print(f"Adding random transitions to {number_of_videos} videos")
        self.video_generator.add_transitions(number_of_videos)
        self.finished_task_3.emit(True)



class ControlWindow(QMainWindow):

    trigger_add_transitions_to_one = Signal(str, object)
    trigger_add_transitions_rand = Signal(int)
    trigger_add_transitions_by_choice = Signal(str, object, int, int, float, int, str, object, object)

    def __init__(self):
        super().__init__()

        self.language = "EN"
        self.threshold = 0.8
        self.image_size = 416

        self.video = None
        self.video_path = None
        self.fps = 0
        self.scene1_id = 1
        self.scene2_id = 2
        self.scene3_id = 3

        self.scene1_objects = None
        self.scene1_descriptions = None
        self.scene2_objects = None
        self.scene2_descriptions = None
        self.scene3_objects = None
        self.scene3_descriptions = None

        self.desc_path = None
        self.num_of_shots = 0

        self.edited_text = None
        self.search_called = False
        self.sorted_searched_scenes = None

        self.transition = None
        self.wipe_way = Transition.WIPE_BOTTOM_TO_TOP
        self.fade_color = None

        self.duration_validator = QIntValidator()
        self.duration_validator.setBottom(1)

        self.transition_progress_dialog = None
        self.start_video_transition_worker()

        self.video_searcher = VS.VideoSearcher()

        loader = QUiLoader()
        design_file = QFile("GUIFiles/GUI_Control_window.ui")
        video_window_file = QFile("GUIFiles/gui_Vid_player.ui")

        self.db = GuiDatabaseHandler.GuiDatabaseHandler()

        self.translations = self.db.get_translations()

        if design_file.exists():
            design_file.open(QIODevice.ReadOnly)
            self.main_window = loader.load(design_file)
            design_file.close()

            if self.main_window:
                self.setCentralWidget(self.main_window)
                self.setFixedSize(self.main_window.size())
                self.main_window.transitionButton.clicked.connect(self.transition_button_click)
                self.main_window.descriptionsButton.clicked.connect(self.desc_button_click)
                self.main_window.scenesButton.clicked.connect(self.scene_button_click)
                self.main_window.transitionComboBox.currentIndexChanged.connect(self.transition_kind_change)
                self.main_window.languageComboBox.currentIndexChanged.connect(self.choose_language)
                self.main_window.colorPickerButton.clicked.connect(self.pick_color)
                self.main_window.chooseFileButton.triggered.connect(self.pick_video_file)
                self.main_window.chooseSaveButton.triggered.connect(self.pick_saving_file)
                self.main_window.escapeMenuButton.triggered.connect(self.close_app)
                self.main_window.loadDockerAction.triggered.connect(self.start_docker_builds)

                self.main_window.objectsButtonScene1.clicked.connect(lambda: self.edit_objects_shot(1, self.scene1_id))
                self.main_window.objectsButtonScene2.clicked.connect(lambda: self.edit_objects_shot(2, self.scene2_id))
                self.main_window.objectsButtonScene3.clicked.connect(lambda: self.edit_objects_shot(3, self.scene3_id))

                self.main_window.descButtonScene1.clicked.connect(lambda: self.edit_shot_descriptions(1, self.scene1_id))
                self.main_window.descButtonScene2.clicked.connect(lambda: self.edit_shot_descriptions(2, self.scene2_id))
                self.main_window.descButtonScene3.clicked.connect(lambda: self.edit_shot_descriptions(3, self.scene3_id))

                self.main_window.searchEdit.returnPressed.connect(self.search_shots)

                self.main_window.scenesForwardButton.clicked.connect(self.show_next_shot)
                self.main_window.scenesBackButton.clicked.connect(self.show_previous_shot)

                self.main_window.saveNewDescButton.clicked.connect(self.save_new_desc)
                self.main_window.cancelNewDescButton.clicked.connect(self.cancel_desc_add)

                self.main_window.wipeWayComboBox.currentIndexChanged.connect(self.change_wipe_way)
                self.main_window.transitionsOkButton.clicked.connect(self.add_transitions)
                self.main_window.transitionsPlusButton.clicked.connect(self.show_transition_detail)
                self.main_window.transitionsCancelButton.clicked.connect(self.cancel_transition_parameters)
                self.main_window.every5SecButton.clicked.connect(self.add_transition_to_one)
                self.main_window.randTransitionsButton.clicked.connect(self.add_transition_random)
                self.main_window.randTransitionsDetailButton.clicked.connect(self.add_transition_random_detailed)
                self.main_window.every5SecButtonDetail.clicked.connect(self.add_transition_to_one_detailed)
                self.load_widget_translations()

                self.main_window.wipeLineEdit.setValidator(self.duration_validator)
                self.main_window.fadeLineEdit.setValidator(self.duration_validator)
                self.main_window.dissolveLineEdit.setValidator(self.duration_validator)

        else:
            print("Missing .ui file for control window!")

        if video_window_file.exists():
            video_window_file.open(QIODevice.ReadOnly)
            self.video_window = loader.load(video_window_file)
            video_window_file.close()

            if self.video_window:
                self.video_widget = QVideoWidget(self)
                self.video_window.videoWidget.setParent(None)
                self.video_window.verticalLayout.insertWidget(0, self.video_widget)
                self.media_player = QMediaPlayer(self)
                self.audio_output = QAudioOutput(self)
                self.media_player.setAudioOutput(self.audio_output)

                self.media_player.setVideoOutput(self.video_widget)

                self.load_icons()

                self.video_control_slider = self.video_window.videoTimeSlider
                self.video_control_slider.setRange(0, 100)
                self.video_control_slider.sliderMoved.connect(self.set_video_slider_step)

                self.video_window.playButton.clicked.connect(self.pressed_play)
                self.video_window.stopButton.clicked.connect(self.pressed_pause)
                self.video_window.nextButton.clicked.connect(self.pressed_next)
                self.video_window.previousButton.clicked.connect(self.pressed_previous)

                self.media_player.positionChanged.connect(self.change_slider_position)
                self.media_player.durationChanged.connect(self.set_video_slider_range)

                self.video_window.hide()
            else:
                print("Video window failed to load!")
        else:
            print("Missing .ui file for video window!")

    def transition_button_click(self):
        self.main_window.stackedWidget.setCurrentIndex(0)
        self.main_window.wipeLineEdit.hide()
        self.main_window.wipeTimeHeader.hide()
        self.main_window.wipeWayComboBox.hide()
        self.main_window.wipeWayHeader.hide()
        self.main_window.wipeWayComboBox.setEnabled(False)
        self.main_window.wipeLineEdit.setEnabled(False)
        self.main_window.fadeTimeHeader.hide()
        self.main_window.fadeLineEdit.hide()
        self.main_window.colorPickerButton.hide()
        self.main_window.fadeColorShow.hide()
        self.main_window.fadeColorHeader.hide()
        self.main_window.colorPickerButton.setEnabled(False)
        self.main_window.fadeLineEdit.setEnabled(False)
        self.main_window.dissolveLineEdit.hide()
        self.main_window.dissolveTimeHeader.hide()
        self.main_window.dissolveLineEdit.setEnabled(False)
        if self.video_path:
            video_file_exists = self.db.check_video_file_exists(self.video_path)
            if not video_file_exists:
                print("Continue here")
                add_video = self.__show_approval_box()
                if add_video:
                    self.db.insert_video_file(self.video_path)

    def __show_approval_box(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(" ")
        box_text = "Uložiť video do databázy na generovanie testovacích videí?" if self.language == "SK" else "Save video into database for test video generation?"
        msg_box.setText(box_text)
        msg_box.setIcon(QMessageBox.Question)
        yes_b_text = "Áno" if self.language == "SK" else "Yes"
        no_b_text = "Nie" if self.language == "SK" else "No"

        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.button(QMessageBox.Yes).setText(yes_b_text)
        msg_box.button(QMessageBox.No).setText(no_b_text)

        result = msg_box.exec_()
        return result == QMessageBox.Yes

    def desc_button_click(self):
        self.main_window.stackedWidget.setCurrentIndex(1)

    def scene_button_click(self):
        self.process_video_call()
        self.main_window.stackedWidget.setCurrentIndex(2)

    def search_shots(self):
        if self.desc_path is not None:
            searched_text = self.main_window.searchEdit.text()
            searched_scenes = self.video_searcher.search_descriptions(self.desc_path, searched_text)
            if searched_scenes is not None and len(searched_scenes) > 0:
                self.sorted_searched_scenes = sorted(searched_scenes, key=lambda x: x[1], reverse=True) #ordering by number of results
                self.num_of_shots = len(self.sorted_searched_scenes)

                self.search_called = True
                self.show_frames_first()
            else:
                title = "Vyhľadávanie" if self.language == "SK" else "Search"
                message_text = f"Nenašli sa žiadne výskyty slova: {searched_text}" if self.language == "SK" else f"No search results for word: {searched_text}"
                QMessageBox.information(self, title, message_text)

        return

    def save_new_desc(self):

        new_desc = self.main_window.newDescEdit.toPlainText()
        if self.desc_path is not None:
            search_frame_id = int(self.video_control_slider.value() / 1000 * self.fps)
            if self.desc_path is not None and search_frame_id is not None and new_desc is not None:

                content = etree.iterparse(self.desc_path, events=("end",), tag="scene")
                temp_xml = tempfile.NamedTemporaryFile(delete=False, mode="wb")
                temp_xml.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
                temp_xml.write(b'<descriptions>')

                for event, scene in content:

                    start_frame = scene.find("startFrame")
                    end_frame = scene.find("endFrame")
                    if (start_frame is not None and end_frame is not None
                            and int(start_frame.text) <= int(search_frame_id) <= int(end_frame.text)):
                        new_desc_elem = etree.Element("description")
                        new_desc_elem.text = new_desc
                        scene.append(new_desc_elem)

                    temp_xml.write(etree.tostring(scene, encoding="utf-8"))
                    scene.clear()
                    previous_elem = scene.getprevious()
                    parent_elem = scene.getparent()
                    while parent_elem is not None and previous_elem is not None:
                        parent_elem.remove(previous_elem)
                        previous_elem = scene.getprevious()
                temp_xml.write(b'</descriptions>')
                temp_xml.close()
                shutil.move(temp_xml.name, self.desc_path)

                self.main_window.newDescEdit.clear()
                self.search_shots()

    def edit_objects_shot(self, shot_num, search_scene_id):
        print(f"Edit object shot called for shot number: {shot_num} and scene id: {search_scene_id}")
        self.open_edit_dialog(shot_num, True)
        if self.desc_path is not None and shot_num is not None and search_scene_id is not None and self.edited_text is not None:

            new_objects = self.edited_text.split("\n#\n")
            content = etree.iterparse(self.desc_path, events=("end",), tag="scene")
            temp_xml = tempfile.NamedTemporaryFile(delete=False, mode="wb")
            temp_xml.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
            temp_xml.write(b'<descriptions>')

            for event, scene in content:
                scene_id = scene.find("id")
                if not self.search_called:
                    target_scene_id = str(search_scene_id)
                else:
                    target_scene_id = str(self.sorted_searched_scenes[search_scene_id - 1][0])
                if scene_id is not None and str(scene_id.text) == target_scene_id:

                    print(f"Scene id in xml: {scene_id.text} and searched id is: {target_scene_id}")

                    for obj in scene.findall("object"):
                        scene.remove(obj) #old objects removal

                    for new_obj in new_objects:
                        new_obj_elem = etree.Element("object")
                        new_obj_elem.text = new_obj
                        scene.append(new_obj_elem)

                temp_xml.write(etree.tostring(scene, encoding="utf-8"))
                scene.clear()
                previous_elem = scene.getprevious()
                parent_elem = scene.getparent()
                while parent_elem is not None and previous_elem is not None:
                    parent_elem.remove(previous_elem) #previous old element delete
                    previous_elem = scene.getprevious()
            temp_xml.write(b'</descriptions>')
            temp_xml.close()
            shutil.move(temp_xml.name, self.desc_path)
            if shot_num == 1:
                self.scene1_objects = "\n#\n".join(new_objects)
            if shot_num == 2:
                self.scene2_objects = "\n#\n".join(new_objects)
            if shot_num == 3:
                self.scene3_objects = "\n#\n".join(new_objects)

    def cancel_desc_add(self):
        self.main_window.newDescEdit.clear()


    def edit_shot_descriptions(self, shot_num, search_scene_id):

        self.open_edit_dialog(shot_num, False)
        if self.desc_path is not None and shot_num is not None and search_scene_id is not None and self.edited_text is not None:

            new_descs = self.edited_text.split("\n#\n")
            content = etree.iterparse(self.desc_path, events=("end",), tag="scene")
            temp_xml = tempfile.NamedTemporaryFile(delete=False, mode="wb")
            temp_xml.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
            temp_xml.write(b'<descriptions>')

            for event, scene in content:
                scene_id = scene.find("id")
                if not self.search_called:
                    target_scene_id = str(search_scene_id)
                else:
                    target_scene_id = str(self.sorted_searched_scenes[search_scene_id - 1][0])
                if scene_id is not None and str(scene_id.text) == target_scene_id:
                    for desc in scene.findall("description"):
                        scene.remove(desc)

                    for new_desc in new_descs:
                        new_desc_elem = etree.Element("description")
                        new_desc_elem.text = new_desc
                        scene.append(new_desc_elem)

                temp_xml.write(etree.tostring(scene, encoding="utf-8"))
                scene.clear()
                previous_elem = scene.getprevious()
                parent_elem = scene.getparent()
                while parent_elem is not None and previous_elem is not None:
                    parent_elem.remove(previous_elem)
                    previous_elem = scene.getprevious()
            temp_xml.write(b'</descriptions>')
            temp_xml.close()
            shutil.move(temp_xml.name, self.desc_path)
            if shot_num == 1:
                self.scene1_descriptions = "\n#\n".join(new_descs)
                self.main_window.scene1Desc.setPlainText(self.scene1_descriptions)
            if shot_num == 2:
                self.scene2_descriptions = "\n#\n".join(new_descs)
                self.main_window.scene2Desc.setPlainText(self.scene2_descriptions)
            if shot_num == 3:
                self.scene3_descriptions = "\n#\n".join(new_descs)
                self.main_window.scene3Desc.setPlainText(self.scene3_descriptions)


    def open_edit_dialog(self, shot_num, is_object):
        edit_dialog = QFile("GUIFiles/edit_dialog.ui")
        loader = QUiLoader()
        if edit_dialog.exists():
            edit_dialog.open(QIODevice.ReadOnly)
            edit_window = loader.load(edit_dialog)
            edit_dialog.close()

            button_box = edit_window.buttonBox
            ok_button = button_box.button(QDialogButtonBox.Ok)
            cancel_button = button_box.button(QDialogButtonBox.Cancel)

            if ok_button is not None:
                ok_button.setText("Potvrdiť" if self.language == "SK" else "Ok")
            if cancel_button is not None:
                cancel_button.setText("Zrušiť" if self.language == "SK" else "Cancel")
            if is_object:
                edit_window.mainLabel.setText("Objekty:" if self.language == "SK" else "Objects:")
            else:
                edit_window.mainLabel.setText("Popis:" if self.language == "SK" else "Description:")

            text_edit = edit_window.textEdit
            edit_dialog_buttons = edit_window.buttonBox

            edit_dialog_buttons.accepted.connect(edit_window.accept)
            edit_dialog_buttons.rejected.connect(edit_window.reject)

            if shot_num is None:
                return
            else:
                if shot_num == 1:
                    if is_object:

                        text_edit.insertPlainText(self.scene1_objects)
                    else:
                        text_edit.insertPlainText(self.scene1_descriptions)
                if shot_num == 2:
                    if is_object:
                        text_edit.insertPlainText(self.scene2_objects)
                    else:
                        text_edit.insertPlainText(self.scene2_descriptions)
                if shot_num == 3:
                    if is_object:
                        text_edit.insertPlainText(self.scene3_objects)
                    else:
                        text_edit.insertPlainText(self.scene3_descriptions)

            result = edit_window.exec()

            if result == QDialog.Accepted:
                self.edited_text = text_edit.toPlainText()
                print(f"edited text: {self.edited_text}")
            else:
                self.edited_text = None
                print("Dialog canceled")


    def transition_kind_change(self):

        selected_transition = self.main_window.transitionComboBox.currentText()

        if selected_transition == "Wipe":
            self.main_window.transitionSpecifics.setCurrentIndex(0)
            self.transition = "wipe"
            self.wipe_way = Transition.WIPE_BOTTOM_TO_TOP
        else:
            self.wipe_way = None
        if selected_transition == "Fade in" or selected_transition == "Fade out":
            self.main_window.transitionSpecifics.setCurrentIndex(1)
            if selected_transition == "Fade in":
                self.transition = "Fade_in"
            else:
                self.transition = "Fade_out"
        if selected_transition == "Dissolve":
            self.main_window.transitionSpecifics.setCurrentIndex(2)
            self.transition = "Dissolve"
        if selected_transition == "Cut":
            self.main_window.transitionSpecifics.setCurrentIndex(4)
            self.transition = "Cut"
        if selected_transition == "Iné" or selected_transition == "Other":
            self.main_window.transitionSpecifics.setCurrentIndex(3)
            self.transition = None

    def change_wipe_way(self):
        chosen_wipe = self.main_window.wipeWayComboBox.currentText()
        if chosen_wipe == "Nahor" or chosen_wipe == "Up":
            self.wipe_way = Transition.WIPE_BOTTOM_TO_TOP
        if chosen_wipe == "Nadol" or chosen_wipe == "Down":
            self.wipe_way = Transition.WIPE_TOP_TO_BOTTOM
        if chosen_wipe == "Do ľava" or chosen_wipe == "Left":
            self.wipe_way = Transition.WIPE_LEFT
        if chosen_wipe == "Do prava" or chosen_wipe == "Right":
            self.wipe_way = Transition.WIPE_RIGHT

    def add_transitions(self):
        if self.video_path is None:
            self._show_video_warning()
        else:
            selected_transition = self.main_window.transitionComboBox.currentText()
            time = self.video_control_slider.value() / 1000
            until_end = int(self.media_player.duration() / 1000) - math.ceil(time)
            warning_title = "Nesprávna hodnota" if self.language == "SK" else "Invalid value"
            warning_text = "Povolené sú len hodnoty väčšie ako 1 a menšie ako čas do konca videa" if self.language == "SK" \
                else "Allowed only values bigger than 1 and smaller than time till video ending"

            if selected_transition != "Iné" and selected_transition != "Other":

                if selected_transition == "Wipe":
                    if self.main_window.wipeLineEdit.text() and not self.main_window.wipeLineEdit.hasAcceptableInput():
                        QMessageBox.warning(self, warning_title, warning_text)
                        return
                    if self.main_window.wipeLineEdit.text() and self.main_window.wipeLineEdit.hasAcceptableInput() and (until_end - int(self.main_window.wipeLineEdit.text()) < 1):
                        QMessageBox.warning(self, warning_title, warning_text)
                        return
                    self.transition_loader_show()
                    duration = int(self.main_window.wipeLineEdit.text()) if self.main_window.wipeLineEdit.text() else 1
                    self.trigger_add_transitions_by_choice.emit(self.video_path, self.wipe_way, 0, 0, time, 0, "", duration, None)
                if selected_transition == "Fade in":
                    if self.main_window.fadeLineEdit.text() and not self.main_window.fadeLineEdit.hasAcceptableInput():
                        QMessageBox.warning(self, warning_title, warning_text)
                        return
                    if self.main_window.fadeLineEdit.text() and self.main_window.fadeLineEdit.hasAcceptableInput() and (until_end - int(self.main_window.fadeLineEdit.text()) < 1):
                        QMessageBox.warning(self, warning_title, warning_text)
                        return
                    self.transition_loader_show()
                    duration = int(self.main_window.fadeLineEdit.text()) if self.main_window.fadeLineEdit.text() else 1
                    self.trigger_add_transitions_by_choice.emit(self.video_path, Transition.FADE_IN, 0,0, time, 0,"", duration,
                                                           self.fade_color)
                if selected_transition == "Fade out":
                    if self.main_window.fadeLineEdit.text() and not self.main_window.fadeLineEdit.hasAcceptableInput():
                        QMessageBox.warning(self, warning_title, warning_text)
                        return
                    if self.main_window.fadeLineEdit.text() and self.main_window.fadeLineEdit.hasAcceptableInput() and (until_end - int(self.main_window.fadeLineEdit.text()) < 1):
                        QMessageBox.warning(self, warning_title, warning_text)
                        return
                    self.transition_loader_show()
                    duration = int(self.main_window.fadeLineEdit.text()) if self.main_window.fadeLineEdit.text() else 1
                    self.trigger_add_transitions_by_choice.emit(self.video_path, Transition.FADE_OUT, 0, 0, time, 0, "", duration,
                                                           self.fade_color)

                if selected_transition == "Dissolve":
                    if self.main_window.dissolveLineEdit.text() and not self.main_window.dissolveLineEdit.hasAcceptableInput():
                        QMessageBox.warning(self, warning_title, warning_text)
                        return
                    if self.main_window.dissolveLineEdit.text() and self.main_window.dissolveLineEdit.hasAcceptableInput() and (until_end - int(self.main_window.dissolveLineEdit.text()) < 1):
                        QMessageBox.warning(self, warning_title, warning_text)
                        return
                    self.transition_loader_show()
                    duration = int(self.main_window.dissolveLineEdit.text()) if self.main_window.dissolveLineEdit.text() else 1
                    self.trigger_add_transitions_by_choice.emit(self.video_path, Transition.DISSOLVE, 0, 0, time, 0, "", duration, None)
                if selected_transition == "Cut":
                    self.transition_loader_show()
                    self.trigger_add_transitions_by_choice.emit(self.video_path, Transition.CUT, 0, 0, time, 0, "", 1, None)
            if selected_transition == "Iné" or selected_transition == "Other":
                print("Chosen OTHER")

    def show_transition_detail(self):
        selected_transition = self.main_window.transitionComboBox.currentText()
        if selected_transition == "Wipe":
            self.main_window.wipeLineEdit.show()
            self.main_window.wipeTimeHeader.show()
            self.main_window.wipeWayComboBox.show()
            self.main_window.wipeWayHeader.show()
            self.main_window.wipeWayComboBox.setEnabled(True)
            self.main_window.wipeLineEdit.setEnabled(True)
        if selected_transition == "Fade in" or selected_transition == "Fade out":
            self.main_window.fadeTimeHeader.show()
            self.main_window.fadeLineEdit.show()
            self.main_window.colorPickerButton.show()
            self.main_window.fadeColorShow.show()
            self.main_window.fadeColorHeader.show()
            self.main_window.colorPickerButton.setEnabled(True)
            self.main_window.fadeLineEdit.setEnabled(True)
        if selected_transition == "Dissolve":
            self.main_window.dissolveLineEdit.show()
            self.main_window.dissolveTimeHeader.show()
            self.main_window.dissolveLineEdit.setEnabled(True)

    def cancel_transition_parameters(self):
        print("Cancelling transition parameters")
        self.main_window.wipeLineEdit.clear()
        self.main_window.fadeLineEdit.clear()
        self.main_window.dissolveLineEdit.clear()
        self.main_window.fadeColorShow.setStyleSheet(f"background-color: rgb(0,0,0);")
        self.fade_color = None

        self.main_window.wipeLineEdit.hide()
        self.main_window.wipeTimeHeader.hide()
        self.main_window.wipeWayComboBox.hide()
        self.main_window.wipeWayHeader.hide()
        self.main_window.wipeWayComboBox.setEnabled(False)
        self.main_window.wipeLineEdit.setEnabled(False)
        self.main_window.fadeTimeHeader.hide()
        self.main_window.fadeLineEdit.hide()
        self.main_window.colorPickerButton.hide()
        self.main_window.fadeColorShow.hide()
        self.main_window.fadeColorHeader.hide()
        self.main_window.colorPickerButton.setEnabled(False)
        self.main_window.fadeLineEdit.setEnabled(False)
        self.main_window.dissolveLineEdit.hide()
        self.main_window.dissolveTimeHeader.hide()
        self.main_window.dissolveLineEdit.setEnabled(False)

    def pick_color(self):
        color = QColorDialog.getColor()

        if color.isValid():
            self.fade_color = color
            self.main_window.fadeColorShow.setText("")
            self.main_window.fadeColorShow.setStyleSheet(f"background-color: {color.name()};")
            print(f"Color chosen for fade: {color.name()}")
            rgb_val = [color.red(), color.green(), color.blue()]
            print(f"RGB values of color: {rgb_val}")
            self.fade_color = rgb_val

    def pick_video_file(self):

        choose_file_header = "Vyberte video" if self.language == "SK" else "Choose a video file"
        video_path = QFileDialog.getOpenFileName(self,
                                                 f"{choose_file_header}",
                                                 "",
                                                 "Video Files (*.mp4 *.avi)")
        if not video_path[0]:
            return

        print(f"Chosen video: {video_path[0]}")
        self.video_path = video_path[0]

        self.media_player.stop()
        self.media_player.setSource(QUrl())

        if self.video is not None:

            try:
                self.video.reader.close()
                if self.video.audio:
                    self.video.audio.reader.close_proc()
            except Exception as e:
                print("Error closing previous video file clip: ", e)
            self.video = None
            self.fps = 0

        if self.video_window:
            self.video_window.show()

        self.media_player.setSource(QUrl.fromLocalFile(self.video_path))


        self.video_widget.show()
        self.pressed_play()
        self.search_called = False

        if self.video is None:
            self.video = VideoFileClip(self.video_path)
        if self.fps == 0:
            self.fps = self.video.fps
        if self.num_of_shots != 0:
            self.num_of_shots = 0
        self.desc_path = self.db.get_desc_file(self.video_path, self.language)


    def pick_saving_file(self):
        choose_dir_header = "Vyberte súbor" if self.language == "SK" else "Choose a file"
        video_dir = QFileDialog.getExistingDirectory(self, choose_dir_header, "/home", QFileDialog.ShowDirsOnly)
        if video_dir:
            self.save_dir = video_dir

    def choose_language(self):
        self.language = self.main_window.languageComboBox.currentText()
        print(f"Language changed to {self.language}")
        self.load_widget_translations()

    def pressed_play(self):
        if self.media_player.source().isEmpty():
            print("No source video!")
        else:
            self.media_player.play()

    def pressed_pause(self):
        self.media_player.pause()

    def pressed_next(self):
        self.media_player.setPosition(self.media_player.position() + 10000)

    def pressed_previous(self):
        self.media_player.setPosition(max(self.media_player.position() - 10000, 0))

    def set_video_slider_range(self, video_length):
        self.video_control_slider.setRange(0, video_length)
        current_duration = QTime(0, 0, 0).addMSecs(self.media_player.duration())
        duration_str = current_duration.toString("hh:mm:ss")
        self.video_window.videoDuration.setText(duration_str)

    def change_slider_position(self, position):

        if self.media_player.duration() > 0:
            self.video_control_slider.setValue(int(position))
            self.update_video_time(self.media_player.position())
            # print("Slider position changed to", position)
        else:
            print("Duration is 0")

    def update_video_time(self, current_time):
        time = QTime(0, 0, 0).addMSecs(current_time)
        time_str = time.toString("hh:mm:ss")
        self.video_window.currentTime.setText(time_str)

    def set_video_slider_step(self, video_step):
        if video_step > 0:
            self.media_player.setPosition(int(video_step))

    def start_video_transition_worker(self):
        self.transition_thread = QThread()
        self.transition_worker = VideoGeneratorWorker()

        self.transition_worker.moveToThread(self.transition_thread)

        self.trigger_add_transitions_to_one.connect(self.transition_worker.add_transitions_to_one)
        self.trigger_add_transitions_rand.connect(self.transition_worker.add_transitions_random)
        self.trigger_add_transitions_by_choice.connect(self.transition_worker.add_transition_by_choice)

        self.transition_worker.finished_task_1.connect(self.on_finished_transition_to_one)
        self.transition_worker.finished_task_2.connect(self.on_transition_by_choice)
        self.transition_worker.finished_task_3.connect(self.on_transition_random)

        self.transition_thread.start()

    @Slot(bool)
    def on_finished_transition_to_one(self, success):
        if success:
            print("Finished adding transitions into single video")
        else:
            print("Adding transitions to single video NOT finished")
        self.close_transition_progress(success)

    @Slot(bool)
    def on_transition_by_choice(self, success):
        if success:
            print("Finished adding transitions by choice")
        else:
            print("Adding transitions by choice NOT finished")
        self.close_transition_progress(success)

    @Slot(bool)
    def on_transition_random(self, success):
        if success:
            print("Finished creating random combination of transitions")
        else:
            print("Creating random combination of transitions NOT finished")
        self.close_transition_progress(success)

    def add_transition_to_one(self):
        print("Add transition to one START")
        if self.video_path is not None:

            self.transition_loader_show()
            self.trigger_add_transitions_to_one.emit(self.video_path, 5)

    def add_transition_to_one_detailed(self):
        print("Add transition to one detailed called")
        if self.video_path is None:
            self._show_video_warning()
        else:
            title = "Výber n sekúnd" if self.language == "SK" else "Choosing number of seconds"
            message_text = "Vyberte ako často v sekundách chcete pridať prechod:" if self.language == "SK"\
                else "Choose how often in seconds do you wish to add a transition"
            value, ok = QInputDialog.getInt(self, title, message_text)
            if ok:
                self.transition_loader_show()
                self.trigger_add_transitions_to_one.emit(self.video_path, value)

    def _show_video_warning(self):
        title = "Varovanie" if self.language == "SK" else "Warning"
        message_text = "Potrebné vybrať video!" if self.language == "SK" else "Choose video first!"
        QMessageBox.warning(self, title, message_text)

    def add_transition_random_detailed(self):
        print("Add transition to random detailed called")
        title = "Výber počtu videí" if self.language == "SK" else "Choosing number of videos"
        message_text = "Vyberte počet videí na zostavenie nového videa" if self.language == "SK" \
            else "Choosenumber of videos to connect into new one"
        value, ok = QInputDialog.getInt(self, title, message_text)
        if ok:
            self.transition_loader_show()
            self.trigger_add_transitions_rand.emit(value)


    def add_transition_random(self):
        print("Add transition to random START")
        self.transition_loader_show()
        self.trigger_add_transitions_rand.emit(2)

    def transition_loader_show(self):

        loading_text = "Prebieha pridávanie prechodov" if self.language == "SK" \
            else "Adding transitions in progress"
        cancel_text = "Zrušiť" if self.language == "SK" else "Cancel"

        title_text = "Spracovanie prechodov" if self.language == "SK" else "Transition processing"

        self.transition_progress_dialog = QProgressDialog(loading_text,
                                                          cancel_text, 0, 0, self)

        self.transition_progress_dialog.setWindowTitle(title_text)
        self.transition_progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.transition_progress_dialog.setCancelButton(None)
        self.transition_progress_dialog.setValue(0)
        self.transition_progress_dialog.setAttribute(Qt.WA_DeleteOnClose)

        self.transition_worker.progress_task_1.connect(self.on_transition_progress_update)
        self.transition_worker.progress_task_2.connect(self.on_transition_progress_update)
        self.transition_worker.progress_task_3.connect(self.on_transition_progress_update)

        self.transition_progress_dialog.show()

    @Slot(int)
    def on_transition_progress_update(self, value):
        if self.transition_progress_dialog:
            self.transition_progress_dialog.setValue(value)

    def close_transition_progress(self, success):
        if self.transition_progress_dialog:
            self.transition_progress_dialog.close()
            self.transition_progress_dialog = None
        if success:
            title = "Úspečne dokončené" if self.language == "SK" else "Success"
            message_text = "Prechody spracované" if self.language == "SK" else "Transitions processed successfully"
            QMessageBox.information(self, title, message_text)
        else:
            title = "Varovanie" if self.language == "SK" else "Warning"
            message_text = "Prechod sa už spracováva" if self.language == "SK" else "Transition is already being processed"
            QMessageBox.warning(self, title, message_text)

    def process_video_call(self):
        if not hasattr(self, "video_path") or not self.video_path:
            print("No video for processing chosen")
            return
        else:
            loading_text = "Video sa spracováva. Proces môže chvíľu trvať" if self.language == "SK" \
                else "Video is being processed. Proces may take some time"
            cancel_text = "Zrušiť" if self.language == "SK" else "Cancel"
            self.vid_proc_progress_dialog = QProgressDialog(loading_text, cancel_text, 0, 0, self)
            title_text = "Spracovanie videa" if self.language == "SK" else "Video processing"
            self.vid_proc_progress_dialog.setWindowTitle(title_text)
            self.vid_proc_progress_dialog.setWindowModality(Qt.ApplicationModal)
            self.vid_proc_progress_dialog.setCancelButton(None)
            self.vid_proc_progress_dialog.setValue(0)
            self.vid_proc_progress_dialog.show()

            self.vid_process_thread = QThread()
            if hasattr(self, "save_dir"):
                saving_dir = self.save_dir
            else:
                saving_dir = None

            self.worker = GUIThreadWorker(self.video_path, saving_dir, self.threshold, self.image_size)
            self.worker.moveToThread(self.vid_process_thread)

            self.vid_process_thread.started.connect(self.worker.run_video_proc)

            self.worker.finished.connect(self.close_video_progress)
            self.worker.finished.connect(self.load_desc_file)
            self.worker.finished.connect(self.show_frames_first)
            self.worker.finished.connect(self.worker.close_database)
            self.worker.finished.connect(self.vid_process_thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.vid_process_thread.finished.connect(self.vid_process_thread.deleteLater)
            self.vid_process_thread.start()

    def show_frames_first(self):
        if self.video is None:
            self.video = VideoFileClip(self.video_path)
        if self.fps == 0:
            self.fps = self.video.fps

        self.scene1_id = 1
        self.scene2_id = 2
        self.scene3_id = 3
        if self.video_path:
            self.desc_path = self.db.get_desc_file(self.video_path, self.language)
        else:
            print(f'No video file chosen!')
            title = "Varovanie" if self.language == "SK" else "Warning"
            message_text = "Nebolo vybrané žiadne video!" if self.language == "SK" else "No video file was chosen!"
            QMessageBox.warning(self, title, message_text)
            return
        self.load_extracted_frames_to_window(self.desc_path)

    def show_searched_shots_first(self, sorted_searched_scenes, searched_word):
        if sorted_searched_scenes is not None:
            if len(sorted_searched_scenes) >= 1:
                self.scene1_id = sorted_searched_scenes[0][0]
            else:
                title = "Vyhľadávanie" if self.language == "SK" else "Search"
                message_text = f"Nenašli sa žiadne výskyty slova: {searched_word}" if self.language == "SK" else "No search results for word: {searched_word}"
                QMessageBox.information(self, title, message_text)
            if len(sorted_searched_scenes) >= 2:
                self.scene2_id = sorted_searched_scenes[1][0]
            else:
                self.scene2_id = None
            if len(sorted_searched_scenes) >= 3:
                self.scene3_id = sorted_searched_scenes[2][0]
            else:
                self.scene3_id = None
            self.num_of_shots = len(sorted_searched_scenes)
            self.load_extracted_frames_to_window(self.desc_path)

    def load_desc_file(self):
        self.desc_path = self.db.get_desc_file(self.video_path, self.language)

    def load_extracted_frames_to_window(self, des_path):
        if des_path is not None and des_path != "":
            print(f"FOUND scene ids: {self.scene1_id} , {self.scene2_id} , {self.scene3_id}")
            scene_ids = []
            self.main_window.scene1Time.setText("")
            self.main_window.scene2Time.setText("")
            self.main_window.scene3Time.setText("")
            if not self.search_called or (len(self.sorted_searched_scenes) == 0):
                scene_ids = [self.scene1_id, self.scene2_id, self.scene3_id]
            if self.search_called:
                search_id_1 = self.sorted_searched_scenes[self.scene1_id-1][0]
                scene_ids.append(search_id_1)
                if len(self.sorted_searched_scenes) >= 2:
                    scene_ids.append(self.sorted_searched_scenes[self.scene2_id-1][0])
                if len(self.sorted_searched_scenes) >= 3:
                    scene_ids.append(self.sorted_searched_scenes[self.scene3_id-1][0])

            print(f"Scene ids in showing method: {scene_ids}")
            scene_contents = self.get_scenes_start(des_path, scene_ids)

            if len(scene_contents) == 0:
                print("No scenes found!")
                return
            if len(scene_contents) >= 1:

                scene_1_start = int(scene_contents[0]["start_frame"])
                scene_1_time = self.format_time_from_frame(scene_1_start)
                self.main_window.scene1Time.setText(scene_1_time)

                img_1 = self.load_video_frame(scene_1_start, self.fps)
                self.main_window.scene1.setPixmap(img_1.scaled(self.main_window.scene1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                self.scene1_objects = "\n#\n".join(scene_contents[0]["objects"])
                self.scene1_descriptions = "\n#\n".join(scene_contents[0]["descriptions"])

                self.main_window.scene1Desc.setPlainText(self.scene1_descriptions)

            if len(scene_contents) >= 2:
                scene_2_start = int(scene_contents[1]["start_frame"])
                scene_2_time = self.format_time_from_frame(scene_2_start)
                self.main_window.scene2Time.setText(scene_2_time)
                img_2 = self.load_video_frame(scene_2_start, self.fps)
                self.main_window.scene2.setPixmap(img_2.scaled(self.main_window.scene2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                self.scene2_objects = "\n#\n".join(scene_contents[1]["objects"])
                self.scene2_descriptions = "\n#\n".join(scene_contents[1]["descriptions"])

                self.main_window.scene2Desc.setPlainText(self.scene2_descriptions)
            else:
                self.main_window.scene2.clear()
                self.main_window.scene2.setStyleSheet("background-color: rgb(200,200, 200)")
                self.scene2_objects = None
                self.scene2_descriptions = None

                self.main_window.scene2Desc.setPlainText(self.scene2_descriptions)

            if len(scene_contents) == 3:
                scene_3_start = int(scene_contents[2]["start_frame"])
                scene_3_time = self.format_time_from_frame(scene_3_start)
                self.main_window.scene3Time.setText(scene_3_time)
                img_3 = self.load_video_frame(scene_3_start, self.fps)
                self.main_window.scene3.setPixmap(img_3.scaled(self.main_window.scene3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                self.scene3_objects = "\n#\n".join(scene_contents[2]["objects"])
                self.scene3_descriptions = "\n#\n".join(scene_contents[2]["descriptions"])
                self.main_window.scene3Desc.setPlainText(self.scene3_descriptions)
            else:
                self.main_window.scene3.clear()
                self.main_window.scene3.setStyleSheet("background-color: rgb(200,200, 200)")
                self.scene3_objects = None
                self.scene3_descriptions = None
                self.main_window.scene3Desc.setPlainText(self.scene3_descriptions)
        else:
            title = "Varovanie" if self.language == "SK" else "Warning"
            message_text = f"Nenájdený súbor s popisom v jazyku {self.language}" if self.language == "SK" \
                else f"Description file not found in language {self.language}"
            QMessageBox.warning(self, title, message_text)


    def show_next_shot(self):

        print(f"Values sent to NEXT: {self.scene1_id}, {self.scene2_id}, {self.scene3_id}")
        scene1_id_temp = (self.scene1_id % self.num_of_shots) + 1
        scene2_id_temp = (self.scene2_id % self.num_of_shots) + 1
        scene3_id_temp = (self.scene3_id % self.num_of_shots) + 1
        print(f"Temporary values for next: {scene1_id_temp}, {scene2_id_temp}, {scene3_id_temp}")
        self.scene1_id = scene1_id_temp
        self.scene2_id = scene2_id_temp
        self.scene3_id = scene3_id_temp

        print(f"Scene ids: {self.scene1_id}, {self.scene2_id}, {self.scene3_id}")
        self.load_extracted_frames_to_window(self.desc_path)

    def show_previous_shot(self):
        print(f"Scene id 1: {self.scene1_id} a nove by malo byt {self.scene1_id - 1}")
        print(f"Values sent to PREVIOUS: {self.scene1_id}, {self.scene2_id}, {self.scene3_id}")
        scene1_id_temp = ((self.scene1_id - 2 + self.num_of_shots) % self.num_of_shots) + 1
        scene2_id_temp = ((self.scene2_id - 2 + self.num_of_shots) % self.num_of_shots) + 1
        scene3_id_temp = ((self.scene3_id - 2 + self.num_of_shots) % self.num_of_shots) + 1
        print(f"Temporary values for previous: {scene1_id_temp}, {scene2_id_temp}, {scene3_id_temp}")

        self.scene1_id = scene1_id_temp
        self.scene2_id = scene2_id_temp
        self.scene3_id = scene3_id_temp
        print(f"Scene ids: {self.scene1_id}, {self.scene2_id}, {self.scene3_id}")

        self.load_extracted_frames_to_window(self.desc_path)

    def load_video_frame(self, frame_id, fps):

        time_index = frame_id / fps
        pic = self.video.get_frame(time_index)
        height, width, channels = pic.shape if len(pic.shape) == 3 else (pic.shape[0], pic.shape[1], 1)
        bytes_per_line = width * channels
        qimg = None

        if channels == 1:
            qimg = QImage(pic.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif channels == 3:
            qimg = QImage(pic.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        else:
            print(f'channels not supported for frame with id {frame_id}')

        if qimg:
            pixmap = QPixmap.fromImage(qimg)
            return pixmap

    def get_scenes_start(self, des_path, searched_ids):
        contents = etree.iterparse(des_path, events=("end",), tag="scene")
        count = 0
        print(f"Number of shots: {self.num_of_shots}")
        scene_contents = []
        for event, scene in contents:
            if len(scene_contents) < len(searched_ids):
                scene_id_elem = scene.find('id')
                if scene_id_elem is not None:
                    scene_id = int(scene_id_elem.text)
                else:
                    continue
                if scene_id in searched_ids:
                    objects = scene.findall('object')
                    obj_contents = []
                    for obj in objects:
                        obj_contents.append(obj.text)

                    descriptions = scene.findall('description')
                    desc_contents = []
                    for desc in descriptions:
                        desc_contents.append(desc.text)

                    start_frame = scene.find('startFrame').text
                    end_frame = scene.find('endFrame').text

                    scene_contents.append({
                        "scene_id": scene_id,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "objects": obj_contents,
                        "descriptions": desc_contents
                    })

            scene.clear()
            count += 1
        if self.num_of_shots == 0 and not self.search_called:
            self.num_of_shots = count
        scene_contents = sorted(scene_contents, key=lambda shot: searched_ids.index((shot['scene_id'])))
        return scene_contents

    def close_video_progress(self, success):
        self.vid_proc_progress_dialog.close()
        if success:
            title = "Úspečne dokončené" if self.language == "SK" else "Success"
            message_text = "Video úspešne spracované" if self.language == "SK" else "Video processed successfully"
            QMessageBox.information(self, title, message_text)
        else:
            title = "Varovanie" if self.language == "SK" else "Warning"
            message_text = "Video sa už spracováva" if self.language == "SK" else "Video is already being processed"
            QMessageBox.warning(self, title, message_text)

    def start_docker_builds(self):
        loading_text = "Vytváranie Docker kontajnerov. Proces môže chvíľu trvať" if self.language == "SK" \
            else "Creating Docker containers. Proces may take some time"
        cancel_text = "Zrušiť" if self.language == "SK" else "Cancel"
        docker_call_dialog = QProgressDialog(loading_text, cancel_text, 0, 0, self)
        title_text = "Docker načítanie" if self.language == "SK" else "Docker load"
        docker_call_dialog.setWindowTitle(title_text)
        docker_call_dialog.setWindowModality(Qt.ApplicationModal)
        docker_call_dialog.setCancelButton(None)
        docker_call_dialog.setValue(0)
        docker_call_dialog.show()

        docker_call_thread = QThread()

        docker_worker = DockerStarterWorker()
        docker_worker.moveToThread(docker_call_thread)

        docker_call_thread.started.connect(docker_worker.run)
        docker_worker.finished.connect(docker_call_dialog.close)
        docker_worker.finished.connect(docker_call_thread.quit)
        docker_worker.finished.connect(docker_worker.deleteLater)
        docker_worker.finished.connect(self.on_docker_worker_finished)
        docker_call_thread.finished.connect(docker_call_thread.deleteLater)
        docker_call_thread.start()

        return

    @Slot(bool)
    def on_docker_worker_finished(self, success):
        if success:
            title = "Úspech" if self.language == 'SK' else "Success"
            message_text = "Docker kontajnery úspešne vytvorené" if self.language == 'SK' else "Docker containers successfully loaded"
            QMessageBox.information(self, title, message_text)
        else:
            title = "Chyba" if self.language == 'SK' else "Error"
            message_text = "Vytváranie kontajnerov zlyhalo" if self.language == 'SK' else "Failed to create Docker containers."
            QMessageBox.warning(self, title, message_text)

    def close_app(self):
        QApplication.instance().quit()

    def load_widget_translations(self):
        print("Translating widgets")
        if self.translations is not None:
            for widget in self.main_window.findChildren(QWidget):
                widget_name = str(widget.objectName())
                translation = self.translations.get(widget_name)
                if translation is not None:
                    widget_text = self._get_current_language_text(translation)
                    if widget_text is None:
                        return
                    if str(translation["parent"]) == "main_window":
                        if widget_name != "searchEdit":
                            if widget_name != "menuFile":
                                widget.setText(widget_text)
                            else:
                                widget.setTitle(widget_text)
                        else:
                            widget.setPlaceholderText(widget_text)
                if widget_name == "wipeWayComboBox":
                    for i in range(0, 4):
                        search_widget = f"{widget_name}_{i}"
                        translation = self.translations.get(search_widget)
                        if str(translation["parent"]) == "wipe_way":
                            widget_text = self._get_current_language_text(translation)
                            if widget_text is None:
                                return
                            widget.setItemText(i, widget_text)

                if widget_name == "transitionComboBox":
                    translation = self.translations.get("transitionComboBox_5")
                    widget_text = self._get_current_language_text(translation)
                    if widget_text is None:
                        return
                    widget.setItemText(5, widget_text)
            for action in self.main_window.findChildren(QAction):
                action_name = str(action.objectName())
                translation = self.translations.get(action_name)
                if translation is not None:
                    action_text = self._get_current_language_text(translation)
                    if action_text is None:
                        return
                    if str(translation["parent"]) == "main_window":
                        if action_text != "menuFile":
                            action.setText(action_text)
                        else:
                            action.setTitle(action_text)
        else:
            print("No translations")

    def format_time_from_frame(self, frame_number):
        seconds = frame_number / self.fps
        formated_time = QTime(0, 0, 0).addSecs(seconds)
        formated_time = formated_time.toString("hh:mm:ss")
        return formated_time

    def _get_current_language_text(self, translation):
        if self.language == "SK":
            widget_text = str(translation["sk_translation"])
        elif self.language == "EN":
            widget_text = str(translation["en_translation"])
        else:
            print("Language not valid")
            return None
        return widget_text

    def load_icons(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        icon_folder = os.path.join(current_dir, 'GUIFiles', 'icons')
        play_icon_path = os.path.join(icon_folder, 'play_new.png')
        stop_icon_path = os.path.join(icon_folder, 'stop_new.png')
        next_icon_path = os.path.join(icon_folder, 'next_new.png')
        previous_icon_path = os.path.join(icon_folder, 'previous_new.png')
        self.video_window.playButton.setIcon(QIcon(play_icon_path))
        self.video_window.stopButton.setIcon(QIcon(stop_icon_path))
        self.video_window.nextButton.setIcon(QIcon(next_icon_path))
        self.video_window.previousButton.setIcon(QIcon(previous_icon_path))


    def closeEvent(self, event):
        if hasattr(self, "transition_thread"):
            self.transition_thread.quit()
            self.transition_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ControlWindow()
    window.show()
    app.exec()

