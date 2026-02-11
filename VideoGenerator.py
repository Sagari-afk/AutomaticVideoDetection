import datetime
import math

import mysql.connector

import config
import random
import AnnotationSaver
import numpy as np
from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET
import XmlHandler
import os


from moviepy.editor import ColorClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx


class Transition(Enum):
    CUT = 1
    DISSOLVE = 2
    WIPE_LEFT = 3
    WIPE_RIGHT = 4
    WIPE_TOP_TO_BOTTOM = 5
    WIPE_BOTTOM_TO_TOP = 6
    FADE_IN = 7
    FADE_OUT = 8

    def get_by_name(self, name):
        try:
            found = Transition[name.upper()]
            return found.value
        except Exception:
            raise Exception("No transition matching required transition: " + name)


def add_wipe(clip, duration, side):

    clip = clip.add_mask()
    mask = clip.mask
    width, height = clip.size

    def wipe_mask(get_frame, t):
        progress = min(1, max(0, t / duration))
        mask_frame = np.zeros((height, width))

        if side == "left":
            mask_frame[:, :int(width * progress)] = 1
        elif side == "right":
            mask_frame[:, int(width * (1 - progress)):] = 1
        elif side == "top":
            mask_frame[:int(height * progress), :] = 1
        elif side == "bottom":
            mask_frame[int(height * (1 - progress)):, :] = 1

        return mask_frame

    mask = mask.fl(wipe_mask, keep_duration=True)
    clip.mask = mask
    return clip


class VideoGenerator:

    db = None
    cursor = None

    def __init__(self):
        self.main_dir = Path(__file__).parent.absolute()

    def cut(self, video1, video2):

        max_width = max(video1.size[0], video2.size[0])
        max_height = max(video1.size[1], video2.size[1])
        video1.set_position('center')
        video2.set_position('center')

        compos_vid_1 = CompositeVideoClip([ColorClip((max_width, max_height), duration=video1.duration,
                                                     color=(0, 0, 0)), video1])

        compos_vid_2 = CompositeVideoClip([ColorClip((max_width, max_height), duration=video2.duration,
                                                     color=(0, 0, 0)), video2])

        result = concatenate_videoclips([compos_vid_1, compos_vid_2], method='chain')
        return result

    def dissolve(self, video1, video2, transition_duration):
        adj_vid1 = video1
        adj_vid2 = video2.set_start(video1.duration - transition_duration)
        adj_vid2 = adj_vid2.crossfadein(transition_duration)
        result = CompositeVideoClip([adj_vid1, adj_vid2])
        result.set_position('center')
        if result.duration is None:
            result.duration = video1.duration + video2.duration - transition_duration
        return result

    def wipe_left(self, base_video, next_video, transition_duration):

        wipe_vid = add_wipe(next_video, transition_duration, "left")
        result = CompositeVideoClip([base_video, wipe_vid.set_start(base_video.duration - transition_duration).set_position('center')])
        if result.duration is None:
            result.duration = base_video.duration + wipe_vid.duration - transition_duration
        return result

    def wipe_right(self, base_video, next_video, transition_duration):

        wipe_vid = add_wipe(next_video, transition_duration, "right")
        result = CompositeVideoClip([base_video, wipe_vid.set_start(base_video.duration - transition_duration).set_position('center')])
        if result.duration is None:
            result.duration = base_video.duration + wipe_vid.duration - transition_duration
        return result

    def wipe_top_to_bottom(self, base_video, next_video, transition_duration):

        wipe_vid = add_wipe(next_video, transition_duration, "top")
        result = CompositeVideoClip([base_video, wipe_vid.set_start(base_video.duration - transition_duration).set_position('center')])
        if result.duration is None:
            result.duration = base_video.duration + wipe_vid.duration - transition_duration
        return result

    def wipe_bottom_to_top(self, base_video, next_video, transition_duration):

        wipe_vid = add_wipe(next_video, transition_duration, "bottom")
        result = CompositeVideoClip([base_video, wipe_vid.set_start(base_video.duration - transition_duration).set_position('center')])
        if result.duration is None:
            result.duration = base_video.duration + wipe_vid.duration - transition_duration
        return result

    def choose_random_videos(self, number_of_videos):
        self.load_database()

        command = "SELECT id FROM video_files"
        self.cursor.execute(command)
        found_ids = self.cursor.fetchall()

        found_ids_list = [row[0] for row in found_ids]

        if number_of_videos > len(found_ids_list):
            raise ValueError("Requested more random videos than available!")

        video_ids = random.sample(found_ids_list, number_of_videos)
        self.close_db()
        return video_ids

    def apply_transition(self, transition_id, video1, video2, transition_duration=1, fade_color=None):
        video1 = self.video_resize(video1)
        video2 = self.video_resize(video2)

        max_width = max(video1.size[0], video2.size[0])
        max_height = max(video1.size[1], video2.size[1])
        canvas = ColorClip((max_width, max_height), col=(0, 0, 0))

        if transition_id == 1:
            result = self.cut(video1, video2)
        elif transition_id == 2:
            result = self.dissolve(video1, video2, transition_duration)
        elif transition_id == 3:
            result = self.wipe_left(video1, video2, transition_duration)
        elif transition_id == 4:
            result = self.wipe_right(video1, video2, transition_duration)
        elif transition_id == 5:
            result = self.wipe_top_to_bottom(video1, video2, transition_duration)
        elif transition_id == 6:
            result = self.wipe_bottom_to_top(video1, video2, transition_duration)
        elif transition_id == 7:
            if fade_color is not None:
                fadein_video = video2.fx(vfx.fadein, duration=transition_duration, initial_color=fade_color)
            else:
                fadein_video = video2.fx(vfx.fadein, duration=transition_duration)
            result = concatenate_videoclips([video1, fadein_video],method='chain')
        else:
            if fade_color is not None:
                fadeout_video = video1.fx(vfx.fadeout, duration=transition_duration, final_color=fade_color)
            else:
                fadeout_video = video1.fx(vfx.fadeout, duration=transition_duration)
            result = concatenate_videoclips([fadeout_video, video2], method='chain')

        canvas.duration = result.duration

        result_canvas = CompositeVideoClip([canvas, result.set_position('center')])
        result_canvas.duration = result.duration
        if result_canvas == None:
            print("Canvas generation failed")
        if result_canvas.duration is None:
            print(f"Canvas duration is None for transition id: {transition_id}")
        return result_canvas



    def add_transitions(self, number_of_videos):
        self.load_database()
        if number_of_videos < 2:
            raise ValueError("The minimum of videos required for this method is 2!")

        command = "SELECT COUNT(*) FROM video_files"
        self.cursor.execute(command)
        videos_in_db = self.cursor.fetchone()[0]
        if videos_in_db < number_of_videos:
            self.close_db()
            raise ValueError("There are not that many unique videos in database!")

        date = datetime.datetime.now()
        date_name = date.strftime("%Y-%m-%d_%H-%M-%S")
        date_name = date_name.replace(" ", "")

        new_path = str(self.main_dir / 'GeneratedVideos' / f"{date_name}.mp4").replace("/","\\")
        new_path2 = str(self.main_dir / 'GeneratedVideos' / f"{date_name}_v2.mp4").replace("/","\\")
        sources_path = str(self.main_dir / 'GeneratedVideos' / 'SectionSources' / f"{date_name}_sections.xml").replace("/","\\")
        self.close_db()
        video_ids = self.choose_random_videos(number_of_videos)
        self.load_database()
        video_paths = []
        transition_videos = []

        transitions_info = []
        used_videos_info = []

        annotation_handler = AnnotationSaver.AnnotationHandler()

        for i in video_ids:
            command = "SELECT path FROM video_files WHERE id = %s"
            self.cursor.execute(command, (i,))
            video_path = self.cursor.fetchone()[0]
            video_paths.append(video_path)

        videos_to_close = []

        anot_start = 5

        for i in range(0, len(video_paths) - 1, 2):
            if i != len(video_paths) - 1:
                video1_orig = VideoFileClip(video_paths[i])
                vid1_length = video1_orig.duration
                begin_vid1_time = random.uniform(0, vid1_length - 5)
                video1 = video1_orig.subclip(begin_vid1_time, begin_vid1_time + 5)
                video2_orig = VideoFileClip(video_paths[i+1])
                vid2_length = video2_orig.duration
                begin_vid2_time = random.uniform(0, vid2_length - 5)
                video2 = video2_orig.subclip(begin_vid2_time, begin_vid2_time + 5)

                transition_id = random.randint(1, 8)
                used_transition = Transition(transition_id).name

                resized_videos = self.resize_videos([video1, video2])
                video1 = resized_videos[0]
                video2 = resized_videos[1]

                created_vid = self.apply_transition(transition_id, video1, video2)

                if created_vid is None or created_vid.duration is None:
                    print(f"After transition, created_vid is None or its duration is None")

                transition_videos.append(created_vid)
                
                hours = int((anot_start - 5) // 3600)
                minutes = int(((anot_start - 5) % 3600) // 60)
                new_seconds = 0 if anot_start == 0 else int((anot_start - 5) % 60)
                
                transitions_info.append({
                    "hours": int(anot_start // 3600),
                    "minutes": int((anot_start % 3600) // 60),
                    "seconds": int(anot_start % 60),
                    "milliseconds": int((anot_start % 1) * 1000),
                    "transition_name": used_transition
                })

                used_videos_info.append({
                    "start": annotation_handler.format_time(hours, minutes, new_seconds),
                    "path": video_paths[i]
                })

                anot_start += 5

                hours = int((anot_start - 5) // 3600)
                minutes = int(((anot_start - 5) % 3600) // 60)
                new_seconds = 0 if anot_start == 0 else int((anot_start - 5) % 60)
                used_videos_info.append({
                    "start": annotation_handler.format_time(hours, minutes, new_seconds),
                    "path": video_paths[i+1]
                })

                if (i + 2 < (len(video_paths) -1) and len(video_paths) % 2 != 0) or (i + 2 < len(video_paths) and len(video_paths) % 2 == 0):
                    transitions_info.append({
                        "hours": int(anot_start // 3600),
                        "minutes": int((anot_start % 3600) // 60),
                        "seconds": int(anot_start % 60),
                        "milliseconds": int((anot_start % 1) * 1000),
                        "transition_name": Transition(1).name
                    })

                    anot_start += 5
                videos_to_close.append(video1)
                videos_to_close.append(video1_orig)
                videos_to_close.append(video2)
                videos_to_close.append(video2_orig)

        transition_videos_formated = []
        if len(transition_videos) > 1:
            transition_videos_formated = self.resize_videos(transition_videos)
            new_vid_ver1 = concatenate_videoclips(transition_videos_formated, method='chain')
        else:
            new_vid_ver1 = transition_videos[0]
        new_vid_ver1.write_videofile(new_path)
        new_vid_ver1.close()

        if len(video_paths) % 2 != 0:
            new_vid_ver2 = VideoFileClip(new_path)
            transition_id = random.randint(1, 8)
            video1_orig = VideoFileClip(video_paths[len(video_paths) - 1])
            vid1_length = video1_orig.duration
            begin_vid1_time = random.uniform(0, vid1_length - 5)
            video1 = video1_orig.subclip(begin_vid1_time, begin_vid1_time + 5)

            resized_videos = self.resize_videos([new_vid_ver2, video1])
            new_vid_ver2 = resized_videos[0]
            video1 = resized_videos[1]

            created_vid = self.apply_transition(transition_id, new_vid_ver2, video1)

            used_transition = Transition(transition_id).name

            hours = int(anot_start // 3600)
            minutes = int((anot_start % 3600) // 60)
            new_seconds = 0 if anot_start == 0 else int(anot_start % 60)

            transitions_info.append({
                "hours": int(int(anot_start) // 3600),
                "minutes": int((int(anot_start) % 3600) // 60),
                "seconds": int(int(anot_start) % 60),
                "milliseconds": int((anot_start % 1) * 1000),
                "transition_name": used_transition
            })

            used_videos_info.append({
                "start": annotation_handler.format_time(hours, minutes, new_seconds),
                "path": video_paths[len(video_paths) - 1]
            })

            created_vid.write_videofile(new_path2)

            videos_to_close.append(video1)
            videos_to_close.append(video1_orig)

            videos_to_close.append(created_vid)

            command = "INSERT INTO transition_videos (path, video_name) VALUES (%s, %s)"
            if len(video_paths) % 2 == 0:
                self.cursor.execute(command, (new_path, new_path))
            else:
                self.cursor.execute(command, (new_path2, new_path2))
            self.db.commit()
            videos_to_close.append(new_vid_ver2)
        else:

            command = "INSERT INTO transition_videos (path, video_name) VALUES (%s, %s)"
            if len(video_paths) % 2 == 0:
                self.cursor.execute(command, (new_path, new_path))
            else:
                self.cursor.execute(command, (new_path2, new_path2))
            self.db.commit()

        if len(video_paths) % 2 == 0:
            video_name = new_path
        else:
            video_name = new_path2

        self.save_element_paths(sources_path, used_videos_info)
        command = "UPDATE transition_videos SET element_paths = (%s) WHERE path = (%s)"
        if len(video_paths) % 2 == 0:
            self.cursor.execute(command, (sources_path, new_path))
        else:
            self.cursor.execute(command, (sources_path, new_path2))
        self.db.commit()
        for item in videos_to_close:
            item.close()

        self.close_db()
        for transition in transitions_info:
            annotation_handler.add_anotation(
                hours=transition['hours'],
                minutes=transition['minutes'],
                seconds=transition['seconds'],
                milliseconds=transition['milliseconds'],
                annotation_type=transition['transition_name'],
                video_name=video_name
            )


    def add_transitions_to_one(self, video_path, part_size=5):
        if part_size == 0:
            print("Part size can't be set to 0")
            return
        self.load_database()
        annotation_handler = AnnotationSaver.AnnotationHandler()
        date = datetime.datetime.now()
        date_name = date.strftime("%Y-%m-%d_%H-%M-%S")
        date_name = date_name.replace(" ", "")
        new_path = str(self.main_dir / 'GeneratedVideos' / f"{date_name}.mp4").replace("/","\\")

        found_video = VideoFileClip(video_path)
        if found_video is None:
            raise Exception("Problem loading video")

        if found_video.duration <= part_size:
            raise Exception(f"Video needs to be longer than {part_size} seconds to apply transitions")
        found_video = self.video_resize(found_video)
        number_of_transitions = math.floor(found_video.duration / part_size)
        clips = []
        videos_to_close = []
        transitions_info = []

        for i in range(number_of_transitions):
            video_start = part_size * i
            video_end = video_start + part_size

            transition_id = random.randint(1, 8)
            transition = Transition(transition_id)

            if i == 0:
                video1 = found_video.subclip(video_start, video_end)
                videos_to_close.append(video1)
            else:
                video1 = clips[i - 1]
                if transition_id == 7 or transition_id == 8:
                    video1 = video1.set_end(video1.duration)

            video2 = found_video.subclip(video_end,
                                         video_end + part_size if video_end + part_size < found_video.duration else found_video.duration)
            if transition_id == 7 or transition_id == 8:
                video2 = video2.set_start(video1.duration - 1)

            if transition_id == 7:
                fadein_video = video2.fx(vfx.fadein, duration=1)
                video1_with_transition = concatenate_videoclips([video1, fadein_video], method='compose')
            elif transition_id == 8:
                fadeout_video = video1.fx(vfx.fadeout, duration=1)
                video1_with_transition = concatenate_videoclips([fadeout_video, video2], method='compose')
            else:
                video1_with_transition = self.apply_transition(transition_id, video1, video2)
            videos_to_close.append(video2)

            clips.append(video1_with_transition)

            found_time = self.transform_seconds_to_hours_minutes_seconds(video_start + part_size)
            transitions_info.append({
                "hours": found_time[0],
                "minutes": found_time[1],
                "seconds": found_time[2],
                "milliseconds": found_time[3],
                "transition_name": transition.name
            })

        final_video = clips[len(clips) - 1]
        final_video.write_videofile(new_path)

        command = "INSERT INTO transition_videos (path, video_name) VALUES (%s, %s)"
        self.cursor.execute(command, (new_path, new_path))
        self.db.commit()
        self.close_db()
        for transition in transitions_info:
            annotation_handler.add_anotation(
                hours=transition['hours'],
                minutes=transition['minutes'],
                seconds=transition['seconds'],
                milliseconds=transition['milliseconds'],
                annotation_type=transition['transition_name'],
                video_name=new_path
            )

        found_video.close()
        final_video.close()
        for video in videos_to_close:
            video.close()
        for clip in clips:
            clip.close()

        return

    def add_transition_by_choice(self, video_path, transition, hour, minute, second, millisecond, second_half_tag="", transition_duration=1, fade_color=None):
        self.load_database()
        date = datetime.datetime.now()
        date_name = date.strftime("%Y-%m-%d_%H-%M-%S")
        date_name = date_name.replace(" ", "")
        new_path = str(self.main_dir / 'GeneratedVideos' / f"{date_name}.mp4").replace("/","\\")

        if video_path is None:
            self.close_db()
            raise Exception("No transition matching required path: " + video_path)

        video = VideoFileClip(video_path)
        second_of_the_clip = second + minute*60 + hour*60*60
        if second_of_the_clip > video.duration:
            self.close_db()
            raise ValueError("Selected time is bigger than video duration!")
        video1 = video.subclip(0, second_of_the_clip)
        video2 = video.subclip(second_of_the_clip)
        transition_id = transition.value

        new_vid = self.apply_transition(transition_id, video1, video2, transition_duration, fade_color)
        new_vid.write_videofile(new_path)
        new_vid.close()
        command = "INSERT INTO transition_videos (path, video_name) VALUES (%s, %s)"

        self.cursor.execute(command, (new_path, new_path))
        self.db.commit()

        video1.close()
        video2.close()
        video.close()
        self.close_db()
        annotation_handler = AnnotationSaver.AnnotationHandler()

        annotation_handler.add_annotation_from_existing(hour, minute, second, millisecond, transition.name, video_path, new_path, second_half_tag)

        return

    def transform_seconds_to_hours_minutes_seconds(self, seconds):
        hours = int(seconds) // 3600
        minutes = (int(seconds) % 3600) // 60
        new_seconds = int(seconds) % 60
        milliseconds = int((seconds % 1) * 1000)
        separated_time = [hours, minutes, new_seconds, milliseconds]
        return separated_time

    def save_element_paths(self, path, elements):
        root = ET.Element('sections_data')
        for element in elements:
            elem_record = ET.SubElement(root, 'section')
            section_start = ET.SubElement(elem_record, 'start')
            section_start.text = str(element['start'])
            section_path = ET.SubElement(elem_record, 'path')
            section_path.text = str(element['path'])

        prettified_xml = XmlHandler.prettify(root)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(prettified_xml)

    def resize_videos(self, videos):
        lower_res_vids = []
        for video in videos:
            new_vid = self.video_resize(video)
            lower_res_vids.append(new_vid)
        videos = lower_res_vids

        max_width = max(video.size[0] for video in videos)
        max_height = max(video.size[1] for video in videos)

        resized_videos = []

        for video in videos:
            original_width, original_height = video.size
            ratio = original_width/original_height
            if max_width / original_width < max_height / original_height:
                new_width = max_width
                new_height = int(max_width / ratio)
            else:
                new_height = max_height
                new_width = int(max_height * ratio)
            new_video = video.resize((new_width, new_height))

            canvas = ColorClip(size=(max_width, max_height), color=(0, 0, 0), duration=video.duration)
            final_video = CompositeVideoClip([canvas, new_video.set_position("center")])
            if final_video.duration is None:
                final_video.duration = video.duration
            resized_videos.append(final_video)
        return resized_videos

    def video_resize(self, video, new_resolution=360):
        width, height = video.size
        aspect_ratio = width / height

        new_height = new_resolution
        new_width = int(new_height * aspect_ratio)

        new_video = video.resize((new_width, new_height))
        return new_video

    def load_database(self):
        self.db = mysql.connector.connect(
            host=config.HOST,
            user=config.USER,
            password=config.PASSWORD,
            database=config.DATABASE
        )
        self.cursor = self.db.cursor()

    def close_db(self):
        self.cursor.close()
        self.db.close()

# Example of possible tests:
# vidGen = VideoGenerator()
# vidGen.add_transitions_to_one("test_path.mp4")
# vidGen.add_transitions_to_one("test_path.mp4")
# vidGen.add_transitions_to_one("test_path.mp4")
# vidGen.add_transitions_to_one("test_path.mp4")
# vidGen.add_transition_by_choice("test_path.mp4", Transition.FADE_IN,0,0,6, 0,"TestTag", fade_color=[173,216,230])
# vidGen.add_transition_by_choice("test_path.mp4", Transition.CUT,0,0,6, 0,"TestTag", fade_color=[173,216,230])
# vidGen.add_transition_by_choice("test_path.mp4", Transition.FADE_IN,0,0,3, "Test")
# vidGen.add_transitions(4)
#
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.add_video_to_database("test_path.mp4")
# vidGen.close_db()
