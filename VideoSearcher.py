
from lxml import etree

from enum import Enum


class Language(Enum):
    EN = "EN"
    SK = "SK"


class VideoSearcher:

    def search_descriptions(self, desc_path, search_text):
        if isinstance(search_text, str):
            search_text = search_text.split(",")

        search_text_lower = {word.lower().strip() for word in search_text}
        search_text = set(search_text_lower)  # to ensure unique results

        search_result = []

        contents = etree.iterparse(desc_path, events=("end",), tag="scene")

        for event, scene in contents:
            scene_id = int(scene.find('id').text)
            objects = scene.findall('object')
            occurrence_count = 0
            for obj in objects:
                if search_text and obj.text.lower() in search_text:
                    occurrence_count += 1

            descriptions = scene.findall('description')
            for desc in descriptions:
                for word in search_text:
                    occurrence_count += desc.text.lower().count(word)

            if occurrence_count != 0:
                search_result.append([scene_id, occurrence_count])
            scene.clear()

        return search_result

