import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify(xml_root):
    string_root = ET.tostring(xml_root, 'utf-8')
    parsed_root = minidom.parseString(string_root)
    return parsed_root.toprettyxml(indent="  ", encoding="utf-8").decode()


def words_xml_to_array(path_to_file):
    tree = ET.parse(path_to_file)
    root = tree.getroot()
    found_words = [word.text.strip() for word in root.findall('word') if word.text]
    return found_words


def scene_words_xml_to_array_with_time(path_to_file):
    tree = ET.parse(path_to_file)
    root = tree.getroot()

    found_scenes = [scene for scene in root.findall('scene')]
    return found_scenes


def load_elements_by_name(path_to_file, elements_name):
    information = ET.iterparse(path_to_file, events=("start", "end"))
    words = []
    for event, elem in information:
        if event == "end" and elem.tag == elements_name:
            words.append(elem.text)
        elem.clear()
    return words


def add_description(path_to_file, scene):
    tree = ET.parse(path_to_file)
    root = tree.getroot()
    new_element = ET.Element('scene')
    scene_id = ET.SubElement(new_element, 'id')
    scene_id.text = scene.id
    sc_time = ET.SubElement(new_element, 'time')
    sc_time.text = scene.time
    sc_desc = ET.SubElement(new_element, 'description')
    sc_desc.text = scene.description
    root.append(new_element)
    prettified_xml = prettify(root)
    with open(path_to_file, 'w', encoding='utf-8') as f:
        f.write(prettified_xml)


def add_all_descriptions(path_to_file, scenes):
    if path_to_file is not None:
        tree = ET.parse(path_to_file)
        root = tree.getroot()
    else:
        root = ET.Element('scene_data')
    for scene in scenes:
        new_element = ET.Element('scene')
        scene_id = ET.SubElement(new_element, 'id')
        scene_id.text = scene.id
        sc_time = ET.SubElement(new_element, 'time')
        sc_time.text = scene.time
        sc_desc = ET.SubElement(new_element, 'description')
        sc_desc.text = scene.description
        root.append(new_element)
    prettified_xml = prettify(root)
    with open(path_to_file, 'w', encoding='utf-8') as f:
        f.write(prettified_xml)


#to transform txt from transnet to xml
def transform_shot_boundary_detection_to_xml(path_to_file, new_path):
    if path_to_file is not None:
        root = ET.Element('shot_boundary_detection')
        with open(path_to_file, 'r', encoding='utf-8') as f:
            for line in f:
                scene = ET.SubElement(root, 'scene')
                start, end = line.split("-")
                start_xml = ET.SubElement(scene, 'start')
                start_xml.text = start
                end_xml = ET.SubElement(scene, 'end')
                end_xml.text = end
        tree = ET.ElementTree(root)
        tree.write(new_path, encoding='utf-8', xml_declaration=True)

