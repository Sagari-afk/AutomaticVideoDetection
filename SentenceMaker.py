# Test with example words
#words = ["person", "dog", "park"]
# words = ["person", "person", "kiss"]
# words = ["child", "ball", "playground"]
# words = ["cat", "tree", "shade"]
# words = ["car", "road", "travel"]
# words = ["car", "person", "person", "car", "bicycle"]
# words = ["cat", "horse"]
# words = ["person", "horse"]
# words = ["cat", "bowl", "milk"]
# words = ["person", "eat", "burger"]
import math
import sys

# result = SentenceMaker().connect_sentence(words)
# print(result)

import torch
import requests
from transformers import T5Tokenizer, T5ForConditionalGeneration,  M2M100ForConditionalGeneration, M2M100Tokenizer
import xml.etree.ElementTree as ET
import XmlHandler
from SceneObjectSumarizer import SceneObjectsSummarizer


class SentenceMaker:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(self.device)

        #translation
        self.model_ts = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(self.device)
        self.tokenizer_ts = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

        if hasattr(torch, 'compile'):
            if sys.platform != "win32" and hasattr(torch, "compile"):
                self.model_ts = torch.compile(self.model_ts)
                self.model = torch.compile(self.model)

        self.src_language = "en"
        self.tgt_language = "sk"
        self.tokenizer_ts.src_language = self.src_language

        self.summarizer = SceneObjectsSummarizer()

    def connect_sentence(self, words):
        print(f"All words: {', '.join(words)}")

        prompt = (f"Given these words: {', '.join(words)}, create a meaningful sentence that describes an action"
                  f" or relationship involving all of them. "
                  f"Each word represents a distinct object, so ensure they are included accurately in your sentence. ")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(inputs.input_ids, max_new_tokens=20, temperature=0.3, do_sample=True, no_repeat_ngram_size=2)
        sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(sentence)

        del inputs
        del outputs
        return sentence

    def check_ollama(self):
        """Kontrola Ollama serveru."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def llama_model(self):
        print("[LLM] Checking Ollama server...")

        if not self.check_ollama():
            print("\n" + "=" * 60)
            print("X CHYBA: Ollama nie je spustená!")
            print("=" * 60)
            print("\nSpusti Ollama server v termináli:")
            print("  → ollama serve")
            print("\nPotom skontroluj, či je nainštalovaný model:")
            print("  → ollama list")
            print("\nAk model nie je nainštalovaný, stiahni ho")
            print("=" * 60 + "\n")
            raise ConnectionError("Ollama server is not running.")

        print(f"[LLM] Ollama server is running.")

    def generate_scene_description(self, scene_objects):
        summary_text = self.summarizer.summarize_scene_objects(scene_objects)
        print(summary_text)

        # prompt = f'''
        #     You are a video scene description model.
        #     Based on the following scene summary, write a natural and meaningful English description of the whole scene.
        #     Include the main action and relevant secondary objects if needed.
        #     {scene_objects}
        # '''
        #
        # inputs = self.tokenizer(
        #     prompt,
        #     return_tensors="pt",
        #     truncation=True,
        #     max_length=512
        # )
        #
        # torch.cuda.empty_cache()
        #
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #
        # outputs = self.model.generate(
        #     input_ids=inputs["input_ids"],
        #     attention_mask=inputs["attention_mask"],
        #     max_new_tokens=20,
        #     temperature=0.3,
        #     do_sample=True
        # )
        # sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #
        # print(sentence)

        if not scene_objects:
            return "No objects detected in the image."

        prompt = f"""
            You are an action-oriented video captioning assistant.

Convert the technical scene summary into a natural, coherent description of the video.

The input may contain separate tracks for different detected object categories. Some of these separate tracks may actually describe parts of the same real-world activity. Your task is to infer the most likely scene-level events from the tracks.

Internal interpretation steps:
1. Group related tracks when they overlap for many frames, move in similar directions, appear at similar times, or have a plausible real-world relationship.
2. Treat long-lasting overlap as a strong visual association cue, even if physical contact is not fully confirmed.
3. Decide whether the scene is better described as several separate objects or as one or more participants/groups performing an activity.
4. Convert technical movement into natural actions.
5. Mention uncertainty only when necessary, using “appears to,” “seems to,” “likely,” or “may.”

Important constraints:
- Do not list tracks one by one.
- Do not mention track IDs, frame numbers, detection counts, confidence, thresholds, or bounding boxes.
- Do not over-focus on screen coordinates.
- Do not claim definite physical contact unless the data clearly supports it.
- Do not say an object is physically still only because it stays in the same part of the frame; the camera may be moving.
- Do not describe only motion. Describe likely actions and relationships.
- Do not invent unsupported dramatic events.
- If objects are visually associated for much of the video, describe a plausible interaction or shared activity in a cautious but meaningful way.
- If several objects appear to move in the same direction or form a group, describe them as moving together, following one another, or progressing through the scene.

Use natural verbs such as:
moves, travels, proceeds, follows, leads, approaches, passes, continues, recedes, advances, stays near, remains alongside, appears connected to, moves together with, crosses, enters, leaves, accompanies, interacts.

Write one polished paragraph of 3–5 sentences.
Output only the final description.

Scene summary:
{summary_text}
            """

        payload = {
            "model": "mistral-small:24b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.8,
                "num_ctx": 4096
            }
        }

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=500)
            response.raise_for_status()
            result = response.json()
            generated = result.get("response", "").strip()

            # čistenie výstupu
            import re
            generated = re.sub(r'[<>`´=]+', '', generated)
            generated = generated.replace('\n', ' ').strip()

            if not generated:
                return "Failed to generate caption."

            # kapitalizácia a bodka
            if generated and not generated[0].isupper():
                generated = generated[0].upper() + generated[1:]
            if generated and not generated.endswith('.'):
                generated += '.'

            print(generated)

            return generated

        except Exception as e:
            print(f"[LLM ERROR] Ollama generation failed: {e}")

            return "Failed to generate caption."
        #
        # del inputs
        # del outputs
        # return sentence


    def connect_desc(self, sentences):
        prompt = (f"Write short summary consisting of exactly 3 to 5 sentences for this text: { '.'.join(sentences) }")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(inputs.input_ids, max_new_tokens=100, temperature=0.6, top_p=0.9, do_sample=True,
                                      num_beams=5, repetition_penalty=1.2)
        sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        del inputs
        del outputs
        return sentence

    def translate_sentence(self, sentenceS):
        inputs = self.tokenizer_ts(sentenceS, return_tensors="pt").to(self.device)
        gen_tokens = self.model_ts.generate(**inputs, forced_bos_token_id=self.tokenizer_ts.get_lang_id(self.tgt_language),
                                            num_beams=5)
        sentence = self.tokenizer_ts.decode(gen_tokens[0], skip_special_tokens=True)

        del inputs
        del gen_tokens
        return sentence

    def create_sentence_xml_file(self, old_file_path):
        desc_file_path = old_file_path.replace(".xml", "_desc.xml")
        new_root = ET.Element("video_objects")
        scenes = XmlHandler.scene_words_xml_to_array_with_time(old_file_path)
        for scene in scenes:
            time = scene.attributes["time"]
            new_scene = ET.SubElement(new_root, "scene")
            new_scene.attrib["time"] = time
            words = [word.text.strip() for word in scene.findall('word') if word.text]
            yield words
            sentence = self.connect_sentence(words)
            sentence2 = self.connect_sentence(words)
            sentence3 = self.connect_sentence(words)
            stc_el = ET.SubElement(new_scene, "sentence")
            stc_el.text = sentence
            stc_el2 = ET.SubElement(new_scene, "sentence")
            stc_el2.text = sentence2
            stc_el3 = ET.SubElement(new_scene, "sentence")
            stc_el3.text = sentence3
        prettified_root = XmlHandler.prettify(new_root)
        with open(desc_file_path, 'w', encoding='utf-8') as f:
            f.write(prettified_root)


    def find_sentence(self, tr_time, path):
       desc_file_path = path + "_desc.txt"
       with open(desc_file_path) as file:
           for line in file:
                line = line.strip()
                start, sentence = line.split("|", 1)
                if tr_time == start:
                    return sentence

    def test_generate_n_translate(self):
        words = ["child", "ball", "playground"]
        result = self.connect_sentence(words)
        print("Generovanie1: " + result)
        translation = self.translate_sentence(result)
        print("Preklad: " + translation)
        words2 = ["cat", "shade", "tree"]
        result2 = self.connect_sentence(words2)
        print("Generovanie2: " + result2)
        translation2 = self.translate_sentence(result2)
        print("Preklad2: " + translation2)
        words3 = ["person", "person", "person", "waiting in line", "person", "truck"]
        result3 = self.connect_sentence(words3)
        print("Generovanie3: " + result3)
        translation3 = self.translate_sentence(result3)
        print("Preklad3: " + translation3)
        words4 = ["person", "person", "car", "jaywalking", "person", "cat"]
        result4 = self.connect_sentence(words4)
        print("Generovanie4: " + result4)
        translation4 = self.translate_sentence(result4)
        print("Preklad4: " + translation4)

    def test_generate_desc(self):
        # sentences = ["Child running with a ball.", "Dog is running.",
        #              "A woman is sitting."]
        sentences = ["A group of people is jaywalking", "Bus on the street",
                     "Person is driving a bicycle."]
        # sentences = ["child", "ball", "running", "dog", "person", "sitting"]
        result = self.connect_desc(sentences)
        print(result)

    def delete_models(self):
        print("----------------Delete models CALLED-----------------")
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'model_ts') and self.model_ts:
            del self.model_ts
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        if hasattr(self, 'tokenizer_ts') and self.tokenizer_ts:
            del self.tokenizer_ts
        torch.cuda.empty_cache()

    def __del__(self):
        self.delete_models()

# sm = SentenceMaker()
# sm.test_generate_n_translate()
# sm.test_generate_desc()
# sm.delete_models()
# SentenceMaker().test_generate_desc()
