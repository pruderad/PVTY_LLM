import os
import sys
import json
import wikipedia
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.stdout.reconfigure(encoding='utf-8')


class PersonParser:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def check_raster_image(self, image_name: str) -> bool:
        supported_formats = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".ppm"]
        return any(image_name.lower().endswith(ext) for ext in supported_formats)

    def file_in_directory(self, path: str, file: str) -> bool:
        return os.path.isfile(os.path.join(path, file))

    def load_caption_json(self, path_to_person: str, caption_file: str):
        with open(f"{path_to_person}/{caption_file}", encoding="utf8") as f:
            return json.load(f)

    def load_text(self, path_to_person: str) -> str:
        txt_file = f"{path_to_person}/text.txt"
        if self.file_in_directory(path_to_person, 'text.txt'):
            with open(txt_file, "r", encoding="utf-8") as f:
                paragraphs = f.read().strip().split("\n\n")
                if len(paragraphs) >= 3:
                    return "\n\n".join(paragraphs[:2] + [paragraphs[-1]])
                return "\n\n".join(paragraphs)
        return ""

    def get_url_to_page(self, name: str, lang: str = "en") -> str | None:
        wikipedia.set_lang(lang)
        try:
            return wikipedia.page(name, auto_suggest=False, preload=False).url
        except:
            return None

    def create_new_person_json(self, path_to_person, subdirectory, saved_filename, image_description, bbox_info, url_to_wiki_page, text):
        return {
            "path": f"{path_to_person}/{subdirectory}/{saved_filename}",
            "caption": image_description["caption"],
            "text": text,
            "bbox_info": bbox_info,
            "url": url_to_wiki_page
        }

    def load_bbox_desc_file(self, parse_bboxes, path, subdirectory, filename="faces_with_bboxes.csv"):
        if not parse_bboxes:
            return None
        sub_path = f"{path}/{subdirectory}"
        if self.file_in_directory(sub_path, filename):
            try:
                return pd.read_csv(f"{sub_path}/{filename}", encoding="utf8")
            except (pd.errors.EmptyDataError, FileNotFoundError):
                return None
        return None

    def select_relevant_bboxes(self, parse_bboxes: bool, bbox_df: pd.DataFrame, filename: str):
        if bbox_df is None or not parse_bboxes:
            return None
        relevant_bboxes = bbox_df[bbox_df['img_path'] == filename]
        relevant_bboxes_selected = relevant_bboxes.iloc[:, 1:9]
        return relevant_bboxes_selected.values.tolist()

    def show_person(self, image_path: str, caption: str, bboxes: list[list[int]]):
        _, ax = plt.subplots(figsize=(6, 6))
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(caption, fontsize=12, pad=10)

        if bboxes is not None:
            for bbox in bboxes:
                top_left_col, top_left_row, top_right_col, top_right_row, \
                bot_right_col, bot_right_row, bot_left_col, bot_left_row = bbox

                width = top_right_col - top_left_col
                height = bot_left_row - top_left_row

                rect = patches.Rectangle((top_left_col, top_left_row), width, height,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
        plt.show()

    def mine_data_for_person(self, person_name: str,
                             captions: list[str] = ["captions.json", "infobox_captions.json"],
                             parse_bboxes: bool = True):
        path_to_person = f"{self.base_path}/{person_name}"
        updated_json = []
        for caption_file in captions:
            subdirectory = "images" if caption_file == "captions.json" else "title_images"
            if not self.file_in_directory(path_to_person, caption_file):
                continue

            person_json = self.load_caption_json(path_to_person, caption_file)
            bbox_csv_file = self.load_bbox_desc_file(parse_bboxes, path_to_person, subdirectory)
            page_url = self.get_url_to_page(person_name)
            text = self.load_text(path_to_person)

            for image_description in person_json:
                saved_filename = image_description['saved_filename'].split("/")[-1]
                if not self.check_raster_image(saved_filename):
                    continue
                bbox_info = self.select_relevant_bboxes(parse_bboxes, bbox_csv_file, saved_filename)
                cur_json = self.create_new_person_json(
                    path_to_person, subdirectory, saved_filename, image_description, bbox_info, page_url, text)
                updated_json.append(cur_json)
        return updated_json

    def parse_all_persons(self, path: str, show_persons: bool = False, write: bool = False):
        if os.path.isfile(path):
            print("INFO-PARSER: Json file already exists.")
            return
        
        directory = os.listdir(self.base_path)
        directory.sort()
        all_jsons = []
        for person in directory:
            print(person)
            cur_json = self.mine_data_for_person(person)
            if show_persons:
                for item in cur_json:
                    self.show_person(item["path"], item["caption"], item["bbox_info"])
            all_jsons.extend(cur_json)
            print(person, len(cur_json))

        if write:
            with open(path, 'w', encoding="utf8") as f:
                json.dump(all_jsons, f, indent=4)
