from pydantic import BaseModel
from ollama import chat
from dateutil import parser as time_parser
import torch
import subprocess
import time
import os
import json
from tqdm import tqdm

from parser import PersonParser
from data_loader import DataLoader
from data_loader import Person
from utils import *


DATASET_PATH = "./datasets/minisubset04_annotated"
DATA_JSON_PATH = "./data_json/data_minisubset04_03.json"
SAVE_PATH = "./outputs/minisubset04_2"

MODEL = 'llama3.3'


class PersonDescription(BaseModel):
    name: str
    birthday: str
    year_of_photo: str
    year_of_photo_int: int
    can_determine : bool

def annotate(prompt: str, caption: str, path: str, verbose: bool = True, save: bool = True, save_path: str = "", num_gpus: int = 1):
    response = chat(
        model=MODEL,
        format=PersonDescription.model_json_schema(),
        messages=[
            {
            'role': 'user',
            'content': prompt
            },
        ],
        options={'temperature': 0},  # Set temperature to 0 for more deterministic output
    )
    image_analysis = PersonDescription.model_validate_json(response.message.content).model_dump()
    image_analysis.update({'caption': caption})
    image_analysis.update({'path': path})
    results.append(image_analysis)

    if verbose:
        print('------------------------------------------------')
        print(f'name: {person.name}\ncaption: {caption}\ntext: {person.text}')
        print('\nDATA FROM LLM:')
        print(image_analysis)
        if extract_year(image_analysis['birthday']) is not None:
            print(f'predicted age: {image_analysis['year_of_photo_int'] - extract_year(image_analysis['birthday'])}')
        print('------------------------------------------------')
        print()

    if save:
        directory = os.path.join(save_path, person.name)
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{person.name}_LLM_data.json")

        # Save the data as JSON
        with open(filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)



if __name__ == "__main__":
    # Set cuda device if available #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO: Using device: {device}\n')
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"INFO: Number of GPUs: {num_gpus}")

    if not is_ollama_running():
        ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)

    # Parser #
    parser = PersonParser(DATASET_PATH)
    parser.parse_all_persons(path=DATA_JSON_PATH, write=True)

    # Data loader #
    dataloader = DataLoader(DATA_JSON_PATH)
    persons_list = dataloader.load_persons_from_json()

    for i, person in tqdm(enumerate(persons_list), total=len(persons_list), desc="Processing Persons"):
        results = []

        for caption, path in zip(person.captions, person.paths):
            prompt = 'Find the year when was photo taken from this caption:\n ' + str(caption) + '.\nFind when was the person born from this description: \n' + person.text
            prompt += (
                        "\n\nReturn JSON description including name, birthday from text and year when was the photo taken from caption. If you can not determine birthday set it as None and if you can not determine year of photo set it to None and in both cases set variable can_determine to FALSE."
                       "Otherwise set can_determine to True. If there is a year in caption take it as a year when was the photo taken and if you can determine the year set it as int to year_of_photo_int. If there is some time interval of photo take the closest number."
                      )

            annotate(prompt=prompt, caption=caption, path=path, verbose=False, save=True, save_path=SAVE_PATH, num_gpus=num_gpus)

 
