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
OUTPUT_SAVE_PATH = "./outputs/minisubset04"

'''
original_prompt = (
                    "Find the year when was photo taken from this caption:\n " + str(caption) + 
                    ".\nFind when was the person born from this description: \n" + person.text +
                    "\n\nReturn JSON description including name, birthday from text and year when was the photo taken from caption. "
                    "If you can not determine birthday set it as None and if you can not determine year of photo set it to None and in both cases set variable can_determine to FALSE."
                    "Otherwise set can_determine to True. If there is a year in caption take it as a year when was the photo taken and if you can determine the year set it as int to year_of_photo_int. "
                    "If there is some time interval of photo take the closest number."
                  )
'''
                    
class PersonDescription(BaseModel):
    name: str
    birthday: str
    year_of_photo: str
    year_of_photo_int: int | None
    can_determine : bool

def annotate(model: str, prompt: str, caption: str, path: str, verbose: bool = True, save: bool = True, save_path: str = "", prompt_id: int = 0, num_gpus: int = 1):
    response = chat(
        model=model,
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
        if model == 'llama3.3':
            model_name = 'llama3-3'
        elif model == 'deepseek-r1:70b':
            model_name = 'deepseek-r1'

        directory = os.path.join(save_path, person.name)
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{person.name}_LLM_data_{model_name}_prompt_{prompt_id}.json")

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

    models = ["llama3.3", "deepseek-r1:70b"]

    for model in models:

        for i, person in tqdm(enumerate(persons_list), total=len(persons_list), desc="Processing Persons"):
            results = []

            for caption, path in zip(person.captions, person.paths):
                prompts = [
                        (
                            f"Find the year when the photo was taken from this caption:\n"
                            f"{caption}\n\n"
                            f"Find the date of birth of the person from this description:\n"
                            f"{person.text}\n\n"
                            f"Return the result as a JSON object with the following fields:\n"
                            f"- name: Full name of the person\n"
                            f"- birthday: The date of birth (as a string), or None if unknown\n"
                            f"- year_of_photo: Year when the photo was taken (as a string), or None if unknown\n"
                            f"- year_of_photo_int: Integer year (e.g. 1984), or None if unknown\n"
                            f"- can_determine: True if both birthday and year_of_photo_int can be determined, otherwise False\n\n"
                            f"If a year is mentioned in the caption, use that as the photo year.\n"
                            f"If there's a time range (e.g., 'between 1950 and 1960'), choose the closest year.\n"
                        ),

                        (
                            f"Analyze the following:\n\n"
                            f"Photo Caption:\n{caption}\n\n"
                            f"Person Description:\n{person.text}\n\n"
                            f"Step 1: Estimate the year the photo was taken from the caption.\n"
                            f"Step 2: Extract the person's date of birth from the description.\n"
                            f"Step 3: Return the results as a JSON object in the following format:\n\n"
                            f"name, birthday, year_of_photo, year_of_photo_int, can_determine\n\n"
                            f"If either birthday or year of photo is not clear, set the value to None and can_determine to FALSE. "
                            f"If a date range is found, choose the closest specific year.\n"
                        ),

                        (
                            f"Your task is to extract structured information from the following:\n\n"
                            f"- Caption of the photo: {caption}\n"
                            f"- Description of the person (e.g., from a Wikipedia article):\n{person.text}\n\n"
                            f"Return the following fields in JSON format:\n"
                            f"- name: Name of the person\n"
                            f"- birthday: Full date of birth (as text)\n"
                            f"- year_of_photo: Year the photo was taken (as text)\n"
                            f"- year_of_photo_int: Year as an integer (e.g. 1998)\n"
                            f"- can_determine: TRUE if both birthday and photo year can be reasonably determined, otherwise FALSE.\n\n"
                            f"If the birthday or year is missing or unclear, use null and set can_determine to FALSE. "
                            f"If a range is mentioned (e.g. 'between 1950 and 1960'), pick the most likely year.\n"
                        ),

                        (
                            f"From the following caption and biography, determine when the photo was likely taken and when the person was born.\n\n"
                            f"Caption:\n{caption}\n\n"
                            f"Biography:\n{person.text}\n\n"
                            f"Please return your answer in this JSON format:\n"
                            f"{{\n"
                            f'  "name": "Full name",\n'
                            f'  "birthday": "Full date or None",\n'
                            f'  "year_of_photo": "Year string or None",\n'
                            f'  "year_of_photo_int": Integer year or None,\n'
                            f'  "can_determine": true or false\n'
                            f"}}\n\n"
                            f"Set can_determine to FALSE if either date is missing. Use your best judgment if an approximate date is available.\n"
                        ),

                        (
                            f"Given the caption:\n\"{caption}\"\n"
                            f"and this Wikipedia text:\n\"{person.text}\"\n\n"
                            f"Extract:\n"
                            f"1. The person's name\n"
                            f"2. Their birthday\n"
                            f"3. The year the photo was taken\n"
                            f"Return your answer in JSON format as:\n"
                            f"{{ 'name': ..., 'birthday': ..., 'year_of_photo': ..., 'year_of_photo_int': ..., 'can_determine': ... }}\n\n"
                            f"Use None if any field is missing or unknown. If both key dates are available, set can_determine to true."
                        ),

                        (
                            f"You are an expert assistant. Please analyze the following input.\n\n"
                            f"Caption of photo: {caption}\n"
                            f"Person biography: {person.text}\n\n"
                            f"Identify:\n"
                            f"- Name\n"
                            f"- Birthday (if available)\n"
                            f"- Year photo was taken (based on caption)\n\n"
                            f"Return the data in this JSON format:\n"
                            f"{{\n"
                            f"  \"name\": str,\n"
                            f"  \"birthday\": str or null,\n"
                            f"  \"year_of_photo\": str or null,\n"
                            f"  \"year_of_photo_int\": int or null,\n"
                            f"  \"can_determine\": true or false\n"
                            f"}}\n"
                        )
                ]
                
                for k, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Processing Prompts", leave=False, colour='MAGENTA'):
                    annotate(model=model, prompt=prompt, caption=caption, path=path, verbose=False, save=True, save_path=OUTPUT_SAVE_PATH, prompt_id=k, num_gpus=num_gpus)

