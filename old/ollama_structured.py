from pydantic import BaseModel
from ollama import chat
import pickle
from dateutil import parser
import os
import subprocess
import time
import torch
import json
from collections import defaultdict

class Person:
    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.paths = []
        self.captions = []
        self.age = None  
        
    def add_caption(self, caption):
        self.captions.append(caption)
    
    def add_paths(self, path):
        self.paths.append(path)

# Load JSON file
def load_persons_from_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    persons_dict = defaultdict(lambda: Person("", ""))  # Defaultdict to store persons
    
    for entry in data:
        path = entry["path"]
        #print(path)
        caption = entry["caption"]
        text = entry["text"]
        
        # Extract person name from path
        parts = path.split("/")
        if len(parts) > 2:
            person_name = parts[-3]  # The name is after "./minisubset03/"
            #print(parts)
        else:
            continue  # Skip if path is malformed
        
        # Get or create person object
        if person_name not in persons_dict:
            persons_dict[person_name] = Person(person_name, text)
        
        persons_dict[person_name].add_caption(caption)
        persons_dict[person_name].add_paths(path)
    
    return list(persons_dict.values())

class PersonDescription(BaseModel):
    name: str
    birthday: str
    year_of_photo: str
    year_of_photo_int: int
    can_determine : bool

def extract_year(date_str):
    try:
        dt = parser.parse(date_str, fuzzy=True)
        return dt.year
    except ValueError:
        return None  # Return None if parsing fails


persons_list = load_persons_from_json("./../data_json/data_minisubset04_02.json") 
print(len(persons_list))


ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)

#model = 'llama3.2:1b'
model = 'llama3.3'

print(torch.cuda.is_available())
print(len(persons_list))
#model = 'deepseek-r1'


i = 0
for person in persons_list:
    results = []
    print(f"{i} out of {len(persons_list)}")
    i += 1
    for caption, path in zip(person.captions, person.paths):
        print('------------------------------------------------')
        print(f'name: {person.name}\ncaption: {caption}\ntext: {person.text}')

        prompt = 'Find the year when was photo taken from this caption:\n ' + str(caption) + '.\nFind when was the person born from this description: \n' + person.text

        response = chat(
            model=model,
            format=PersonDescription.model_json_schema(),
            messages=[
                {
                'role': 'user',
                'content': prompt + '.\n Return JSON description including name, birthday from text and year when was the photo taken from caption. If you can not determine birthday set it as None and if you can not determine year of photo set it to None and in both cases set variable can_determine to FALSE. Otherwise set can_determine to True. If there is a year in caption take it as a year when was the photo take and if you can determine the year set it as int to year_of_photo_int. If there is some time interval of photo take the closest number.'
                },
            ],
            options={'temperature': 0},  # Set temperature to 0 for more deterministic output
        )

        image_analysis = PersonDescription.model_validate_json(response.message.content).dict()
        image_analysis.update({'caption': caption})
        image_analysis.update({'path': path})
        results.append(image_analysis)

        print('\nDATA FROM LLM:')
        print(image_analysis)
        if extract_year(image_analysis['birthday']) is not None:
            print(f'predicted age: {image_analysis['year_of_photo_int'] - extract_year(image_analysis['birthday'])}')
        print('------------------------------------------------')
        print()

    directory = os.path.join("./../outputs/minisubset04", person.name)


    # Define the filename
    filename = os.path.join(directory, f"{person.name}_LLM_data.json")

    # Save the data as JSON
    with open(filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)
