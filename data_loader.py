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

class DataLoader:
    def __init__(self, json_file):
        self.json_file = json_file

    def load_persons_from_json(self):
        with open(self.json_file, "r") as f:
                data = json.load(f)
        
        persons_dict = defaultdict(lambda: Person("", ""))  # Defaultdict to store persons
        
        for entry in data:
            path = entry["path"]
            caption = entry["caption"]
            text = entry["text"]
            
            # Extract person name from path
            parts = path.split("/")
            if len(parts) > 2:
                person_name = parts[-3]  # The name is infront images or title_images
            else:
                continue  
            
            # Get or create person object
            if person_name not in persons_dict:
                persons_dict[person_name] = Person(person_name, text)
            
            persons_dict[person_name].add_caption(caption)
            persons_dict[person_name].add_paths(path)
        
        return list(persons_dict.values())

            