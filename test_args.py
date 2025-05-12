import yaml
from pathlib import Path

from parser import PersonParser
from data_loader import DataLoader
from data_loader import Person
from utils import *


# Load YAML
with open("config/annotation_config.yaml") as f:
    annotation_config = yaml.safe_load(f)

with open("config/prompts.yaml") as f:
    prompts_config = yaml.safe_load(f)

prompt_templates = prompts_config["prompt_templates"]
dataset_path = Path(annotation_config["dataset_path"])
output_path = Path(annotation_config["output_path"])
#models = ensure_list(annotation_config.get("models"))
#model_output_names = annotation_config["model_output_names"]

data_json_path = Path("./data_json/") / f"{dataset_path.name}.json"

parser = PersonParser(dataset_path)
parser.parse_all_persons(path=data_json_path, write=True)

# Data loader #
dataloader = DataLoader(data_json_path)
persons_list = dataloader.load_persons_from_json()

models = ensure_list(annotation_config.get("models")['ollama_model_names'])
model_output_names = ensure_list(annotation_config.get("models")["output_names"])

for model, model_name in zip(models, model_output_names):
    print(model)
    print(model_name)


'''
for person in persons_list:
    for caption, path in zip(person.captions, person.paths):
        print(prompt_templates)
        filled_prompts = [template.format(caption=caption, person_text=person.text) for template in prompt_templates]

        for promt in filled_prompts:
            print(person.name)
            print(caption)
            print(path)
            print(promt)
'''