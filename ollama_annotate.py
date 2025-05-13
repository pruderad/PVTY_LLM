from pydantic import BaseModel
from ollama import chat
import torch
import subprocess
import time
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import sys

from parser import PersonParser
from data_loader import DataLoader
from data_loader import Person
from utils import *


class PersonDescription(BaseModel):
    name: str
    birthday: str
    year_of_photo: str
    year_of_photo_int: int | None
    can_determine : bool


def annotate(model: str, prompt: str, caption: str, path: str, person: Person, verbose: bool = True):
    """
    Annotates a single image-caption-person instance using an LLM model.

    Parameters:
    - model (str): The name of the LLM model to use.
    - prompt (str): The input prompt to send to the model.
    - caption (str): The caption associated with the image.
    - path (str): The file path to the image.
    - person (Person): A Person object containing metadata such as name and description text.
    - verbose (bool): Whether to print detailed output.

    Returns:
    - dict: A dictionary with structured fields extracted by the model, 
            following the schema defined in the PersonDescription class.
    """
    response = chat(
        model=model,
        format=PersonDescription.model_json_schema(),
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0},  # Use deterministic output
    )

    # Parse model response into a structured dictionary
    image_analysis = PersonDescription.model_validate_json(response.message.content).model_dump()

    # Add extra context to the output
    image_analysis.update({
        'caption': caption,
        'path': path
    })

    if verbose:
        print('-' * 48)
        print(f'Name    : {person.name}')
        print(f'Caption : {caption}')
        print(f'Text    : {person.text}\n')
        print('DATA FROM LLM:')
        print(image_analysis)

        birthday_year = extract_year(image_analysis.get('birthday'))
        photo_year = image_analysis.get('year_of_photo_int')
        if birthday_year is not None and photo_year is not None:
            predicted_age = photo_year - birthday_year
            print(f'Predicted Age: {predicted_age}')
        else:
            print('Predicted Age: None')
        print('-' * 48)
        print()

    return image_analysis


if __name__ == "__main__":
    # ------------------------------
    # Set cuda device if available
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO: Using device: {device}')

    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"INFO: Number of GPUs: {num_gpus}")

    # ------------------------------
    # Load Configs
    # ------------------------------
    config_dir = Path("config")
    annotation_config = load_yaml_config(config_dir / "annotation_config.yaml")
    prompts_config = load_yaml_config(config_dir / "prompts.yaml")

    # ------------------------------
    # Resolve Paths
    # ------------------------------
    dataset_path = Path(annotation_config["dataset_path"])
    output_path = Path(annotation_config["output_path"])
    prompt_templates = ensure_list(prompts_config.get("prompt_templates", []))

    data_json_path = Path("data_json") / f"{dataset_path.name}.json"
    data_json_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    save_stats = annotation_config["save_stats"]

    # ------------------------------
    # Models
    # ------------------------------
    models = ensure_list(annotation_config.get("models")['ollama_model_names'])
    model_output_names = ensure_list(annotation_config.get("models")["output_names"])

    # ------------------------------
    # Parse and Load Data
    # ------------------------------
    print(f'INFO: Parsing data')
    parser = PersonParser(dataset_path)
    parser.parse_all_persons(path=data_json_path, write=True)
    print(f'INFO: Data parsed')

    dataloader = DataLoader(data_json_path)
    persons_list = dataloader.load_persons_from_json()

    # ------------------------------
    # Start Ollama
    # ------------------------------
    if not is_ollama_running():
        ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        print(f"INFO: Ollama running")

    # ------------------------------
    # Timing statistics container
    # ------------------------------
    stats = {
        "model_load_time_sec": 0.0,
        "mean_annotation_time_sec_per_image": {},
        "GPU_memory": {}
    }

    # ------------------------------
    # Annotate
    # ------------------------------
    for model, model_name in zip(models, model_output_names):

        # ------------------------------
        # Model load stats
        # ------------------------------
        start_model_load = time.perf_counter()
        start_total_time = time.perf_counter()
        print(f"INFO: Loading model: {model_name}")
        ollama_load_model(model)
        print(f"INFO: Model loaded: {model_name}")
        end_model_load = time.perf_counter()
        stats["model_load_time_sec"] = end_model_load - start_model_load

        gpu_mem_usage = get_gpu_memory_usage()
        if gpu_mem_usage:
            for idx, mem in enumerate(gpu_mem_usage):
                stats["GPU_memory"][f"GPU_{idx}"] = mem / 1024

        prompt_times = defaultdict(list)
        n_annotations = 0

        for person in tqdm(persons_list, total=len(persons_list), desc="Processing People", file=sys.stdout, dynamic_ncols=True):
            for prompt_id, template_prompt in tqdm(enumerate(prompt_templates), total=len(prompt_templates), desc="Processing Prompts", leave=False, colour="MAGENTA", disable=None):
                prompt_results = []

                for caption, path in zip(person.captions, person.paths):
                    prompt = template_prompt.format(caption=caption, person_text=person.text)
                    n_annotations += 1

                    start_annot = time.perf_counter()

                    result = annotate(model=model, prompt=prompt, caption=caption, path=path, person=person, verbose=False)

                    end_annot = time.perf_counter()
                    prompt_times[prompt_id].append(end_annot - start_annot)

                    prompt_results.append(result)

                save_json_annotation(output_path=output_path, person_name=person.name, model_name=model_name, prompt_id=prompt_id, prompt_results=prompt_results)


        end_total_time = time.perf_counter()
        # ------------------------------
        # Calculate mean annotation time per prompt
        # ------------------------------
        for prompt_id, times in prompt_times.items():
            mean_time = sum(times) / len(times) if times else 0.0
            stats["mean_annotation_time_sec_per_image"][f"prompt_{prompt_id}"] = round(mean_time, 4)

        stats_file = Path("stats/timing_stats.json")

        if save_stats:
            save_stats_entry(
                stats_path=stats_file,
                model_name=model_name,
                mean_annotation_time=stats["mean_annotation_time_sec_per_image"],
                total_annotation_time=end_total_time-end_model_load,
                model_load_time=stats["model_load_time_sec"],
                n_annotations=n_annotations,
                total_time=end_total_time-start_total_time,
                GPU_mem=stats["GPU_memory"] 
            )


    


    