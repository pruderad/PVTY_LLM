from dateutil import parser
import subprocess
import yaml
from ollama import chat
import json
from pathlib import Path
from datetime import datetime


def extract_year(date_str: str):
    try:
        dt = parser.parse(date_str, fuzzy=True)
        return dt.year
    except ValueError:
        return None  # Return None if parsing fails
    
    
def ensure_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_ollama_running():
    try:
        output = subprocess.check_output(["pgrep", "-f", "~/home/bin/ollama"])
        return True
    except subprocess.CalledProcessError:
        return False
    

def ollama_load_model(model):
    response = chat(
        model=model,
        messages=[{'role': 'user', 'content': ""}],
        options={'temperature': 0},  # Use deterministic output
    )


def save_stats_entry(stats_path: Path, model_name: str, mean_annotation_time: dict, total_annotation_time: float, model_load_time: float, n_annotations: int, total_time: float, GPU_mem: float):
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing stats if they exist
    if stats_path.exists():
        try:
            with stats_path.open("r") as f:
                stats_data = json.load(f)
        except json.JSONDecodeError:
            stats_data = []
    else:
        stats_data = []

    # New stats entry
    entry = {
        "timestamp": datetime.now().replace(microsecond=0).isoformat(sep=","),
        "model": model_name,
        "model_load_time": model_load_time,
        "model_GPU_usage_GB": GPU_mem,
        "mean_annotation_time_sec_per_image": mean_annotation_time,
        "number_of_annotations": n_annotations,
        "total_annotation_time": total_annotation_time,
        "total_time": total_time
    }

    stats_data.append(entry)

    # Write back to file
    with stats_path.open("w") as f:
        json.dump(stats_data, f, indent=4)

    print(f"INFO: Stats saved to {stats_path}")


def save_json_annotation(output_path: Path, person_name: str, model_name: str, prompt_id: int, prompt_results: list[dict]):
    directory = output_path / person_name
    directory.mkdir(parents=True, exist_ok=True)

    filename = directory / f"{person_name}_LLM_data_{model_name}_prompt_{prompt_id}.json"

    with filename.open("w") as json_file:
        json.dump(prompt_results, json_file, indent=4)


def get_gpu_memory_usage():
    """Returns a list of GPU memory usages in MB."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        memory_list = [int(x.strip()) for x in output.strip().split("\n")]
        return memory_list
    except subprocess.CalledProcessError:
        print("WARNING: Failed to get GPU memory usage.")
        return []