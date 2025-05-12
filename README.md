# LLM Annotation

This project uses Large Language Models (LLMs) to automatically annotate images of people from Wikipedia by estimating their age.

## How to setup the LLM annotation?
The simplest way to get started is by using the provided preconfigured Conda environment.

### On Linux
1. Clone this repository.



## 📂 Project Structure

.
├── config/
│ ├── annotation_config.yaml # Main config for dataset path, output, and model settings
│ └── prompts.yaml # Prompt templates for the LLM
├── data_json/ # Intermediate JSON files containing parsed person data
├── datasets/ # Input datasets with images and associated text
├── output/ # Final annotated JSON outputs by model and prompt
├── parser.py # Parser to extract person data from input directories
├── data_loader.py # Loader to read and structure parsed person data
├── utils.py # Utility functions (e.g. year extraction)
├── ollama_annotate.py # Main script to annotate using LLMs
├── test_args.py # Test script for development/debugging
└── README.md

Always show details


## ⚙️ Configuration

### 1. `config/annotation_config.yaml`
```yaml
dataset_path: ./datasets/minisubset04_annotated
output_path: ./output
models: ["llama", "deepseek"]
model_output_names: ["LLaMA", "DeepSeek"]
```
2. config/prompts.yaml

Contains prompt templates used for each annotation step.

Always show details

prompt_templates:
  - |
    Given the caption:
    "{caption}"
    and biography:
    "{person_text}"

    Return a JSON with name, birthday, year_of_photo, year_of_photo_int, and can_determine.
  ...

🚀 Running the Annotation

Always show details

python ollama_annotate.py

This script will:

    Parse all person entries in the dataset

    Load the data from JSON

    Iterate over captions and prompts

    Send prompts to the selected LLMs

    Save the structured results as JSON per person, model, and prompt

📊 Stats Logging

    The script tracks:

        Model load time

        Annotation time per person and prompt

        Mean annotation times

Stats are saved as a JSON file in the output directory.
📦 Dependencies

    Python 3.8+

    tqdm

    pyyaml

    Your preferred LLM backend (e.g., Ollama)

Install requirements (if needed):

Always show details

pip install -r requirements.txt

🧠 Example Output

An output JSON looks like:

Always show details

{
  "name": "John Doe",
  "birthday": "1934-05-14",
  "year_of_photo": "1982",
  "year_of_photo_int": 1982,
  "can_determine": true,
  "caption": "John at a political rally in 1982.",
  "path": "datasets/.../image.jpg"
}

📌 Notes

    Prompts are modular and can be swapped or expanded.

    The PersonDescription class defines the expected schema.

    Prompt results are saved per prompt and per model.

📧 Contact

If you have questions, open an issue or contact [Your Name].
"""

readme_path = Path("README.md")
readme_path.write_text(readme_content)
readme_path

Always show details

Result

PosixPath('README.md')
