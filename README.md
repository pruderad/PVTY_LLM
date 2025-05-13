# LLM Annotation

This project uses Large Language Models (LLMs) to automatically annotate images of people from Wikipedia by estimating their age from text and image captions.

## 🔧 How to setup the LLM annotation?
The simplest way to get started is by using the provided preconfigured Conda environment.

### On Linux (locally)
1. Clone this repository.
2. Download and install Anaconda.
3. Run this command:

       conda env create -f environment.yml
   
4. Activate conda enviroment:

       conda activate ollama_env
   
5. Download and install Ollama:

       curl -fsSL https://ollama.com/install.sh | sh
   
6. You can download small model to test if everything works. For example:

       ollama run llama3.2:1b
   
7. Create your configuration in `config/annotation_config.yaml` and `config/prompts.yaml`. You can use example configuration from `config/example/...`

8. To run annotation:

        python3 ollama_annotate.py

### On RCI server
1-4. Same as in the "On Linux (Locally)" section.

5. To download and install Ollama you need to do manual installation:

       curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
       tar -xzf ollama-linux-amd64.tgz -C ~/

6. Add Ollama to PATH. Open your `.bashrc` file and add the following lines (replace <your_name> with your actual username):

       export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/<your_name>/lib/ollama/"
       export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/<your_name>/lib/ollama/cuda_v12/"
       export PATH="$PATH:/home/<your_name>/bin"

7. Apply changes:

       source ~/.bashrc

8. Fix SSL Certificate issues (Optional):


   If you're running on the `interactive` partition and encounter SSL certificate errors, run:

       export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

9. You can download small model to test if everything works. For example:

       ollama serve # In new terminal
       ollama run llama3.2:1b

11. Create your configuration in `config/annotation_config.yaml` and `config/prompts.yaml`. You can use example configuration from `config/example/...`

12. To run annotation on `interactive` partition:

        python3 ollama_annotate.py

13. To run annotation on noninteractive batch job you can use provided `job.batch` file:

        sbatch job.batch


## ⚙️ Custom configuration of annotation
To configurate annotation edit file `config/annotation_config.yaml`. To create custom prompts edit file `config/prompts.yaml`. You can use example configuration from `config/example/...`.


### `config/annotation_config.yaml`:
```yaml
models:
    ollama_model_names: model or [model1, model2, ...]                  # List (or single string) of model identifiers as recognized by Ollama (see: https://ollama.com/library)
    output_names: model_name or [model_name1, model_name2, ...]         # List (or single string) of custom names of models used in output filenames (<person_name>_LLM_data_<output_name>_prompt_<prompt_id>.json)

dataset_path: ...                 # Path to the dataset directory
output_path: ...                  # Path to the output directory

save_stats: ...                   # True/False to enable saving time statistics (e.g., model load time, annotation time)
```

### `config/prompts.yaml`:
⚠️ **Important Note: To include literal curly braces in the text, use double braces: {{your text}}.**
```yaml
prompt_templates:     # List of prompt templates. Use {caption} and {person_text} as placeholders.
                      # IMPORTANT: To include literal curly braces in the text, use double braces: {{your text}}.
  - |
      prompt 1
    
  - |
      prompt 2
```

## 📊 Model library
Ollama supports a list of models available on [ollama.com/library](https://ollama.com/library).

Some example models that can be downloaded are also listed on [github.com/ollama](https://github.com/ollama/ollama).

## 📄 Output format

```json
[
    {
        "name": str,
        "birthday": str,
        "year_of_photo": str,
        "year_of_photo_int": int,
        "can_determine": bool,
        "caption": str,
        "path": str
    }
]
```

To extract birtday year, you can use this function from `utils.py`:
```python
def extract_year(date_str: str) -> int | None:
    try:
        dt = parser.parse(date_str, fuzzy=True)
        return dt.year
    except ValueError:
        return None  # Return None if parsing fails
```

 
