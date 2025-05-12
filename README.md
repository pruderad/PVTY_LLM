# LLM Annotation

This project uses Large Language Models (LLMs) to automatically annotate images of people from Wikipedia by estimating their age.

## How to setup the LLM annotation?
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

### On RCI server
1-4 Same steps as On Linux (locally)

📂 ⚙️ 🚀 📊 📦 🧠 📌 📧 
 

