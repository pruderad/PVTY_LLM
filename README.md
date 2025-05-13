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

8. To run annotation:

           python3 ollama_annotate.py

### On RCI server
1-4. Same as in the "On Linux (Locally)" section.

5. To download and install Ollama you need to do manual installation:

        curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
        tar -xzf ollama-linux-amd64.tgz -C ~/

6. Add Ollama to PATH. Open your `.bashrc` file and add the following lines:

           export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/<your_name>/lib/ollama/"
           export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/<your_name>/lib/ollama/cuda_v12/"
           export PATH="$PATH:/home/<your_name>/bin"

7. Apply Changes:

           source ~/.bashrc

8. Fix SSL Certificate Issues (Optional):


   If you're running on the `interactive` partition and encounter SSL certificate errors, run:

           export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

📂 ⚙️ 🚀 📊 📦 🧠 📌 📧 
 
