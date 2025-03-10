# RAGSum - Text Summarization with Retrieval-Augmented Generation (RAG)

This is the codebase to my Master's thesis at Maastricht University.

## Python Environment

### Linux

On Linux, we need to make sure we have the proper Python environment, the following shell script gets us the required 3.12.3 Python version:

```sh
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa 
apt update
apt install python3.10
python3.10 --version 
which python3.10 - /usr/bin/python3.10
pip install --upgrade virtualenv
cd persistent
virtualenv -p $(which python3.10) venv
source venv/bin/activate 
```

To install the jupyter kernel and make it visible in VS Code:

```
pip install ipykernel
python -m ipykernel install --user --name venv
```

This way, the virtual environment will be created, and sourced, we are ready to code!

### Windows

First, download the python version you need.

Second, create a virtual environment for development:

```sh
python -m venv venv
venv\Scripts\Activate.bat
pip install -r requirements.txt
```

### Ollama Environment

To shoot up locally running LLMs, you need to install [Ollama](https://ollama.com/download)

#### Linux installation

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

This command will install ollama to run LLMs locally.

In a new terminal, run your desired model, I'll use deepseek-r1:8b for now

To start an ollama server use the following prompt:


```sh
ollama serve
ollama run deepseek-r1:8b
```