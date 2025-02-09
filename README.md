# RAGSum - Text Summarization with Retrieval-Augmented Generation (RAG)

This is the codebase to my Master's thesis at Maastricht University.

### Python Environment

#### Linux

On Linux, we need to make sure we have the proper Python environment, the following shell script gets us the required 3.12.3 Python version:

```
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa 
apt update
apt install python3.12
python3.12 --version 
which python3.12 - /usr/bin/python3.12
pip install --upgrade virtualenv
cd persistent
virtualenv -p $(which python3.12) venv
source venv/bin/activate 
```

This way, the virtual environment will be created, and sourced, we are ready to code!

#### Windows

First, download the python version you need.

Second, create a virtual environment for development:

```
python -m venv venv
```

Then, activate the environment with:

```
venv\Scripts\Activate.bat
```

Finally, install the dependencies with:

```
pip install -r requirements.txt
```

### Ollama Environment

To shoot up locally running LLMs, you need to install [Ollama](https://ollama.com/download)

#### Linux installation

```
curl -fsSL https://ollama.com/install.sh | sh
```

This command will install ollama to run LLMs locally.

To start an ollama server use the following prompt:

```
ollama serve
```

In a new terminal, run your desired model, I'll use deepseek-r1:8b for now

```
ollama run deepseek-r1:8b
```