#!/bin/bash
set -e

ssh-keyscan github.com >> ~/.ssh/known_hosts

# 3) Miniconda
cd ~
wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/miniconda/bin:$PATH"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# 4) Conda ortamı
conda init bash
conda create -y -n slm-jb python=3.11

# 5) Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# 6) Modelleri indir
ollama pull gemma3:1b || true
ollama pull moondream || true
ollama pull llama3.2:1b || true
ollama pull phi4-mini || true
ollama pull gemma3 || true
ollama pull mistral || true
ollama pull codellama || true
ollama pull llama2-uncensored || true
ollama pull granite3.3:2b || true
ollama pull deepseek-r1:1.5b || true

# 7) Github repo
cd /workspace
git clone git@github.com:taskncan/slm-jailbreak-detection.git
cd slm-jailbreak-detection

# 8) FOR CUDA USERS:
if [ -f environment-cuda.yml ]; then
    conda env update -n slm-jb -f environment-cuda.yml
fi

# FOR MAC USERS:
# if [ -f environment-mac.yml ]; then
#     conda env update -n slm-jb -f environment-mac.yml
# fi

conda info
conda env list
ollama list
git status

echo "SLM Jailbreak environment setup completed successfully."
echo "!!!! FOR INITIALIZATION: source ~/.bashrc && conda activate slm-jb"