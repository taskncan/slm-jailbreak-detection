#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
REPO_SSH="git@github.com:taskncan/slm-jailbreak-detection.git"
REPO_DIR="/workspace/slm-jailbreak-detection"
ENV_NAME="slm-jb"
MINICONDA="$HOME/miniconda"
CUDA_CUPY_WHL="cupy-cuda12x"   # matches CUDA 12.x images
PYTORCH_CUDA="12.4"            # use conda pytorch-cuda=12.4 (NVIDIA channel)

# =========================
# Helpers
# =========================
die() { echo "[ERROR] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

echo "[0/8] Pre-flight checks"
export LC_ALL=C.UTF-8 LANG=C.UTF-8
need_cmd bash
need_cmd ssh
need_cmd git
need_cmd wget
need_cmd curl

if [[ $(id -u) -ne 0 ]]; then
  echo "[i] You're not root. That's fine — this script does not use sudo."
fi

# =========================
# 1) Base apt packages (if apt-get available)
# =========================
if command -v apt-get >/dev/null 2>&1; then
  echo "[1/8] Update apt and install base tools"
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git wget curl tmux build-essential nano vim ca-certificates
fi

# =========================
# 2) SSH setup for GitHub
# =========================
echo "[2/8] Prepare SSH for GitHub"
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Preload GitHub host keys (quietly)
ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true
chmod 644 ~/.ssh/known_hosts

if [[ ! -f ~/.ssh/id_ed25519 ]]; then
  echo
  echo ">>> Paste your GitHub *private* SSH key below."
  echo ">>> Finish with CTRL-D. (It will be written to ~/.ssh/id_ed25519)"
  umask 077
  cat > ~/.ssh/id_ed25519
fi

chmod 600 ~/.ssh/id_ed25519

# Normalize CRLF if pasted from Windows
sed -i 's/\r$//' ~/.ssh/id_ed25519

# Validate key structure before proceeding
echo "[2/8] Validating SSH key…"
ssh-keygen -y -f ~/.ssh/id_ed25519 >/dev/null 2>&1 || die "Invalid private key (~/.ssh/id_ed25519). Re-run and paste a valid key."

# Start agent and add key (will fail fast if passphrase needed and no TTY)
eval "$(ssh-agent -s)" >/dev/null
if ! ssh-add ~/.ssh/id_ed25519 >/dev/null 2>&1; then
  die "Failed to add SSH key to agent. If key is passphrase-protected, run 'ssh-add ~/.ssh/id_ed25519' interactively, then retry."
fi

# Smoke test (won’t fail script if policy blocks)
ssh -T git@github.com 2>/dev/null || true

# =========================
# 3) Miniconda install
# =========================
echo "[3/8] Miniconda install/check"
if [[ ! -d "$MINICONDA" ]]; then
  cd ~
  wget -qO Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3.sh -b -p "$MINICONDA"
  echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
fi

# Load conda into this shell
eval "$("$MINICONDA/bin/conda" shell.bash hook)"

# Accept Anaconda ToS (prevents annoying errors later)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

# =========================
# 4) Conda env create/activate
# =========================
echo "[4/8] Create/activate conda env: $ENV_NAME"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# Ensure channels
conda config --add channels conda-forge   || true
conda config --add channels pytorch       || true
conda config --add channels nvidia        || true

# =========================
# 5) Clone (or update) repo via SSH
# =========================
echo "[5/8] Clone private repo via SSH"
mkdir -p /workspace
cd /workspace

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[i] Repo already exists. Pulling latest…"
  cd "$REPO_DIR"
  git fetch --all -p
  git pull --rebase --autostash || true
else
  git clone "$REPO_SSH" "$REPO_DIR"
  cd "$REPO_DIR"
fi

# =========================
# 6) Environment deps
# =========================
echo "[6/8] Install core dependencies"

# If your repo has environment-cuda.yml, use it. Otherwise install a minimal, GPU-ready stack.
if [[ -f environment-cuda.yml ]]; then
  conda env update -n "$ENV_NAME" -f environment-cuda.yml
else
  # GPU PyTorch matching CUDA 12.4
  conda install -y pytorch pytorch-cuda="$PYTORCH_CUDA" -c pytorch -c nvidia

  # Scientific stack
  conda install -y numpy pandas scipy scikit-learn lightgbm faiss-cpu \
                    sentencepiece tokenizers regex nltk matplotlib seaborn tqdm

  # Pip extras
  python -m pip install --upgrade pip
  pip install \
    sentence-transformers transformers datasets tiktoken langdetect rapidfuzz \
    accelerate joblib xgboost "$CUDA_CUPY_WHL"
fi

# =========================
# 7) Optional: NLTK data (stopwords)
# =========================
echo "[7/8] Download NLTK stopwords (if not cached)"
python - <<'PY'
import nltk
nltk.download('stopwords', quiet=True)
print("NLTK stopwords ready.")
PY

# =========================
# 8) Final checks
# =========================
echo "[8/8] Final checks"
python - <<'PY'
try:
    import torch, xgboost, sentence_transformers
    print("Torch CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Sanity check error:", e)
PY

git -C "$REPO_DIR" status || true
conda info | sed -n '1,60p'
conda env list

echo
echo "✓ Setup completed."
echo "→ Activate environment in new shells with:  conda activate $ENV_NAME"