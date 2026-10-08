#!/bin/bash
# Modified by the Prompteina authors in 2026.
# Changes are licensed under Apache 2.0; see LICENSE and NOTICE.

set -e

# ==============================================================================
# Protein Foundation Models - Public UV Environment Installation Script
# ==============================================================================
# Installs every Python dependency by default; --minimal skips JAX.
# Does NOT install Foldseek, MMseqs2 or Foldcomp.
#
# Usage: ./build_uv_env.sh [OPTIONS] [INSTALL_ROOT]
#
# Options:
#   --clean         Remove existing .venv and UV cache before building (fresh start)
#   --minimal       Skip optional dependencies (JAX)
#   --python VER    Python version: 3.11 or 3.12 (default: 3.12)
#   --name NAME     Custom prompt name shown when venv is activated (default: complexa)
#   --root PATH     Specify installation root directory (where .venv will be created)
#   -h, --help      Show this help message
#
# Examples:
#   ./build_uv_env.sh                      # Full install (Python 3.12)
#   ./build_uv_env.sh --minimal            # Base dependencies only
#   ./build_uv_env.sh --python 3.11        # Full install with Python 3.11
#   ./build_uv_env.sh --name myenv         # Custom prompt: (myenv)
#   ./build_uv_env.sh --root /path/to/dir  # Create .venv in custom directory
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INSTALL_ROOT=""
FULL_INSTALL=true
CLEAN=false
PYTHON_VERSION="3.12"
VENV_NAME="complexa"

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --minimal)
            FULL_INSTALL=false
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            if [[ "$PYTHON_VERSION" != "3.11" && "$PYTHON_VERSION" != "3.12" ]]; then
                echo "Error: Python version must be 3.11 or 3.12"
                exit 1
            fi
            shift 2
            ;;
        --name)
            VENV_NAME="$2"
            shift 2
            ;;
        --root)
            INSTALL_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./build_uv_env.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean         Remove existing .venv and UV cache before building"
            echo "  --minimal       Skip optional dependencies (JAX)"
            echo "  --python VER    Python version: 3.11 or 3.12 (default: 3.12)"
            echo "  --name NAME     Custom prompt name (default: complexa)"
            echo "  --root PATH     Specify where to create .venv (default: project dir)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./build_uv_env.sh                # Full install (Python 3.12)"
            echo "  ./build_uv_env.sh --minimal      # Base dependencies only"
            echo "  ./build_uv_env.sh --python 3.11  # Full install with Python 3.11"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$INSTALL_ROOT" ]]; then
                INSTALL_ROOT="$1"
            else
                echo "Error: Multiple paths specified"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -n "$INSTALL_ROOT" ]]; then
    VENV_DIR="$INSTALL_ROOT"
    mkdir -p "$VENV_DIR"
else
    VENV_DIR="$PROJECT_DIR"
fi

cd "$VENV_DIR"

export UV_CACHE_DIR="$VENV_DIR/.uv-cache"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

if [[ "$CLEAN" == "true" ]]; then
    echo "Cleaning existing environment and cache..."
    rm -rf "$VENV_DIR/.venv"
    rm -rf "$UV_CACHE_DIR"
    uv cache clean 2>/dev/null || true
    echo "Clean complete."
fi

echo "=============================================="
echo "  Proteina-Complexa - Public Environment Builder"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Install directory: $VENV_DIR"
echo "Python version: $PYTHON_VERSION"
echo "Prompt name: $VENV_NAME"
echo "Full install: $FULL_INSTALL"
echo ""

# ------------------------------------------------------------------------------
# 1. Check/Install UV
# ------------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "[1/8] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/8] uv already installed: $(uv --version)"
fi

# ------------------------------------------------------------------------------
# 2. Create virtual environment
# ------------------------------------------------------------------------------
echo "[2/8] Creating virtual environment with Python $PYTHON_VERSION..."
uv venv --python "$PYTHON_VERSION" --prompt "$VENV_NAME" "$VENV_DIR/.venv"

source "$VENV_DIR/.venv/bin/activate"
echo "      Python: $(which python)"

# ------------------------------------------------------------------------------
# 3. Install PyTorch with CUDA 12.6
# ------------------------------------------------------------------------------
echo "[3/8] Installing PyTorch 2.7.0 with CUDA 12.6..."
# torchvision is pinned to its torch 2.7.0 companion. Unpinned, it resolves
# against a newer torch and the compiled extension fails to load.
# torchaudio is not installed: nothing here imports it, and transformers only
# imports it when it is present (audio_utils.py, behind is_torchaudio_available).
# Installed unpinned it resolved to 2.11.0 against torch 2.7.0, and then
# `undefined symbol: torch_library_impl` took down every transformers import.
uv pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# ------------------------------------------------------------------------------
# 4. Install base dependencies from pyproject.toml
# ------------------------------------------------------------------------------
echo "[4/8] Installing base dependencies from pyproject.toml..."
uv pip install --index-strategy unsafe-best-match -e "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# 5. Install PyTorch Geometric packages
# ------------------------------------------------------------------------------
echo "[5/8] Installing PyTorch Geometric..."
# torch_geometric only; it is pure Python. torch_scatter, torch_sparse and
# torch_cluster are not installed: the first was used in exactly one place
# (replaced by native torch in nn/feature_factory/seq_cond_feats.py) and the
# other two are imported nowhere. Their compiled wheels for torch 2.7 + cu126
# need GLIBC_2.32, so on RHEL 8 (glibc 2.28) they break the whole package.
uv pip install torch_geometric

# ------------------------------------------------------------------------------
# 6. Install Graphein and Atomworks
# ------------------------------------------------------------------------------
echo "[6/8] Installing Graphein and Atomworks..."
echo "      -> Graphein..."
uv pip install graphein==1.7.7 --no-deps

echo "      -> Atomworks..."
# Pinned. atomworks 3.0.1 removed AtomSelectionStack, which
# src/proteinfoundation/utils/pdb_utils.py imports, so an unpinned install
# produces an environment where the package cannot be imported at all.
# Not tolerated with `|| true`: without atomworks nothing here runs, so a
# failure here must fail the build rather than leave a broken venv behind.
# The openbabel extra is left out on purpose: OpenBabel is GPL-2.0 and only
# ProteinaLigandTransform(use_openbabel=True) needs it. Install the project's
# `openbabel` extra to get it.
uv pip install "atomworks[ml,dev]==2.2.0"

# ------------------------------------------------------------------------------
# 7. Install optional/full dependencies
# ------------------------------------------------------------------------------
if [ "$FULL_INSTALL" = true ]; then
    echo "[7/8] Installing optional dependencies (JAX, for AlphaFold3)..."

    # JAX is only needed by the AlphaFold3 reward and folding backend. The
    # ESMFold2 path is pure torch. AlphaFold3 pins a newer JAX than this, so
    # run it in its own environment rather than here.
    echo "      -> JAX with CUDA..."
    uv pip install jaxlib==0.4.29+cuda12.cudnn91 \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    uv pip install "jax[cuda12]==0.4.29" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    uv pip install flax==0.9.0 --no-deps

else
    echo "[7/8] Skipping optional dependencies (omit --minimal to install)"
fi

echo "[8/8] Pinning biotite to 1.6.0..."
uv pip install biotite==1.6.0

# ------------------------------------------------------------------------------
# Done!
# ------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/.venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo ""
echo "Note: Foldseek and MMseqs2 are not installed. They are needed only to"
echo "rebuild the cluster TSVs yourself; the data release ships them prebuilt."
echo "To install:"
echo "  wget https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz"
echo "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
