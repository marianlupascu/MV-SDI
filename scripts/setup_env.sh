#!/bin/bash
# =============================================================================
# Setup script for threestudio + MV-SDI experiments
#
# Target hardware: NVIDIA H100 (compute capability 9.0) -- override TORCH_CUDA_ARCH_LIST
#                  / TCNN_CUDA_ARCHITECTURES below for A100 (8.0 / 80) etc.
# Target stack:    Python 3.10, CUDA toolkit 12.4, torch 2.6.0+cu124
#
# Usage: bash scripts/setup_env.sh
# Run from the repository root.
#
# This script captures every fix discovered during the May 2026 rebuild:
#   - torch must match local CUDA toolkit (12.4) -> torch 2.6.0+cu124
#   - huggingface_hub must stay >=0.27 to satisfy diffusers 0.37.1
#   - numpy MUST stay <2.0 (pandas/pyarrow/sklearn break otherwise)
#   - nvdiffrast / tinycudann / nerfacc must be rebuilt against torch 2.6+cu124
#   - xformers / flash-attn must be uninstalled (they were built for torch 2.10)
#   - HPSv2 ships without bpe vocab; needs a manual download
#   - jaxtyping is required by threestudio.utils.typing but missing on fresh installs
# =============================================================================

set -e

echo "=== Threestudio MV-SDI Environment Setup ==="
echo "Python: $(python --version 2>&1) at $(which python)"
echo "Pip:    $(pip --version)"
echo ""

# -----------------------------------------------------------------------------
# Step 0: Sanity check CUDA toolkit matches what we will install for torch
# -----------------------------------------------------------------------------
echo "=== Step 0: Verifying CUDA toolkit ==="

# CUDA_HOME may be left over from previous sessions pointing to non-existent paths.
unset CUDA_HOME 2>/dev/null || true
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME=/usr/local/cuda-12.4
fi
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit 12.4 first."
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "  nvcc release: ${NVCC_VERSION}  (expected: 12.4)"
echo "  CUDA_HOME:    ${CUDA_HOME}"

if [ "${NVCC_VERSION}" != "12.4" ]; then
    echo "  WARNING: CUDA toolkit ${NVCC_VERSION} != 12.4. Torch wheel below assumes cu124."
fi

# Default to H100 (compute capability 9.0). Override to "8.0"/"80" for A100.
# These can be set in the calling shell before running this script.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-90}"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "  TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Install/pin torch, torchvision, torchaudio for cu124
# -----------------------------------------------------------------------------
echo "=== Step 1: Installing torch 2.6.0+cu124 ==="
pip install --quiet --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

python -c "
import torch
assert torch.__version__.startswith('2.6.0+cu124'), f'Got {torch.__version__}'
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')
"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Pin numpy<2 (pandas/pyarrow/sklearn break with numpy 2.x)
# -----------------------------------------------------------------------------
echo "=== Step 2: Pinning numpy<2 (compat with pandas/pyarrow) ==="
pip install --quiet --no-cache-dir --no-deps "numpy==1.26.4"
python -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__; print(f'  numpy={numpy.__version__}')"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Pin diffusers + huggingface_hub to compatible range
# -----------------------------------------------------------------------------
echo "=== Step 3: Installing diffusers + huggingface_hub ==="
# transformers must be <5.0 for ImageReward (uses `apply_chunking_to_forward`
# from transformers.modeling_utils which was removed in 5.x).
pip install --quiet --no-cache-dir --upgrade \
    "diffusers==0.37.1" \
    "transformers>=4.45.0,<5.0" \
    "huggingface_hub>=0.27.0,<2.0" \
    "peft>=0.10.0" \
    "accelerate>=0.30.0"
python -c "
import diffusers, transformers, huggingface_hub, peft
print(f'  diffusers={diffusers.__version__}')
print(f'  transformers={transformers.__version__}')
print(f'  huggingface_hub={huggingface_hub.__version__}')
print(f'  peft={peft.__version__}')
"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Threestudio Python-only deps
# -----------------------------------------------------------------------------
echo "=== Step 4: Installing threestudio Python deps ==="
# Core threestudio runtime deps. Skip GUI/notebook stuff (gradio, IPython,
# ipywidgets, wandb) and packages we deliberately reject (xformers).
pip install --quiet --no-cache-dir \
    ninja \
    pytorch-lightning==2.0.0 \
    omegaconf \
    typeguard \
    jaxtyping \
    trimesh \
    kornia \
    controlnet-aux \
    open3d \
    ftfy regex \
    bitsandbytes \
    tensorboard \
    matplotlib \
    "imageio>=2.28.0" \
    "imageio[ffmpeg]" \
    pysdf \
    PyMCubes \
    xatlas \
    networkx \
    safetensors \
    sentencepiece \
    libigl \
    einops
# envlight is only used by PBR material (not by SDI/MVSDI). pbr_material.py was
# patched to gracefully handle the missing import, but install it best-effort.
pip install --quiet --no-cache-dir envlight 2>/dev/null \
    || pip install --quiet --no-cache-dir git+https://github.com/ashawkey/envlight.git 2>/dev/null \
    || echo "  envlight not installed (non-fatal; PBR material disabled)"
echo "  done."
echo ""

# -----------------------------------------------------------------------------
# Step 5: Build CUDA extensions against torch 2.6+cu124
# (each build can take 5-10 minutes)
# -----------------------------------------------------------------------------
echo "=== Step 5: Building CUDA extensions (this can take 15-25 min) ==="

echo "  [5a] nvdiffrast..."
pip install --quiet --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git
python -c "import nvdiffrast.torch; print('     nvdiffrast OK')"

echo "  [5b] tiny-cuda-nn (TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES})..."
pip install --quiet --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch
python -c "import tinycudann; print('     tinycudann OK')"

echo "  [5c] nerfacc (with --no-deps to keep torch pinned)..."
rm -rf ~/.cache/torch_extensions/py3*/nerfacc_cuda 2>/dev/null || true
pip install --quiet --no-cache-dir --no-deps nerfacc==0.5.3
python -c "import nerfacc; print(f'     nerfacc {nerfacc.__version__}')"
echo ""

# -----------------------------------------------------------------------------
# Step 6: Evaluation deps (CLIP, HPSv2, ImageReward)
# Eval pkg checks are best-effort: a single optional failure here should not
# abort the whole setup. Step 8 below does the authoritative sanity check.
# -----------------------------------------------------------------------------
echo "=== Step 6: Installing evaluation packages ==="
set +e

# OpenAI CLIP
pip install --quiet --no-cache-dir --no-deps git+https://github.com/openai/CLIP.git
python -c "import clip; print('  clip OK')"

# HPSv2 (note: hpsv2 module does not expose __version__, so just check import)
pip install --quiet --no-cache-dir hpsv2 image-reward
python -c "
import hpsv2
print(f'  hpsv2 OK ({hpsv2.__file__})')
" || true

# Patch ImageReward/models/BLIP/med.py for transformers >=4.40
# (`apply_chunking_to_forward`, `find_pruneable_heads_and_indices`,
# `prune_linear_layer` were moved to `transformers.pytorch_utils`;
# `BaseModelOutputWith*` moved to `transformers.modeling_outputs`).
# IMPORTANT: locate path WITHOUT importing ImageReward (the broken import is
# exactly what we are trying to fix here, so it raises before we can patch).
echo "  Patching ImageReward med.py for new transformers..."
python <<'PYEOF'
import os, re, importlib.util

spec = importlib.util.find_spec('ImageReward')
if spec is None or spec.submodule_search_locations is None:
    print("  WARNING: ImageReward package not found; skipping patch")
    raise SystemExit(0)

pkg_dir = list(spec.submodule_search_locations)[0]
med_path = os.path.join(pkg_dir, 'models', 'BLIP', 'med.py')
if not os.path.isfile(med_path):
    print(f"  WARNING: {med_path} not found, skipping patch")
    raise SystemExit(0)

with open(med_path, 'r') as f:
    src = f.read()

if 'PATCHED_FOR_TRANSFORMERS_4_40' in src:
    print("  med.py already patched")
    raise SystemExit(0)

# Match the broken tuple import; tolerant to whitespace, comments and any order
# of names inside the parentheses.
m = re.search(r"from\s+transformers\.modeling_utils\s+import\s*\(([^)]*)\)", src)
if not m:
    print("  WARNING: could not find tuple import in med.py; skipping patch")
    raise SystemExit(0)

names_inside = [n.strip() for n in m.group(1).split(',') if n.strip()]
print(f"  med.py original imports: {names_inside}")

# Map each symbol to its new home in modern transformers.
remap = {
    'apply_chunking_to_forward':                   'transformers.pytorch_utils',
    'find_pruneable_heads_and_indices':            'transformers.pytorch_utils',
    'prune_linear_layer':                          'transformers.pytorch_utils',
    'BaseModelOutputWithPastAndCrossAttentions':   'transformers.modeling_outputs',
    'BaseModelOutputWithPoolingAndCrossAttentions':'transformers.modeling_outputs',
    'PreTrainedModel':                             'transformers.modeling_utils',
}
buckets = {}
for name in names_inside:
    home = remap.get(name, 'transformers.modeling_utils')
    buckets.setdefault(home, []).append(name)

lines = ["# PATCHED_FOR_TRANSFORMERS_4_40 -- some symbols moved upstream"]
for home, names in buckets.items():
    lines.append(f"from {home} import (")
    for nm in names:
        lines.append(f"    {nm},")
    lines.append(")")
replacement = "\n".join(lines)

new_src = src[:m.start()] + replacement + src[m.end():]
with open(med_path, 'w') as f:
    f.write(new_src)
print(f"  Patched {med_path}")
PYEOF

# Verify ImageReward now imports
python -c "
import ImageReward
print(f'  ImageReward OK ({ImageReward.__file__})')
" || echo "  WARNING: ImageReward import still failing"

# HPSv2 ships without bpe vocab; download from open_clip (the authoritative
# source). The original URL on tgxs002/HPSv2 returns an empty file because the
# branch was renamed.
HPSV2_VOCAB="$(python -c "import hpsv2, os; print(os.path.join(os.path.dirname(hpsv2.__file__), 'src/open_clip/bpe_simple_vocab_16e6.txt.gz'))")"
if [ ! -s "${HPSV2_VOCAB}" ]; then
    echo "  Downloading HPSv2 / open_clip BPE vocab..."
    HPSV2_DIR="$(dirname "${HPSV2_VOCAB}")"
    mkdir -p "${HPSV2_DIR}"
    rm -f "${HPSV2_VOCAB}"  # remove any 0-byte file from a previous attempt
    DOWNLOAD_URLS=(
        "https://github.com/mlfoundations/open_clip/raw/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
        "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/bpe_simple_vocab_16e6.txt.gz"
        "https://github.com/tgxs002/HPSv2/raw/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
    )
    for URL in "${DOWNLOAD_URLS[@]}"; do
        echo "    trying $URL"
        if command -v wget &>/dev/null; then
            wget -q -O "${HPSV2_VOCAB}" "$URL" || true
        else
            curl -sL -o "${HPSV2_VOCAB}" "$URL" || true
        fi
        if [ -s "${HPSV2_VOCAB}" ]; then
            echo "    OK"
            break
        fi
    done
fi
ls -la "${HPSV2_VOCAB}" 2>/dev/null || echo "  WARNING: HPSv2 vocab still missing"

# Torchmetrics for CLIP IQA (optional, skip on numpy ABI mismatch)
pip install --quiet --no-cache-dir torchmetrics 2>/dev/null || true
set -e
echo ""

# -----------------------------------------------------------------------------
# Step 7: Remove broken xformers/flash-attn (built for the wrong torch)
# -----------------------------------------------------------------------------
echo "=== Step 7: Removing broken xformers / flash-attn (built for torch 2.10) ==="
pip uninstall -y xformers flash-attn flash_attn 2>/dev/null || true
echo ""

# -----------------------------------------------------------------------------
# Step 8: Final sanity check
# -----------------------------------------------------------------------------
echo "=== Step 8: Final sanity check ==="
python <<'EOF'
import sys
ok = True

def check(name, code):
    global ok
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        print(f'  FAIL: {name} - {e}')
        ok = False

check('torch + nms', 'import torch, torchvision; from torchvision.ops import nms; print(f"  torch {torch.__version__}, torchvision {torchvision.__version__}, nms OK")')
check('numpy<2',      'import numpy; assert numpy.__version__.startswith("1."), numpy.__version__; print(f"  numpy {numpy.__version__}")')
check('pandas',       'import pandas; print(f"  pandas {pandas.__version__}")')
check('pyarrow',      'import pyarrow; print(f"  pyarrow {pyarrow.__version__}")')
check('diffusers',    'import diffusers; print(f"  diffusers {diffusers.__version__}")')
check('transformers', 'import transformers; print(f"  transformers {transformers.__version__}")')
check('huggingface_hub', 'import huggingface_hub; print(f"  huggingface_hub {huggingface_hub.__version__}")')
check('peft',         'import peft; print(f"  peft {peft.__version__}")')
check('pytorch_lightning', 'import pytorch_lightning; print(f"  pytorch_lightning {pytorch_lightning.__version__}")')
check('nvdiffrast',   'import nvdiffrast.torch; print("  nvdiffrast OK")')
check('tinycudann',   'import tinycudann; print("  tinycudann OK")')
check('nerfacc',      'import nerfacc; print(f"  nerfacc {nerfacc.__version__}")')
check('clip',         'import clip; print("  clip OK")')
check('hpsv2',        'import hpsv2; print("  hpsv2 OK")')
check('ImageReward',  'import ImageReward; print("  ImageReward OK")')
check('jaxtyping',    'from jaxtyping import Float; print("  jaxtyping OK")')
check('typeguard',    'import typeguard; print("  typeguard OK")')
check('omegaconf',    'import omegaconf; print("  omegaconf OK")')
check('tensorboard',  'import tensorboard; print("  tensorboard OK")')
check('matplotlib',   'import matplotlib; print(f"  matplotlib {matplotlib.__version__}")')
check('imageio',      'import imageio; print(f"  imageio {imageio.__version__}")')
check('pysdf',        'import pysdf; print("  pysdf OK")')
check('mcubes',       'import mcubes; print("  PyMCubes OK")')
check('xatlas',       'import xatlas; print("  xatlas OK")')
check('safetensors',  'import safetensors; print("  safetensors OK")')
check('sentencepiece','import sentencepiece; print("  sentencepiece OK")')
check('libigl',       'import igl; print("  libigl OK")')
check('einops',       'import einops; print(f"  einops {einops.__version__}")')

if ok:
    print("\nALL CHECKS PASSED")
else:
    print("\nSome checks FAILED. Look above.")
    sys.exit(1)
EOF

echo ""
# Make sure all our shell scripts are executable
chmod +x scripts/*.sh 2>/dev/null || true

echo "================================================================"
echo "Setup complete."
echo "================================================================"
echo ""
echo "Before running experiments:"
echo "  1. Login to HuggingFace (one-time):"
echo "       huggingface-cli login"
echo "  2. Re-export the env vars in any new shell (or add to ~/.bashrc):"
echo "       export CUDA_HOME=${CUDA_HOME}"
echo "       export PATH=\$CUDA_HOME/bin:\$PATH"
echo "       export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo "       export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "       export TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}"
echo ""
echo "Run benchmarks (in tmux):"
echo "  tmux new -s bench"
echo "  ./scripts/run_mvsd_benchmark_20.sh 2>&1 | tee results/bench30.log"
echo ""
echo "  tmux new -s ablation"
echo "  ./scripts/run_mvsd_ablation_axes.sh 2>&1 | tee results/ablation30.log"
echo ""
