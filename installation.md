# EgoAlign Installation Guide

EgoAlign depends on a mix of standard pip packages and a few libraries that require special installation steps (SAM2, ViTPose, Shadow fileio, and projectaria_tools). Follow the section for your operating system, then complete the shared steps at the bottom.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Platform setup](#2-platform-setup)
   - [Linux](#linux)
   - [macOS](#macos)
   - [Windows](#windows)
3. [Create a virtual environment](#3-create-a-virtual-environment)
4. [Install PyTorch](#4-install-pytorch)
5. [Install pip dependencies](#5-install-pip-dependencies)
6. [Install special packages](#6-install-special-packages)
   - [SAM2](#sam2)
   - [projectaria_tools](#projectaria_tools)
   - [ViTPose](#vitpose)
   - [Shadow fileio](#shadow-fileio)
7. [Verify the installation](#7-verify-the-installation)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 – 3.12 | 3.11 recommended |
| CUDA-capable GPU | Strongly recommended for SAM2 and ViTPose; CPU-only mode is slow |
| CUDA Toolkit 11.8 or 12.x | Must match the PyTorch build you install |
| Conda or venv | Either works; instructions below use `venv` |


---

## 2. Create a virtual environment

Run these commands from the root of the EgoAlign repository.

**Linux / macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

> If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first.

---

## 3. Install PyTorch

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and use the selector to generate the exact command for your OS + CUDA version. Common examples are shown below.

**CUDA 12.1 (Linux / Windows):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8 (Linux / Windows):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU only / macOS:**

```bash
pip install torch torchvision
```

Verify the installation:

```python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 4. Install pip dependencies

With the virtual environment active, install all remaining packages from the requirements file:

```bash
pip install -r requirements.txt
```

---

## 5. Install special packages

These packages cannot be installed directly from PyPI and require a few extra steps.

### SAM2

SAM2 (Segment Anything 2) is installed directly from the Facebook Research GitHub repo:

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

Download the model checkpoint you need (e.g. `sam2_hiera_large.pt`) from the [SAM2 github page](https://github.com/facebookresearch/sam2?tab=readme-ov-file) and note its path (you will pass it to `segment_video.py` via `--checkpoint`).

---

### projectaria_tools

Install via PyPI:

```bash
pip install projectaria-tools
```

> On some platforms you may need to install the full version with extras:
> ```bash
> pip install "projectaria-tools[all]"
> ```
> See the [projectaria_tools docs](https://facebookresearch.github.io/projectaria_tools/docs/installation) for troubleshooting.

---

### ViTPose

```bash
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -r requirements.txt
pip install -v -e .
cd ..
```

Download the ViTPose++ checkpoint(s) you need from the [ViTPose model zoo](https://github.com/ViTAE-Transformer/ViTPose#model-zoo) and note the directory (you will pass it to `vitpose_inference.py` via `--checkpoints_dir`).

If you run into issues installing chumpy, run the following command: `pip install chumpy --no-build-isolation`

---

### Shadow fileio

Clone the shadow github to convert motion capture data to .csv:

```bash
git clone https://github.com/motion-workshop/shadow-fileio-python.git
cd shadow-fileio-python
python setup.py --install
```

Change name of folder to "shadow"

---

## 6. Verify the installation

Run the following snippet to confirm that all major imports succeed:

```bash
python - <<'EOF'
import numpy, scipy, pandas, cv2, matplotlib, open3d, torch, tqdm
import pytz, timezonefinder
from projectaria_tools.core import data_provider
from sam2.build_sam import build_sam2
import shadow.shadow.fileio as sf
print("All imports OK")
print(f"  torch  {torch.__version__}  CUDA={torch.cuda.is_available()}")
print(f"  open3d {open3d.__version__}")
EOF
```

If everything is installed correctly you will see `All imports OK` followed by version information.

For the Shadow and ViTPose lines to succeed, make sure you have installed those packages and that your virtual environment is active.
