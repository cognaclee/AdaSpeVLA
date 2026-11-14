#!/usr/bin/env bash
# Source from any working directory. Select the interpreter explicitly per benchmark.
export ADASPEVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ADASPEVLA_WORKSPACE="$(dirname "$ADASPEVLA_ROOT")"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$ADASPEVLA_WORKSPACE/external/RoboTwin}"
export HF_HOME="${HF_HOME:-$ADASPEVLA_WORKSPACE/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$ADASPEVLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TRAINING_PYTHON="${TRAINING_PYTHON:-/home/haichao/miniconda3/envs/adaspevla/bin/python}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/home/haichao/miniconda3/envs/RoboTwin/bin/python}"
export ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-/home/haichao/miniconda3/envs/adaspevla_robocasa/bin/python}"
