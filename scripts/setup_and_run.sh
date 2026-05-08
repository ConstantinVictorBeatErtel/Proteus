#!/bin/bash
set -e
cd ~/RAID

echo "=== Downloading GR-1 checkpoints ==="
mkdir -p checkpoints/gr1
wget -q --show-progress \
  https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr1_code_release/snapshot_ABCD.pt \
  -O checkpoints/gr1/snapshot_ABCD.pt
wget -q --show-progress \
  https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth \
  -O checkpoints/gr1/mae_pretrain_vit_base.pth

echo "=== Installing dependencies ==="
pip install -q robosuite==1.4.0 bddl cloudpickle gym easydict \
               imageio imageio-ffmpeg timm Pillow
pip install -q git+https://github.com/openai/CLIP.git
pip install -q git+https://github.com/Lifelong-Robot-Learning/LIBERO.git

echo "=== Verifying EGL rendering ==="
MUJOCO_GL=egl python3 -c "import mujoco; print('EGL OK')" || {
  echo "EGL failed, installing mesa and retrying..."
  sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
  MUJOCO_GL=egl python3 -c "import mujoco; print('EGL OK')" || {
    echo "Falling back to osmesa"
    sudo apt-get install -y libosmesa6-dev
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
  }
}

echo "=== Caching GR-1 features ==="
MUJOCO_GL=egl python3 src/cache_gr1_features.py \
  --dataset_dir data/libero_spatial/libero_spatial \
  --output_dir data/libero_spatial/features \
  --device cuda

echo "=== Running BC sweep ==="
MUJOCO_GL=egl python3 src/run_all_libero.py \
  --feature_dir data/libero_spatial/features \
  --device cuda

echo "=== Generating videos ==="
mkdir -p outputs
MUJOCO_GL=egl python3 scripts/compare_video.py \
  --task_idx 1 --n_demos 200 --n_episodes 3 --output_dir outputs/

echo ""
echo "=== DONE. Files in outputs/: ==="
ls -lh outputs/
echo ""
echo "=== Download command (run on your laptop): ==="
echo "scp -r ubuntu@$(curl -s ifconfig.me):~/RAID/outputs/ ./raid_outputs/"
