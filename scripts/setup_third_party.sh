# GVHMR
if [ -d "third-party/GVHMR" ]; then
  echo "[GVHMR] Already exists, skipping."
else
  echo "[GVHMR] Cloning..."
  git clone git@github.com:carlosedubarreto/GVHMR.git "third-party/GVHMR"
fi
echo "Follow GVHMR README for setup"
echo ""

# Depth Anything 3 (per-frame metric depth; gvhmr env)
if [ -d "third-party/depth-anything-3" ]; then
  echo "[DA3] Already exists, skipping."
else
  echo "[DA3] Cloning..."
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "third-party/depth-anything-3"
fi
# Install into the gvhmr env without DA3's heavy pins (open3d/pycolmap/numpy<2 would
# break gvhmr); only the inference path is needed. Checkpoint auto-downloads from HF.
echo "[DA3] Install with:"
echo "  conda run -n gvhmr pip install -e third-party/depth-anything-3 --no-deps"
echo "  conda run -n gvhmr pip install e3nn addict 'moviepy==1.0.3' plyfile && conda run -n gvhmr pip install evo --no-deps"
echo ""