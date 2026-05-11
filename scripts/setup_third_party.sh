# GVHMR
if [ -d "third-party/GVHMR" ]; then
  echo "[GVHMR] Already exists, skipping."
else
  echo "[GVHMR] Cloning..."
  git clone git@github.com:carlosedubarreto/GVHMR.git "third-party/GVHMR"
fi
echo "Follow GVHMR README for setup"
echo ""