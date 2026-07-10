# Runtime Environment Inventory

This file is the current source of truth for AudioHOI runtime environments. It records what is actually installed on this machine, which environment should run each stage, and which environments must not be mixed.

Checked on: 2026-07-10

Repo config:

```text
scripts/shared/generic_contact_pipeline/configs/runtime_envs.yaml
```

Conda reported these environments on this machine:

```text
base
agent-backend-lab
all-in-rag
articraft-py312
audiohoi
bodyrender
da3
fastapi-learn
gvhmr
hamer
langent
learn-claude-code
qwen-vl
sam3d-objects-inference
```

Only the AudioHOI-related environments below are part of the current pipeline map.

## Short Rule

Use explicit Python paths. Do not rely on the activated shell environment.

| Task | Environment | Python |
| --- | --- | --- |
| Main generic pipeline | `audiohoi` | `/home/yang/miniconda3/envs/audiohoi/bin/python` |
| Final HOI evaluator | `audiohoi` | `/home/yang/miniconda3/envs/audiohoi/bin/python` |
| Ablation evaluator | `audiohoi` | `/home/yang/miniconda3/envs/audiohoi/bin/python` |
| SAM2 mask consumption / OpenCV overlay metrics | `audiohoi` | `/home/yang/miniconda3/envs/audiohoi/bin/python` |
| Local Qwen-VL judge | `qwen-vl` | `/home/yang/miniconda3/envs/qwen-vl/bin/python` |
| DA3 depth / point cloud preprocessing | `da3` | `/home/yang/miniconda3/envs/da3/bin/python` |
| GVHMR body reconstruction | `gvhmr` | `/home/yang/miniconda3/envs/gvhmr/bin/python` |
| HaMeR hand reconstruction / pyrender hand diagnostics | `hamer` | `/home/yang/miniconda3/envs/hamer/bin/python` |
| Articraft wrapper check | `articraft-py312` | `/home/yang/miniconda3/envs/articraft-py312/bin/python` |
| Optional SAM3D/Open3D object experiments | `sam3d-objects-inference` | `/home/yang/miniconda3/envs/sam3d-objects-inference/bin/python` |

## Pipeline Stage Mapping

| Pipeline part | Use env | Why |
| --- | --- | --- |
| `run_pipeline.py` orchestration | `audiohoi` | Has numpy, OpenCV, torch, transformers, qwen helpers, trimesh, scipy, pandas, smplx, sam2. |
| Stage0 preprocessing manifest / artifact checks | `audiohoi` | Reads and validates artifacts; dispatches external outputs but should not switch shell Python. |
| DINO/SAM2 mask usage inside generic pipeline | `audiohoi` | `sam2` import is available here. |
| Qwen-VL stage/final visual judge | `qwen-vl` | Has Qwen/transformers stack and CUDA torch; intentionally light on geometry packages. |
| LLM CSV audit / Mistral API client logic | `audiohoi` | Main evaluator/runtime env; API key comes from shell env. |
| DA3 depth generation or depth utility reruns | `da3` | Has Open3D, moviepy, DA3-friendly geometry stack. |
| GVHMR human body generation | `gvhmr` | Has torch 2.3, smplx, trimesh, imageio for GVHMR outputs. |
| HaMeR hand generation and hand overlay diagnostics | `hamer` | Has `hamer` and `pyrender`; the main env does not. |
| Final result evaluation table generation | `audiohoi` | The evaluator is written against the generic pipeline schema and lightweight metrics. |
| Render-mask evaluation fallback | `audiohoi` first | Uses OpenCV/trimesh/lightweight render code; only switch if a renderer explicitly requires `pyrender`. |

## Environment Details

### audiohoi

Path:

```text
/home/yang/miniconda3/envs/audiohoi/bin/python
Python 3.10.20
CUDA available: yes, torch CUDA 12.1
```

Role:

```text
Default runtime for the generic pipeline, final HOI evaluator, ablation evaluator,
CSV/JSON metrics, OpenCV overlays, mask usage, temporal/contact metrics, and most
lightweight render/evaluation code.
```

Installed key packages:

| Package | Status |
| --- | --- |
| numpy | ok 1.26.4 |
| cv2 | ok 4.13.0 |
| PIL | ok 12.2.0 |
| torch | ok 2.4.1+cu121 |
| torchvision | ok 0.19.1+cu121 |
| transformers | ok 5.11.0 |
| qwen_vl_utils | ok 0.0.14 |
| accelerate | ok 1.13.0 |
| bitsandbytes | ok 0.49.2 |
| trimesh | ok 4.12.2 |
| scipy | ok 1.15.3 |
| matplotlib | ok 3.9.1 |
| yaml | ok 6.0.3 |
| pandas | ok 2.3.3 |
| smplx | ok 0.1.28 |
| sam2 | ok |
| imageio | ok 2.34.1 |
| ffmpeg | ok |
| pyrender | missing |
| hamer | missing |
| segment_anything | missing |
| open3d | missing |
| moviepy | missing |
| pytest | missing |

Use for:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case stick \
  --result-name benchmark_vlm_qwen
```

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  --result-name benchmark_vlm_qwen \
  --output-dir samples_known_object/final_result_evaluation
```

Do not use it for HaMeR/pyrender-specific hand diagnostics. It does not have `pyrender` or `hamer`.

### qwen-vl

Path:

```text
/home/yang/miniconda3/envs/qwen-vl/bin/python
Python 3.10.20
CUDA available: yes, torch CUDA 12.1
```

Role:

```text
Dedicated local Qwen-VL visual judge environment.
```

Installed key packages:

| Package | Status |
| --- | --- |
| numpy | ok 2.2.6 |
| cv2 | ok 4.13.0 |
| PIL | ok 12.2.0 |
| torch | ok 2.5.1+cu121 |
| torchvision | ok 0.20.1+cu121 |
| transformers | ok 5.11.0 |
| qwen_vl_utils | ok 0.0.14 |
| accelerate | ok 1.13.0 |
| bitsandbytes | ok 0.49.2 |
| yaml | ok 6.0.3 |
| pandas | ok 2.3.3 |
| trimesh | missing |
| scipy | missing |
| matplotlib | missing |
| smplx | missing |
| sam2 | missing |
| imageio | missing |
| moviepy | missing |
| pytest | missing |

Use for:

```text
Qwen-VL stage audit
Qwen-VL final visual judge
visual evidence prompt/response parsing when the local Qwen model is required
```

Do not use it as the default pipeline/evaluator environment. It lacks geometry and HOI metric dependencies.

### bodyrender

Path:

```text
/home/yang/miniconda3/envs/bodyrender/bin/python
Python 3.10.20
CUDA available: yes, torch CUDA 12.1
```

Role:

```text
Optional body-render experiments only.
```

Installed key packages:

| Package | Status |
| --- | --- |
| numpy | ok 2.2.5 |
| cv2 | ok 4.13.0 |
| PIL | ok 12.2.0 |
| torch | ok 2.4.1+cu121 |
| torchvision | ok 0.19.1+cu121 |
| scipy | ok 1.15.3 |
| matplotlib | ok 3.10.9 |
| smplx | ok 0.1.28 |
| transformers | missing |
| qwen_vl_utils | missing |
| trimesh | missing |
| pyrender | missing |
| yaml | missing |
| pandas | missing |
| imageio | missing |
| pytest | missing |

Use this only for narrow body-render experiments that do not need the missing packages. It is not the default evaluator or mesh renderer runtime.

### da3

Path:

```text
/home/yang/miniconda3/envs/da3/bin/python
Python 3.10.20
CUDA available: yes, torch CUDA 12.1
```

Role:

```text
DA3 depth, point-cloud preprocessing, and depth utility reruns.
```

Installed key packages:

| Package | Status |
| --- | --- |
| numpy | ok 2.2.6 |
| cv2 | ok 4.11.0 |
| PIL | ok 12.2.0 |
| torch | ok 2.4.1+cu121 |
| torchvision | ok 0.19.1+cu121 |
| trimesh | ok 4.12.2 |
| scipy | ok 1.14.1 |
| matplotlib | ok 3.10.9 |
| yaml | ok 6.0.3 |
| pandas | ok 2.3.3 |
| open3d | ok 0.19.0 |
| imageio | ok 2.37.3 |
| moviepy | ok 1.0.3 |
| transformers | missing |
| qwen_vl_utils | missing |
| smplx | missing |
| sam2 | missing |
| pyrender | missing |
| pytest | missing |

Use this for DA3/depth-specific preprocessing. Do not use it for Qwen, final HOI evaluation, or human hand/body reconstruction.

### gvhmr

Path:

```text
/home/yang/miniconda3/envs/gvhmr/bin/python
Python 3.10.20
CUDA available: yes, torch CUDA 12.1
```

Role:

```text
GVHMR human body reconstruction and SMPL-X body outputs.
```

Installed key packages:

| Package | Status |
| --- | --- |
| numpy | ok 1.23.5 |
| cv2 | ok 4.11.0 |
| PIL | ok 12.2.0 |
| torch | ok 2.3.0+cu121 |
| torchvision | ok 0.18.0+cu121 |
| trimesh | ok 4.12.2 |
| scipy | ok 1.15.3 |
| matplotlib | ok 3.10.9 |
| yaml | ok 6.0.3 |
| pandas | ok 2.3.3 |
| smplx | ok 0.1.28 |
| imageio | ok 2.34.1 |
| hamer | missing |
| sam2 | missing |
| open3d | missing |
| pyrender | missing |
| pytest | missing |

Use this to generate human reconstruction artifacts. Use `audiohoi` to consume those artifacts in the generic pipeline/evaluator.

### hamer

Path:

```text
/home/yang/miniconda3/envs/hamer/bin/python
Python 3.10.20
CUDA available: yes, torch CUDA 12.1
```

Role:

```text
HaMeR hand reconstruction, hand diagnostics, and pyrender hand/body overlays.
```

Installed key packages:

| Package | Status |
| --- | --- |
| numpy | ok 1.23.5 |
| cv2 | ok 4.11.0 |
| PIL | ok 12.2.0 |
| torch | ok 2.4.1+cu121 |
| torchvision | ok 0.19.1+cu121 |
| trimesh | ok 4.12.2 |
| pyrender | ok 0.1.45 |
| scipy | ok 1.15.3 |
| matplotlib | ok 3.10.9 |
| yaml | ok 6.0.3 |
| pandas | ok 2.3.3 |
| smplx | ok 0.1.28 |
| hamer | ok 0.0.0 |
| imageio | ok 2.37.3 |
| sam2 | missing |
| qwen_vl_utils | missing |
| pytest | missing |

Use for:

```text
scripts/shared/human/hands/run_hamer_hands.py
scripts/shared/human/hands/diagnose_hand_overlay.py
pyrender-based hand/body diagnostic renders
```

Do not use it as the Qwen or main evaluator environment.

### articraft-py312

Path:

```text
/home/yang/miniconda3/envs/articraft-py312/bin/python
Python 3.12.13
CUDA available: not applicable in probe
```

Role:

```text
Articraft asset generation wrapper/check only.
```

Installed key packages:

| Package | Status |
| --- | --- |
| numpy | missing |
| cv2 | missing |
| PIL | missing |
| torch | missing |
| torchvision | missing |
| transformers | missing |
| qwen_vl_utils | missing |
| trimesh | missing |
| yaml | missing |
| pandas | missing |
| imageio | missing |
| pytest | missing |

This is not a ready-to-run geometry/evaluation environment. Treat it as a thin wrapper or placeholder until project-local Articraft dependencies are confirmed.

### sam3d-objects-inference

Path:

```text
/home/yang/miniconda3/envs/sam3d-objects-inference/bin/python
Python 3.11.0
CUDA available: yes, torch CUDA 12.1
```

Role:

```text
Optional SAM3D object inference experiments and Open3D object geometry utilities.
```

Installed key packages:

| Package | Status |
| --- | --- |
| numpy | ok 1.26.4 |
| cv2 | ok 4.9.0 |
| PIL | ok 12.2.0 |
| torch | ok 2.5.1+cu121 |
| torchvision | ok 0.20.1+cu121 |
| trimesh | ok 4.12.2 |
| scipy | ok 1.17.1 |
| matplotlib | ok 3.10.9 |
| yaml | ok 6.0.3 |
| pandas | ok 3.0.3 |
| open3d | ok 0.18.0 |
| imageio | ok 2.37.3 |
| transformers | missing |
| qwen_vl_utils | missing |
| smplx | missing |
| sam2 | missing |
| pyrender | missing |
| pytest | missing |

Use this only for optional SAM3D/Open3D object experiments. It is not the default pipeline/evaluator runtime.

## Test Runner Note

The current shell was in `base` during inspection:

```text
/home/yang/miniconda3
```

`pytest` is available from base:

```text
/home/yang/miniconda3/bin/pytest
```

The `audiohoi` environment does not currently have `pytest`. For repo tests, use:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/yang/miniconda3/bin/pytest -q \
  tests/test_final_hoi_evaluation.py \
  tests/test_vlm_trace_final_evaluator_benchmark.py
```

For compile/import sanity on pipeline code, use:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python -B -m compileall -q \
  scripts/shared/generic_contact_pipeline
```

## Refresh Command

Run from repo root to refresh the key-package probe:

```bash
python - <<'PY'
from pathlib import Path
import subprocess, json
relevant = [
    ("audiohoi", "/home/yang/miniconda3/envs/audiohoi/bin/python"),
    ("qwen-vl", "/home/yang/miniconda3/envs/qwen-vl/bin/python"),
    ("bodyrender", "/home/yang/miniconda3/envs/bodyrender/bin/python"),
    ("da3", "/home/yang/miniconda3/envs/da3/bin/python"),
    ("gvhmr", "/home/yang/miniconda3/envs/gvhmr/bin/python"),
    ("hamer", "/home/yang/miniconda3/envs/hamer/bin/python"),
    ("articraft-py312", "/home/yang/miniconda3/envs/articraft-py312/bin/python"),
    ("sam3d-objects-inference", "/home/yang/miniconda3/envs/sam3d-objects-inference/bin/python"),
]
mods = [
    "numpy", "cv2", "PIL", "torch", "torchvision", "transformers",
    "qwen_vl_utils", "accelerate", "bitsandbytes", "trimesh", "pyrender",
    "scipy", "matplotlib", "yaml", "pandas", "smplx", "hamer", "sam2",
    "segment_anything", "open3d", "imageio", "moviepy", "ffmpeg", "pytest",
]
probe = r'''
import importlib, importlib.metadata as md, json, sys, platform
mods = __MODS__
out = {"python": sys.executable, "version": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            try:
                ver = md.version(m)
            except Exception:
                ver = "ok"
        out["packages"][m] = str(ver)
    except Exception:
        out["packages"][m] = "missing"
try:
    import torch
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["cuda_version"] = getattr(torch.version, "cuda", None)
except Exception:
    out["cuda_available"] = None
    out["cuda_version"] = None
print(json.dumps(out, ensure_ascii=False))
'''.replace('__MODS__', repr(mods))
for name, py in relevant:
    if not Path(py).exists():
        print(json.dumps({"env": name, "python": py, "exists": False}, ensure_ascii=False))
        continue
    p = subprocess.run([py, "-c", probe], text=True, capture_output=True, timeout=30)
    print(name, p.stdout if p.returncode == 0 else p.stderr)
PY
```
