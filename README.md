# SAM Weak Labels

Turns a Roboflow project full of unlabelled frames into a project full of reviewable
segmentation weak labels using three commands: `seed`, `fit`, and `label`.

**How it works:** `seed` picks a small, visually diverse set of frames for you to annotate
by hand. `fit` learns — from those few annotations — which of SAM's unsupervised candidate
masks is your target object. `label` runs SAM over every image, selects the target mask,
simplifies it to a minimal-vertex polygon, and attaches it to each image in Roboflow as a
prediction routed into Review.

---

## Prerequisites

| Requirement | Version / note |
|---|---|
| Python | **3.12** — `sam-2` does not build on 3.13 (no cp313 wheel) |
| `uv` | pip-compatible installer; used because `/tmp` may be small |
| CUDA GPU | Required for SAM2 and DINOv2. Tested on RTX 4090 + CUDA 13.0 |
| Roboflow account | Project must already contain images (Unassigned) |

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

### 1. Create the virtual environment

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA support

```bash
TMPDIR=~/tmp_pip UV_CACHE_DIR=~/.cache/uv \
  uv pip install torch==2.12.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu130
```

Set `TMPDIR` and `UV_CACHE_DIR` to avoid failures if `/tmp` is nearly full. Adjust
`cu130` to match your CUDA version (`cu118`, `cu121`, etc.).

### 3. Install SAM-2 from source

```bash
TMPDIR=~/tmp_pip uv pip install "git+https://github.com/facebookresearch/sam2.git"
```

SAM-2 has no PyPI release; this fetches HEAD from GitHub.

### 4. Install remaining dependencies

```bash
TMPDIR=~/tmp_pip uv pip install -r requirements.txt
```

### 5. Download SAM 2.1 hiera-large weights

```bash
mkdir -p models
wget -q -O models/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The file is ~856 MB. The path `models/sam2.1_hiera_large.pt` matches the default in
`config.example.yaml`; if you put it elsewhere, update `sam.weights` in `config.yaml`.

### 6. Configure the project

```bash
cp config.example.yaml config.yaml
```

Open `config.yaml` and fill in the two required fields:

```yaml
roboflow:
  workspace: your-workspace-slug   # the slug in your Roboflow project URL
  project_id: your-project-slug
```

Everything else has working defaults; see the [Config reference](#config-reference) below.

### 7. Set your Roboflow API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```
ROBOFLOW_API_KEY=your_key_here
```

Find the key at **Roboflow → Account → Settings → API**.  `.env` is gitignored and never
committed.

### 8. Verify the setup

```bash
python main.py doctor
```

Expected output includes `sam2 import OK`, your GPU name, and `ROBOFLOW_API_KEY : set ✓`.

```bash
python main.py doctor --remote
```

Expected: a table of your project's batches and the unassigned image count.

- **Exit 0** — environment is fully healthy (total image count shown).
- **Exit 1 with "degraded — /search unavailable"** — `/search` returns 404 on your Roboflow
  plan. This is expected on many plans. The pipeline uses the annotation-batch admin API
  (`annotation-batches/*/images`) for all image enumeration, so `manifest build`, `seed`, and
  `label` all work without `/search`.

If you see a traceback or "Config error", something in steps 1–7 above needs attention.

---

## Quick-start: the three-command workflow

```
seed  →  [annotate in Roboflow UI]  →  fit  →  label
```

### Step 1 — Build the image manifest

```bash
python main.py manifest build
```

Enumerates all images in your project via the Roboflow annotation-batch API and writes
`data/manifest.jsonl`. Required before `seed` or `label`. Re-run whenever images are
added to the project.

### Step 2 — Select seed frames

```bash
python main.py seed --dry-run     # preview selection without writing
python main.py seed               # tag 40 frames in Roboflow
```

Selects `seed.n_frames` visually diverse images by farthest-point sampling over Roboflow's
per-image embeddings, then tags each selected image with `seed_tag` (default:
`to_manually_annotate`). Falls back to seeded-random sampling if embeddings are absent.

The command prints the exact Roboflow UI click-path for turning the tagged set into an
annotation job.

### Step 3 — Annotate in Roboflow (manual)

In the Roboflow web UI:

1. Open your project → **Annotate** tab.
2. Filter images by the tag `to_manually_annotate`.
3. Select all filtered images → **Assign** → create a new annotation job.
4. Draw one segmentation polygon per target object on each frame.
5. Review and approve the annotations.
6. (Optional) Export as a COCO-segmentation JSON if you want to use `--coco` below.

There is no shortcut here — the quality of these hand-drawn polygons is what the classifier
is trained on.

### Step 4 — Train the matcher

```bash
python main.py fit
```

For each annotated seed frame:
- Runs SAM over the full granularity ladder (all three rungs).
- Labels each SAM candidate positive (IoU ≥ 0.75 vs your annotation) or negative.
- Extracts DINOv2 + geometry/colour features per candidate.

Then trains an L2 logistic regression and reports:
- `pool_recall` — what fraction of seed frames had any good SAM candidate at all, per rung.
- `top1_accuracy` — of frames with a good candidate, how often the classifier picked it.
- A precision/coverage table and the threshold achieving ≥ 0.95 precision.

The model is saved to `data/model/matcher.joblib`.  If `pool_recall` at the finest rung is
low (below ~0.7), SAM's candidate pool is missing the target — adding more seed frames or
changing the ladder config is more productive than re-running `label`.

If you exported a COCO file from Roboflow, pass it with `--coco`:

```bash
python main.py fit --coco path/to/export.json
```

### Step 5 — Label all images

Always pilot with `--dry-run` first:

```bash
python main.py label --dry-run --limit 20
```

This writes overlay PNGs and COCO files to `data/debug/` and makes **zero** write calls to
Roboflow. Inspect the overlays to confirm the masks are on target.

Then run a small live pilot:

```bash
python main.py label --limit 5
```

Check the Roboflow UI: those 5 images should appear in Review with the polygon and the tag
`weak_label_v1`. Re-running the same command reports `5 already processed, 0 uploaded` —
the progress ledger prevents double-annotation.

When satisfied:

```bash
python main.py label
```

Labels every image in the manifest. The run is fully resumable: interrupt with Ctrl-C and
re-run; already-processed images are skipped.

---

## Retuning

If the initial `label` run produces too many wrong masks:

1. **Annotate more seed frames.** Add a second batch in Roboflow (filter by a new tag,
   assign, annotate). The more annotated frames, the better the classifier generalises.
2. **Re-run `fit`.** The model re-trains on all annotated seed frames.
3. **Start a new label run** with an incremented version tag:

```bash
python main.py label --version 2
```

This writes the tag `weak_label_v2` to each newly labelled image, so the two runs are
distinguishable in Roboflow's tag filter. Old `weak_label_v1` images are not touched.

---

## Reverting a bad run

Every weak-labelled image receives a tag `weak_label_v<N>` (e.g. `weak_label_v1`).

To revert:

1. In the Roboflow UI, filter by the tag `weak_label_v1`.
2. Select all → **Delete annotations** (or move back to Unassigned if you prefer).

The tag name is controlled by `roboflow.weak_label_tag_prefix` + `--version` (default 1).

---

## Commands reference

```
python main.py doctor                 Verify config, torch, sam2, and API key
python main.py doctor --remote        Also query Roboflow batches and image counts
python main.py doctor --fetch N       Download N images and verify the local cache
python main.py doctor --gt            Report ground-truth coverage for seed frames
python main.py doctor --gt --coco P   Same, using a COCO export file instead of the API

python main.py manifest build         Enumerate images → data/manifest.jsonl

python main.py seed                   Select and tag seed frames
python main.py seed --n N             Override n_frames from config
python main.py seed --dry-run         Preview selection; no writes

python main.py fit                    Train the matcher (requires annotated seed frames)
python main.py fit --coco PATH        Use a COCO export as the ground-truth source

python main.py label                  Label all manifest images
python main.py label --dry-run        Write overlays/COCO locally; no Roboflow writes
python main.py label --limit N        Cap run at N images
python main.py label --version N      Use tag suffix N (default 1)
```

---

## Config reference

`config.yaml` (copy from `config.example.yaml`; gitignored):

```yaml
roboflow:
  workspace: <workspace-slug>          # REQUIRED — slug from your Roboflow project URL
  project_id: <project-slug>           # REQUIRED
  seed_tag: to_manually_annotate       # tag applied by `seed`
  weak_label_tag_prefix: weak_label_v  # combined with --version → weak_label_v1

target:
  class_name: <target-class>           # REQUIRED — name used in the COCO annotation
  category_id: 0

seed:
  n_frames: 40                         # number of frames to select for manual annotation
  strategy: farthest_point             # "farthest_point" (default) or "random"

sam:
  weights: models/sam2.1_hiera_large.pt   # path relative to repo root
  cfg: configs/sam2.1/sam2.1_hiera_l.yaml # path relative to the sam2 package root
  device: cuda
  ladder:                              # granularity ladder; rungs run in order
    - name: coarse
      points_per_side: 32
      pred_iou_thresh: 0.80
      stability_score_thresh: 0.95
      crop_n_layers: 0
    - name: fine
      points_per_side: 64
      pred_iou_thresh: 0.70
      stability_score_thresh: 0.90
      crop_n_layers: 0
    - name: finest
      points_per_side: 64
      pred_iou_thresh: 0.60
      stability_score_thresh: 0.85
      crop_n_layers: 1
      box_nms_thresh: 0.9

matcher:
  positive_iou: 0.75                   # candidate IoU >= this → positive training label
  negative_iou: 0.25                   # candidate IoU <= this → negative; band is ignored
  accept_threshold: 0.60               # tuned by `fit`, written back to the model manifest
  embedding_model: facebook/dinov2-base

polygon:
  fidelity_iou: 0.98                   # min re-rasterised IoU to accept a simplification
  max_points: null                     # hard vertex cap; null = fidelity decides
  morph_kernel: 3                      # pre-smoothing kernel size; 0 = disabled
  min_ring_area_frac: 0.01             # drop polygon rings < 1% of total mask area
```

Secrets live only in `.env`:

```
ROBOFLOW_API_KEY=your_key_here
```

---

## Known limitations (v1)

- **One instance per frame.** `label` takes the highest-scoring candidate per image.
  Multi-instance support is a later addition that does not require a code rewrite.
- **Holes are filled.** `polygonize` uses `RETR_EXTERNAL`, so donut-shaped masks lose their
  holes. This is acceptable for weak labels that go into human review.
- **`/search` is not required.** Image enumeration uses the Roboflow annotation-batch admin
  API (`annotation-batches/*/images`), which works even when `/search` returns 404. If your
  Roboflow plan later gains `/search`, the pipeline continues to work without change.
- **Python 3.12 only.** `sam-2` has no cp313 wheel. The venv must use Python 3.12, installed
  via `uv` if 3.12 is not the system default.
