# Lab Workstation Validation Checklist

## 1. Purpose

This checklist validates the Doosan ForceVLA data-tools repository on the lab workstation inside the already validated ForceVLA environment. The goal is to confirm that the laptop-developed dataset pipeline, skeleton export, dependency checks, preflight reports, and dependency-optional real-export attempt behave correctly in the environment that will later run ForceVLA loader, normalization-statistics, and training checks.

This checklist does not start ROS recording, does not upload to Hugging Face, and does not integrate any4lerobot.

## 2. Path Variables

Set or adapt these variables on the lab workstation. The lab paths may differ from laptop paths.

```bash
export LAB_DATA_TOOLS_REPO="$HOME/robotics_thesis/doosan_forcevla_data_tools"
export LAB_FORCEVLA_REPO="$HOME/robotics_thesis/forcevla_lab/ForceVLA"
export FORCEVLA_ENV="<validated_forcevla_env_name>"
```

Reference laptop path:

```bash
export LAPTOP_REPO="/home/horus/robotics_thesis/doosan_forcevla_data_tools"
```

Before running commands, verify that `LAB_DATA_TOOLS_REPO`, `LAB_FORCEVLA_REPO`, and `FORCEVLA_ENV` match the actual lab workstation layout.

## 3. Transfer Options

### Option A: GitHub Pull, Preferred

Use this if a private GitHub repository is available.

Laptop steps:

```bash
cd "$LAPTOP_REPO"
git status --short
git remote -v
git push
```

Lab workstation first-time clone:

```bash
mkdir -p "$(dirname "$LAB_DATA_TOOLS_REPO")"
git clone <private_github_repo_url> "$LAB_DATA_TOOLS_REPO"
```

Lab workstation update after clone exists:

```bash
cd "$LAB_DATA_TOOLS_REPO"
git status --short
git pull --ff-only
```

### Option B: Zip Transfer

Use this if GitHub access is not convenient.

Laptop steps:

```bash
cd "$LAPTOP_REPO"
zip -r doosan_forcevla_data_tools.zip . -x ".git/*" "data/*" "**/__pycache__/*" "*.pyc" ".venv/*" "*.egg-info/*" ".pytest_cache/*"
```

Copy `doosan_forcevla_data_tools.zip` to the lab workstation by the available lab transfer method.

Lab workstation unzip:

```bash
mkdir -p "$LAB_DATA_TOOLS_REPO"
cd "$LAB_DATA_TOOLS_REPO"
unzip /path/to/doosan_forcevla_data_tools.zip
```

## 4. Lab Environment Activation

Use the validated ForceVLA environment. If the lab uses a different activation method, adapt these commands.

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$FORCEVLA_ENV"
```

Confirm the active Python belongs to the intended ForceVLA environment:

```bash
which python3
python3 --version
```

## 5. Basic Repo Sanity Commands

```bash
cd "$LAB_DATA_TOOLS_REPO"
pwd
git status --short
git log --oneline -n 5
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

If the repository was transferred by zip and has no `.git` directory, skip the `git status` and `git log` commands.

## 6. Dependency Check

Run the dependency check inside the activated lab ForceVLA environment:

```bash
cd "$LAB_DATA_TOOLS_REPO"
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.check_export_dependencies
```

Save or paste the full output. This is the dependency result that matters for ForceVLA compatibility decisions.

## 7. Full forcevla_13d Dummy Pipeline On Lab

Run the first/default compatibility target end to end.

```bash
cd "$LAB_DATA_TOOLS_REPO"
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.dummy.make_dummy_raw_episode \
  --output data/raw_dummy/episode_000000
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.raw_to_processed \
  --raw data/raw_dummy/episode_000000 \
  --output data/processed_dummy/episode_000000
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.plan_lerobot_export \
  --processed data/processed_dummy/episode_000000 \
  --profile forcevla_13d \
  --output data/processed_dummy/episode_000000/export_plan_forcevla_13d.json
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.stage_lerobot_export \
  --processed data/processed_dummy/episode_000000 \
  --export-plan data/processed_dummy/episode_000000/export_plan_forcevla_13d.json \
  --output data/staged_dummy/forcevla_13d/episode_000000
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_lerobot_skeleton \
  --staged data/staged_dummy/forcevla_13d/episode_000000 \
  --output data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 \
  --episode-index 0 \
  --task-index 0 \
  --profile forcevla_13d \
  --image-mode symlink \
  --overwrite
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_lerobot_skeleton \
  data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.preflight_real_export \
  data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 \
  --output data/lerobot_dummy/forcevla_13d/preflight_report_lab.json
```

## 8. Real-Export Attempt On Lab

Run the dependency-optional local real-export attempt. This writes a report in all cases and writes parquet/videos only when dependencies are available.

```bash
cd "$LAB_DATA_TOOLS_REPO"
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_real_lerobot_export \
  --skeleton data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 \
  --output data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 \
  --mode write-if-available
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_real_lerobot_export_attempt \
  data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
```

## 9. Outputs To Paste Back Into ChatGPT

Paste the complete output of these commands:

```bash
cd "$LAB_DATA_TOOLS_REPO"
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.check_export_dependencies
```

```bash
cat data/lerobot_dummy/forcevla_13d/preflight_report_lab.json
```

```bash
cat data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0/export_attempt_report.json
```

## 10. What To Check In Lab Results

Check these items in the pasted outputs:

- Is `pyarrow` available?
- Is `lerobot` available?
- Are `imageio`, `cv2`, `PIL`, and `ffmpeg` available?
- Is `parquet_ready` true or false?
- Is `video_ready` true or false?
- Is `real_export_ready` true or false?
- Is `parquet_written` true or false?
- Is `videos_written` true or false?
- Are there any schema errors?
- Are there any ForceVLA import conflicts?
- Does `prompt` equal `task` in the preflight report?
- Are terminal padding fields absent from exported records?

## 11. Next Decision After Lab Validation

Use the lab results to decide the next coding step.

- If `pyarrow` works and parquet is written, the next step is ForceVLA loader and normalization-statistics compatibility.
- If videos are written, inspect video paths and file sizes before trying loader integration.
- If `lerobot` is available, decide whether to use LeRobot APIs or continue direct `pyarrow` writing.
- If `pyarrow` is missing, decide whether to install `pyarrow` in the lab ForceVLA environment.
- If video dependencies are missing or encoding fails, decide whether to use `imageio`, `cv2`, or ForceVLA/LeRobot utilities.
- Do not start real robot recording until the dummy export and load path is validated.
- Do not add Hugging Face upload, ROS recording, or any4lerobot integration before local/lab dummy compatibility is confirmed.

## 12. Design Reminders

- `forcevla_13d` is the first/default compatibility target.
- `doosan_full_25d` is supported as secondary.
- The primary action label is measured TCP delta from robot state.
- `prompt` equals `task` for now.
- The terminal-padded final frame is excluded from export/training.
- Images are staged first via symlink/copy.
- The first video writer encodes from controlled `image_staging`, not arbitrary raw references.
- No fake third camera is generated in the dataset writer.
- Missing third camera handling belongs in the ForceVLA transform/config layer if needed.
