# Laptop Development Setup

This repository is designed so that the Doosan-to-ForceVLA dataset tools
can be developed on a laptop independently of the robot-control workspace.

## Requirements

The core Python package requires Python >=3.10.

The core package has no mandatory third-party runtime dependencies.

The optional `lerobot-v21` extra installs `pyarrow>=24.0.0`.

System ffmpeg/ffprobe and additional image/video packages may be required
for specific export paths.

## Clone

On the laptop:

    git clone https://github.com/Raunak-Chakraborty/doosan-forcevla-data-tools.git
    cd doosan-forcevla-data-tools

For an existing checkout:

    git status
    git fetch origin
    git pull --ff-only

## Bootstrap core development environment

Run:

    ./scripts/bootstrap_laptop_dev.sh

This creates:

    ~/.venvs/doosan-forcevla-data-tools

Activate it later with:

    source ~/.venvs/doosan-forcevla-data-tools/bin/activate

A different location can be selected with:

    export DOOSAN_FORCEVLA_DATA_VENV=/desired/path

## LeRobot-v2.1 development environment

To also install the optional PyArrow dependency:

    ./scripts/bootstrap_laptop_dev.sh lerobot-v21

## Tests

Use the repository's portable test runner:

    ./scripts/run_laptop_tests.sh

One dependency-doctor test intentionally verifies that optional runtime
libraries such as OpenCV were not imported by the doctor itself.

Because that property depends on interpreter state, the portable runner
executes that test in a fresh Python process and executes all remaining
tests separately.

This preserves the intent of the test without making the result depend on
which unrelated test happened to import OpenCV earlier in the same Python
process.

## Laptop versus lab workstation

Laptop development is appropriate for source editing, schema work,
synchronization logic, validators, tests, processed-data inspection,
non-ROS conversion work, and LeRobot-v2.1 development when its optional
dependencies are installed.

Production raw Doosan MCAP decoding depends on ROS 2 Jazzy and the Doosan /
SCHUNK message definitions used by the real recordings.

The production lab conversion path should therefore remain in the validated
ROS environment unless that same ROS/message environment is deliberately
reproduced on the laptop.

## ForceVLA compatibility

The separate ForceVLA thesis repository contains the validated ForceVLA
software/environment reference.

Validated ForceVLA reproducibility commit:

    ef5ba4f5522d241fe72b559d2ccc331405f56b67

Large datasets, model weights, Hugging Face caches, and training checkpoints
are intentionally not stored in this repository.

## Git workflow on two computers

Before starting work on either machine:

    git status
    git fetch origin
    git pull --ff-only

After validating changes:

    git add <files>
    git diff --cached --check
    git commit -m "<description>"
    git push origin main

Before continuing on the other machine:

    git status
    git pull --ff-only

Do not maintain conflicting uncommitted versions of the same change on both
machines.
