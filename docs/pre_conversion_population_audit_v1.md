# Doosan pre-conversion population audit v1

Run this **before** raw episodes are converted to processed/LeRobot datasets.  It
is read-only and performs the population checks that were used to freeze the
thesis dataset after Patch 11B.

## What is audited

For every raw episode, the tool builds the action-bearing synchronized rows in
memory and checks:

1. all four orientation representations (`rotvec_principal`,
   `rotvec_continuous`, `quaternion`, `rotation6d`) in both `full` and
   `no_wrench` layouts;
2. exact 7D action invariance, non-orientation invariance, physical rotation
   reconstruction, and final-six wrench placement;
3. numerical joint discontinuities and the strong `+pi <-> -pi` wrap signature;
4. raw SCHUNK held->released behavior, SpaceMouse RIGHT rising-edge provenance,
   and the release-only processed gripper action;
5. baseline-relative force/torque behavior up to the release action.

Principal-rotvec and stateless canonical quaternion branch/sign jumps are
reported as diagnostics.  Continuous rotvec and rotation6D large jumps are
structural review conditions.

## CPU parallelism

`--workers auto` is the default.  It uses **up to four available physical CPU
cores**.  This conservative cap avoids excessive concurrent MCAP
read/deserialization streams while keeping episode-level multiprocessing simple.
On Linux the audit respects CPU affinity and uses sysfs topology to collapse
SMT/hyper-thread siblings. An explicit `--workers N` may request more parallelism
when the storage/RAM subsystem has been validated for it, but values greater than
the available physical-core count are rejected.

Each worker owns one episode at a time and worker processes use the `spawn`
start method.  The shell launcher also pins common BLAS thread pools to one
thread per worker to avoid nested oversubscription.

## Runtime

MCAP decoding must run in the ROS 2 Jazzy Python environment.  The launcher
uses `/usr/bin/python3` by default and sources `/opt/ros/jazzy/setup.bash`.
It does **not** trust a workspace merely because `install/setup.bash` exists:
each candidate is probed in an isolated subshell against every ROS message type
in `DOOSAN_RAW_V1_TOPICS` (including `dsr_msgs2` and `gripper_msgs`).  Rejected
overlays therefore cannot pollute the final audit environment.

When `$ROS_WORKSPACE_SETUP` is supplied, that exact setup file must pass the
message-contract probe or the launcher fails closed.  Otherwise the launcher
probes these known thesis workspaces and selects the first one that actually
satisfies the raw-MCAP contract:

- `~/robotics_thesis/lab_myros2_ws/install/setup.bash` (lab workstation)
- `~/robotics_thesis/lab_myros2_git_clone/install/setup.bash` (laptop clone)

`ROS_WORKSPACE_CANDIDATES` may be set to a colon-separated list of alternative
setup files.  If no overlay is required because all message packages are in the
base ROS environment, the base environment is accepted only after the same
contract probe passes.

Example:

```bash
./scripts/run_pre_conversion_audit.sh \
  /media/ktt_rc/External_SSD_1TB/ktt_rc_robotics_thesis/dataset/cranfield_assembly \
  /home/ktt_rc/robotics_thesis/forcevla_lab/logs/pre_conversion_audit \
  --workers auto
```

Use `--expected-episodes N` when a fixed collection size is expected.

## Outputs

The output directory contains:

- `pre_conversion_audit.json`: complete machine-readable evidence;
- `summary.txt`: compact population gates/counts;
- `episode_metrics.csv`: one row per successful episode;
- `structural_review.txt`: episodes with state/action/joint/gripper structural
  concerns;
- `critical_force_review.txt`: short high-priority human review list;
- `broad_force_review.txt`: deliberately sensitive force/torque review list;
- `high_then_recovered.txt`: episodes where meaningful force was followed by a
  clear low-force recovery.

Force screening does **not** automatically discard episodes.

## Force review tiers

The broad tier preserves the original sensitive screen:

- persistent baseline-relative force, or
- peak baseline-relative force >= 25 N, or
- peak baseline-relative torque >= 5 Nm.

The critical tier is intentionally much smaller and is meant for manual
re-review of the genuinely suspicious demonstrations:

- persistent force with no terminal recovery and median force in the final
  0.5 s >= 30 N (`HIGH_FORCE_AT_RELEASE`), or
- extreme force with no terminal recovery sustained for >= 3 s
  (`SUSTAINED_EXTREME_FORCE`), or
- baseline-relative torque >= 5 Nm (`EXTREME_TORQUE`).

All thresholds are CLI-configurable.  Raw recorded wrench remains untouched;
these diagnostics subtract an episode-local initial median baseline only for
screening.
