# alpamayo_workspace

This repository bundles the following submodules and runs Bench2Drive on CARLA 0.9.16.

- <https://github.com/NVlabs/alpamayo>
- <https://github.com/NVlabs/alpamayo1.5>
- <https://github.com/SakodaShintaro/Bench2Drive>

## Prerequisites

CARLA_0.9.16 must be extracted two directories above this repo (with Additional Maps already imported).

```bash
 ls ../../CARLA_0.9.16
CHANGELOG  CarlaUE4  CarlaUE4.sh  Co-Simulation  Dockerfile  Engine  HDMaps  Import  ImportAssets.sh  LICENSE  Manifest_DebugFiles_Linux.txt  PythonAPI  README  Tools  VERSION
```

## Setup

Pull submodules with either of the following:

- (A) Clone with submodules in one shot

```bash
git clone --recurse-submodules https://github.com/SakodaShintaro/alpamayo_workspace
```

- (B) Clone first, then pull submodules

```bash
git clone https://github.com/SakodaShintaro/alpamayo_workspace
cd alpamayo_workspace
git submodule update --init --recursive
```

First, run the standalone inference smoke tests.

```bash
uv sync
uv run python alpamayo/src/alpamayo_r1/test_inference.py
uv run python alpamayo1.5/src/alpamayo1_5/test_inference.py
```

Both scripts auto-download the sample data (a few hundred MB) and the model weights (~22 GB each) on first run.

Run a single Bench2Drive scenario.

```bash
./scripts/run_b2d_alpamayo15.sh
```

The result is written out as a spectator video, among other artifacts.

```bash
ls ./results/b2d_alpamayo15_20260524_190913/spectator.mp4
```

Run 220 scenarios.

```bash
./scripts/run_b2d_alpamayo15.sh ./Bench2Drive/leaderboard/data/bench2drive220.xml
```
