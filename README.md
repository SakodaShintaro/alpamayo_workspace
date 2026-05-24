# alpamayo_workspace

このリポジトリは以下のリポジトリをsubmoduleとして含み、CARLA 0.9.16を使ってBench2Driveを動かします。

- <https://github.com/NVlabs/alpamayo>
- <https://github.com/NVlabs/alpamayo1.5>
- <https://github.com/SakodaShintaro/Bench2Drive>

## 事前設定

2階層上にCARLA_0.9.16が展開されていること（Additional Mapの展開済みを前提とする）

```bash
 ls ../../CARLA_0.9.16
CHANGELOG  CarlaUE4  CarlaUE4.sh  Co-Simulation  Dockerfile  Engine  HDMaps  Import  ImportAssets.sh  LICENSE  Manifest_DebugFiles_Linux.txt  PythonAPI  README  Tools  VERSION
```

## Setup

以下のいずれかでsubmoduleを展開する。

- (A) cloneと同時に展開

```bash
git clone --recurse-submodules https://github.com/SakodaShintaro/alpamayo_workspace
```

- (B) cloneと同時に展開

```bash
git clone https://github.com/SakodaShintaro/alpamayo_workspace
cd alpamayo_workspace
git submodule update --init --recursive
```

まず推論を実行する。

```bash
uv sync
uv run python alpamayo/src/alpamayo_r1/test_inference.py
uv run python alpamayo1.5/src/alpamayo1_5/test_inference.py
```

両スクリプトともサンプルデータ(数百MB)+ モデルウェイト(各22GB)を初回に自動DLする。

Bench2Driveの1シナリオを実行する。

```bash
./scripts/run_b2d_alpamayo15.sh
```

結果は以下の動画などとして出る。

```bash
ls ./results/b2d_alpamayo15_20260524_190913/spectator.mp4
```
