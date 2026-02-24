# High-Performance Dual-Arm Task and Motion Planning for Tabletop Rearrangement [![Paper](https://img.shields.io/badge/SDAR-Arxiv-blue)](https://arxiv.org/pdf/2512.08206) [![Video](https://img.shields.io/badge/Video-TBD-red)](#citation)

<a href="https://duozhangrobotics.github.io/" target="_blank"><nobr>Duo Zhang</nobr></a> &emsp;
<a href="https://duozhangrobotics.github.io/" target="_blank"><nobr>Junshan Huang</nobr></a> &emsp;
<a href="https://arc-l.github.io/group.html" target="_blank"><nobr>Jingjin Yu</nobr></a> &emsp; (Accepted to [ICRA 2026](https://2026.ieee-icra.org/))


## Table of Contents
- [Introduction](#intro)
  - [What This Repository Does](#what)
  - [Methods and Variants](#methods)
  - [Dataset / Task Examples](#examples)
- [Installation](#install)
- [Usage](#usage)
  - [Run a Single Case](#single-case)
  - [Run All Cases](#all-cases)
  - [Evaluate Task Planner Only](#task-planner-only)
  - [Visualize Aggregate Results](#visualize)
- [Task JSON Format](#json-format)
- [Repository Structure](#repo-structure)
- [Citation](#citation)

## <a name="intro"></a>Introduction
### <a name="what"></a>What this repository does
This repository implements a dual-arm task and motion planning (TAMP) pipeline for object rearrangement on a tabletop. It includes:

- A dual-arm planner (`dual_arms.py`) using CuRobo-based motion planning.
- A baseline rearrangement pipeline (`baseline_dual_arm.py` + `baseline/`).
- Benchmark scripts for evaluating multiple planner combinations on generated task instances.
- Utilities for generating task instances and visualizing results.

The codebase appears to compare task-planning and motion-planning combinations (e.g., "ours", baseline, and hybrid variants) across synthetic rearrangement datasets such as `random`, `cycled`, `double_cycled`, and `mixed`.

### <a name="methods"></a>Methods and variants
`evaluation.py` currently supports the following methods:

- `ours`
- `baseline`
- `baseline_with_our_tp`
- `ours_with_baseline_tp`
- `our_tp_with_curobo`
- `baseline_tp_with_curobo`

These names correspond to different combinations of task planning (TP) and motion planning (MP) components. If you are preparing a paper release, this is a good place to add a short mapping from method names to algorithm descriptions.

## <a name="install"></a>Installation

### 1. Create the conda environment
```bash
conda env create -f environment.yml
conda activate dual_arm_tamp
```

### 2. Install CuRobo from the following repo 

Please use the CuRobo repo provided by the author for this project:

https://github.com/DuoZhangRobotics/curobo_dual_ur5e

Reason: the official CuRobo repo had bugs when attaching objects in a multi-arm system.  
Reference issue: https://github.com/NVlabs/curobo/issues/553

## <a name="usage"></a>Usage

### <a name="single-case"></a>Run a single case
Run one method on one task JSON file:
```bash
python evaluation.py --input data/cycled_objects_4.json --method ours
```

Example methods:

```bash
python evaluation.py --input data/cycled_objects_4.json --method our_tp_with_curobo
```

### <a name="task-planner-only"></a>Evaluate task planner only (action count)
To compare only task-planner outputs (number of actions / action lists), use:

```bash
python evaluate_task_planner.py --input data/cycled_objects_4.json --method ours
```

## <a name="json-format"></a>Task JSON Format
Tasks are loaded using `Task.read_from_json(...)` in `task.py`.

Required fields:
- `starts`: list of object poses `[x, y, z]`
- `goals`: list of object poses `[x, y, z]`
- `object_size`: cube side length (meters)
- `map_width`: workspace width (meters)
- `map_length`: workspace length (meters)

Example:

```json
{
  "starts": [
    [-0.11826439433444093, -0.11124986556905428, 0.0225],
    [0.13192161251950177, 0.157525877115014, 0.0225]
  ],
  "goals": [
    [0.13192161251950177, 0.157525877115014, 0.0225],
    [-0.11826439433444093, -0.11124986556905428, 0.0225]
  ],
  "object_size": 0.045,
  "map_width": 0.6,
  "map_length": 0.6
}
```

## <a name="citation"></a>Citation

If you use this codebase, please cite:
```bibtex
@inproceedings{ZhaHuaYu26ICRA,
author = {Duo Zhang and Junshan Huang and Jingjin Yu},
title = {High-Performance Dual-Arm Task and Motion Planning for Tabletop Rearrangement},
booktitle={IEEE International Conference on Robotics and Automation},
year={2026} }
```
