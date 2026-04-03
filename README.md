# ml-clara-implementation

Lightweight wrapper repo for local experiments with CLaRa.

## Quick Start

1. Open [setup_repo.ipynb](setup_repo.ipynb).
2. Run all cells to clone:
   - `ml-clara` codebase
   - `2WikiMultihopQA` benchmark
   - `CLaRa-7B-Instruct` pretrained model, including 2 compression checkpoints: compression-16 and compression-128

## Included

- [setup_repo.ipynb](setup_repo.ipynb): one-click workspace setup via `git clone`
- [inference_2wiki.ipynb](inference_2wiki.ipynb): inference example
- [evaluate_2wiki.ipynb](evaluate_2wiki.ipynb): evaluation example

## Environment Snapshot

- `requirements.txt` in this repo is generated with:
   - `pip freeze > requirements.txt`
- Recreate the same Python package set with:
   - `pip install -r requirements.txt`