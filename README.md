# introml-clara-implementation

Course project repo for an Intro to ML class: research + hands-on implementation of CLaRa (Continuous Latent Reasoning) and small-scale experiments on Kaggle T4.

- Reimplementation (Stage I/II training) and dataset loaders.
- Apple-native evaluation and fine-tuning using their original `modeling_clara.py` (with modifications for quantization and VRAM optimization) with a Kaggle-ready workflow.
- Kaggle notebooks for reproduction (train from scratch, fine-tune, evaluate).

## Kaggle notebooks

Open any notebook in the `notebook/` folder on Kaggle. The notebooks handle cloning, deps, and environment patching.

Main notebooks:

- `notebook/clara-ml-final-evaluation-apple.ipynb` — Apple-native eval on SQuAD + TriviaQA.
- `notebook/clara-ft-squad.ipynb` / `notebook/clara-ft-triviaqa.ipynb` — fine-tune on SQuAD or TriviaQA.
- `notebook/clara-eval-ft-squad.ipynb` / `notebook/clara-eval-ft-triviaqa.ipynb` — evaluate fine-tuned checkpoints.

## Checkpoints as Kaggle datasets

The notebooks expect model checkpoints to be attached as Kaggle datasets:

- `tokiggle/clara-7b-e2e-4q` (Apple E2E pretrained checkpoint)
- `tokiggle/clara-ft-squad` and `tokiggle/clara-ft-triviaqa` (fine-tuned checkpoints uploaded from this project)

## Code map (quick)

- `models/` — reimplementation of CLaRa core and helpers.
- `data/` — unified dataset loader.
- `configs/` — training config defaults.
- `scripts/` — Stage I/II training and Apple-native fine-tune/eval drivers.
- `modeling_clara.py` — patched Apple model code used by the Kaggle pipelines. The original lives inside the Apple checkpoint, which is not tracked in this repo's git.
