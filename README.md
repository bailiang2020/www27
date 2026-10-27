# HMDR

Decoupling and Regulating Modality Heads for Fake News Detection under Missing Modality.

## Setup

Python 3.10 or 3.11, PyTorch and a CUDA device.

```bash
pip install -r requirements.txt
```

CUDA kernels can be installed with `pip install --no-build-isolation -r requirements-cuda.txt`.

Set the Qwen3.5-4B checkpoint path and dataset paths in `configs/finefake.json` or `configs/mdsm_nyt.json`. Dataset fields are listed in [data/README.md](data/README.md).
The supplied configurations use fixed dataset splits and training reference pairs.

## Training and evaluation

```bash
python calibrate.py --config configs/finefake.json --device cuda:0
python train.py --config configs/finefake.json --output outputs/finefake --device cuda:0
python evaluate.py --checkpoint outputs/finefake/best_adapter --output outputs/finefake/test --device cuda:0
```

Evaluation includes complete image-text, text-only and image-only inputs. The same validation-selected checkpoint is used for all views.

## Files

- `hmdr/attention.py`: modality head decomposition and regulation.
- `hmdr/model.py`: HMDR prediction and modality invariance learning.
- `hmdr/data.py`: input processing, modality masks and reference pairing.
- `configs/`: FineFake and MDSM configurations.
