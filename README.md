# VITA: Variational Pretraining of Transformers for Climate-Robust Crop Yield Forecasting

Official implementation of VITA, a variational pretraining framework that learns weather representations from rich satellite data and transfers them to yield prediction tasks with limited ground-based measurements.

[[arXiv:2508.03589]](https://arxiv.org/abs/2508.03589) [[Pretrained Model]]( https://huggingface.co/notadib/VITA) [[AAAI-26]](https://aaai.org/conference/aaai/aaai-26/)

## Overview

VITA addresses the data asymmetry problem in agricultural AI: pretraining uses 31 meteorological variables from NASA POWER satellite data, while deployment relies on only 6 basic weather features. Through variational pretraining with a seasonality-aware sinusoidal prior, VITA achieves state-of-the-art performance in predicting corn and soybean yields across 763 U.S. Corn Belt counties, particularly during extreme weather years.

## Usage

### Data Download

🛰️ Pretraining dataset: [NASA POWER Daily Weather](https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather)

🌽 Crop yield dataset: [USA Corn Belt Crop Yield](https://huggingface.co/datasets/notadib/usa-corn-belt-crop-yield)

```bash
pip install -r requirements.txt

python -m src.downloaders.nasa_power_dataset
python -m src.downloaders.khaki_corn_belt_dataset
```

### Pretraining

```bash
python -m src.pretraining.main --batch-size 256 --n-epochs 100 --model-size small --alpha 0.5 --data-dir data/
```

**Pretrained model weights:** https://huggingface.co/notadib/VITA

### Crop Yield Prediction

**Note:** This is an example run. For full hyperparameter configurations that reproduce paper results, see the paper's appendix. Due to non-determinism from hardware differences (GPU type, cuDNN versions) and stochastic training, you may observe small numerical variations from the exact values reported in the paper, though performance should remain in the same ballpark.

```bash
python -m src.crop_yield.main --batch-size 16 --n-epochs 40 --model-size small --beta 1e-4 --init-lr 2.5e-4 --test-type extreme --crop-type soybean --pretrained-model-path path/to/pretrained_model.pth
```

## Citation

```bibtex
@inproceedings{hasan2026vita,
      title={VITA: Variational Pretraining of Transformers for Climate-Robust Crop Yield Forecasting},
      author={Adib Hasan and Mardavij Roozbehani and Munther Dahleh},
      booktitle={Proceedings of the 40th AAAI Conference on Artificial Intelligence},
      year={2026},
      url={https://arxiv.org/abs/2508.03589},
}
```
