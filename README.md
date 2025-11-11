# VITA: Variational Pretraining of Transformers for Climate-Robust Crop Yield Forecasting

Official implementation of VITA, a variational pretraining framework that learns weather representations from rich satellite data and transfers them to yield prediction tasks with limited ground-based measurements.

**Paper**: [arXiv:2508.03589](https://arxiv.org/abs/2508.03589) | **AAAI 2026**

## Overview

VITA addresses the data asymmetry problem in agricultural AI: pretraining uses 31 meteorological variables from NASA POWER satellite data, while deployment relies on only 6 basic weather features. Through variational pretraining with a seasonality-aware sinusoidal prior, VITA achieves state-of-the-art performance in predicting corn and soybean yields across 763 U.S. Corn Belt counties, particularly during extreme weather years.

## Usage

### Data Download

```bash
python -m src.downloaders.nasa_power_dataset
python -m src.downloaders.khaki_corn_belt_dataset
```

### Pretraining

```bash
python -m src.pretraining.main --batch-size 256 --n-epochs 100 --model-size small --alpha 0.5 --data-dir data/
```

### Crop Yield Prediction

```bash
python -m src.crop_yield.main --batch-size 64 --n-epochs 40 --model-size small --beta 1e-4 --crop-type soybean --pretrained-model-path path/to/pretrained_model.pt
```

## Citation

```bibtex
@inproceedings{hasan2026vita,
      title={VITA: Variational Pretraining of Transformers for Climate-Robust Crop Yield Forecasting},
      author={Hasan, Adib and Roozbehani, Mardavij and Dahleh, Munther},
      booktitle={Proceedings of the 40th AAAI Conference on Artificial Intelligence},
      year={2026},
      url={https://arxiv.org/abs/2508.03589},
}
```
