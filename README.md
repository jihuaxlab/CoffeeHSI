# CoffeeHSI

Classification of Green Coffee Beans with Machine Learning-Assisted Visible/Near-infrared Hyperspectral Imaging

## Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy==1.16.5
matplotlib==3.1.0
sklearn==0.21.3
seaborn==0.11.2
xgboost==1.5.0
lightgbm==3.2.1
```

## Usage

```
python main.py
```

## Dataset

- `x.txt` includes 3900 spectra from 13 classes of coffee beans with different flavor quality, and the details are in our [paper](https://doi.org/10.1016/j.foodcont.2025.111310).
- `place-x.txt` includes 35200 spectra of coffee beans from 11 places.

Each spectrum is ranging from 400 to 1000 nm with 224 dimensions. The label of `place-x.txt` is `place-y.txt`. The label of `x.txt` is directly written in `main.py`.

## Citation

```
@article{wu2025coffee,
  title = {Predicting the flavor potential of green coffee beans with machine learning-assisted visible/near-infrared hyperspectral imaging (Vis-NIR HSI): Batch effect removal and few-shot learning framework},
  journal = {Food Control},
  volume = {175},
  pages = {111310},
  year = {2025},
  author = {Minping Wu and Zhuangwei Shi and Haiyu Zhang and Rui Wang and Jiayi Chu and Shao Quan Liu and Heming Zhang and Hai Bi and Weihua Huang and Rui Zhou and Chenhui Wang}
}
```
