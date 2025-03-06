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

- `x.txt` includes 3900 spectra from 13 classes of coffee beans with different flavor quality, and the details are in our paper.
- `place-x.txt` includes 35200 spectra of coffee beans from 11 places.

Each spectrum is ranging from 400 to 1000 nm with 224 dimensions. The label of `place-x.txt` is `place-y.txt`. The label of `x.txt` is directly written in `main.py`.