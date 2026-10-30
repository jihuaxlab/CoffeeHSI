# CoffeeHSI

Prediction of Green Coffee Beans with Machine Learning-Assisted Visible/Near-infrared Hyperspectral Imaging

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

### Quality prediction

`x.txt` includes 3900 spectra from 13 classes of coffee beans with different flavor quality, each class includes 300 spectra. 5 classes are with the Q-grade score >=80, defined as "Pass", and named from P1 to P5. Other 8 classes are with the Q-grade score <80, defined as "fail", and named from F1 to F8. The Q-grade scores of all 13 classes are listed correspondingly as below.
```
labels = ['F1','F2','F3','F4','F5','F6','F7','F8','P1','P2','P3','P4','P5']
score = [79,79,79,78,78,78,77,77,83,82,81,81,80]
```
There are four prediction tasks provided in this repo.
- Binary classification between pass samples and fail samples.
- Multi classification among all 13 classes.
- Multi classification for Q-grade, i.e. 7 classes (77,78,...,83).
- Regression prediction for Q-grade.

The details of binary classification are provided in our [paper](https://doi.org/10.1016/j.foodcont.2025.111310).

### Product origin prediction

`place-x.txt` includes 35200 spectra of coffee beans from 11 places, including China, Columbia, CostaRica, Ethiopia, Guatemala, Indonesia, Kenya, Mexico, Peru, Rwanda, and Uganda. There is a high-fidelity classification task to discriminate all 11 places, and a low-fidelity classification to discriminate different continents (Africa, America and Asia).

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
