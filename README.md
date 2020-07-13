# Data Mining Cup 2020

Files and codes with HU Team solution to the [Data Mining Cup 2020 competition](https://www.data-mining-cup.com).


## Project summary

Forecasting demand is an important managerial task that helps to optimize inventory planning. The optimized stocks can reduce retailer's costs and increase customer satisfaction due to faster delivery time. This project uses historical purchase data to predict future demand for different products.


## Project structure

The project has the follwoing structure:
- `codes/`: jupyter notebooks with codes for different project stages: 
    - data preparation
    - feature enginering
    - modeling
    - meta-parameter tuning
    - ensembling (blending and stacking)
    - helper functions
- `data/`: input data. The folder is not uploaded to Github due to size constraints. The raw data can be downloaded [here](https://www.data-mining-cup.com/dmc-2020/).
- `documentation/`: task documentation provided by the competition organizers
- `oof_preds/`: out-of-fold predictions produced by the train models within cross-validation.
- `submissions/`: test sample predictions produced by the trained models.


## Requirements

You can create a new conda environment using:

```
conda create -n dmc python=3.7
conda activate dmc
```

and then install the requirmenets:

```
conda install -n dmc --yes --file requirements.txt
pip install lightgbm
pip install imblearn
pip install catboost
```

Alternatively, you can install the packages in your base environment:

```
conda install --yes --file requirements.txt
pip install lightgbm
pip install imblearn
pip install catboost
```
