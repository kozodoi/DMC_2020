# Data Mining Cup 2020 Solution

Top-15 solution to the [Data Mining Cup 2020](https://www.data-mining-cup.com) competition on profit-driven demand forecasting. 

![pipeline](https://i.postimg.cc/PfvKjkJt/fig-part.png)


## Summary

Forecasting demand is an important managerial task that helps to optimize inventory planning. The optimized stocks can reduce retailer's costs and increase customer satisfaction due to faster delivery time. This project uses historical purchase data to predict future demand for different products.

To approach this task, we perform a thorough feature engineering and data aggregation, implement custom profit-driven loss functions and build an ensemble of LightGBM classification models. A detailed walkthrough of our solution covering the most important steps is provided in [this blog post](https://kozodoi.me/python/time%20series/demand%20forecasting/competitions/2020/07/27/demand-forecasting.html).


## Project structure

The project has the following structure:
- `codes/`: notebooks with codes for different project stages:
    - data preparation
    - feature engineering
    - modeling
    - meta-parameter tuning
    - ensembling (blending and stacking)
    - helper functions
- `data/`: input data. The folder is not uploaded to Github due to size constraints. The raw data can be downloaded [here](https://www.data-mining-cup.com/dmc-2020/).
- `output/`: output files and plots exported from the notebooks.
- `documentation/`: task documentation provided by the competition organizers
- `oof_preds/`: out-of-fold predictions produced by the train models within cross-validation.
- `submissions/`: test sample predictions produced by the trained models.


## Requirements

You can create a new conda environment using:

```
conda create -n dmc python=3.7
conda activate dmc
```

and then install the requirements:

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
