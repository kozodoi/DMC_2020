# Profit-Driven Demand Forecatsing

Top-15 solution to the [Data Mining Cup 2020](https://www.data-mining-cup.com) competition on profit-driven demand forecasting.

![pipeline](https://kozodoi.me/images/copied_from_nb/images/fig_partitioning.png)

- [Summary](#summary)
- [Repo structure](#repo-structure)
- [Working with the repo](#working-with-the-repo)


## Summary

Forecasting demand is an important managerial task that helps to optimize inventory planning. Optimized stocks can reduce retailer's costs and increase the customer satisfaction due to faster delivery times. This project uses historical purchase data to predict future demand for different products.

To approach this task, we perform a thorough feature engineering and data aggregation, implement custom profit-driven loss functions and build an ensemble of LightGBM classification models. A detailed walkthrough of the project covering the most important steps is provided in [this blog post](https://kozodoi.me/blog/20200727/demand-forecasting).


## Repo structure

The project has the following structure:
- `codes/`: Python codes with functions used in Jupyter notebooks
- `notebooks/`: Jupyter notebooks covering key project stages:
    - data preparation
    - feature engineering
    - predictive modeling
    - meta-parameter tuning
    - ensembling (blending and stacking)
- `data/`: input data (not included due to size constraints, can be downloaded [here](https://www.data-mining-cup.com/dmc-2020/))
- `documentation/`: task documentation provided by the competition organizers
- `output/`: output files and plots exported from the notebooks
- `oof_preds/`: out-of-fold predictions produced by the models within cross-validation
- `submissions/`: test sample predictions produced by the trained models


## Working with the repo

To work with the repo, I recommend to create a virtual Conda environment:

```
conda create -n dmc python=3.7
conda activate dmc
```

and then install the requirements specified in `requirements.txt`:

```
conda install -n dmc --yes --file requirements.txt
pip install lightgbm
pip install imblearn
pip install catboost
```

More details are provided in the documentation within the scripts & notebooks. A detailed walkthrough is available in [this blog post](https://kozodoi.me/python/time%20series/demand%20forecasting/competitions/2020/07/27/demand-forecasting.html).
