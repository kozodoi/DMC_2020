import pandas as pd
import numpy as np

def profit(y_true, y_pred, price):
    '''
    Computes profit according to DMC 2020 task.
    
    Arguments:
    - y_true (numpy array or list): ground truth (correct) target values.
    - y_pred (numpy array or list): estimated target values.
    - price (numpy array or list): item prices.

    Returns:
    - profit value
    
    Examples:
    
    profit(y_true = np.array([5, 5, 5]),
           y_pred = np.array([0, 0, 0]),
           price  = np.array([1, 1, 1]))

    '''

    # round preds
    y_pred_round = np.round(y_pred)

    # remove negative preds
    y_pred_round[y_pred_round < 0] = 0


    # sold units
    units_sold = np.minimum(y_true, y_pred_round)

    # overstocked units
    units_overstock = y_pred_round - y_true
    units_overstock[units_overstock < 0] = 0

    # profit
    revenue = units_sold * price
    fee     = units_overstock * price * 0.6
    profit  = revenue - fee
    profit  = profit.sum()
    
    # return values
    return profit