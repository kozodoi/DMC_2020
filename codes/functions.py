import pandas as pd
import numpy as np

# profit function is non-differentiable => it can only be used for evaluation
# asymmetric MSE can be used as a training loss to approximate profit:
# - verpredicting demand by one unit decreases profit by 0.6p
# - underpredicting demand by one unit decreases profit by p
# - hence, underpredicting is 1 / 0.6 = 1.67 times more costly

# training loss
def asymmetric_mse_train(y_true, y_pred):
    underpredict_mult = 1 / 0.6
    residual = (y_true - y_pred).astype('float')
    grad = np.where(residual > 0, -2*residual*underpredict_mult, -2*residual)
    hess = np.where(residual > 0, 2*underpredict_mult, 2.0)
    return grad, hess

# validation loss
def asymmetric_mse_eval(y_true, y_pred):
    underpredict_mult = 1 / 0.6
    residual = (y_true - y_pred).astype('float')
    loss = np.where(residual > 0, (residual**2)*underpredict_mult, residual**2) 
    return 'asymmetric_mse', np.mean(loss), False

# profit function
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