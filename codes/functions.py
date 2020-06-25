import pandas as pd
import numpy as np


##### TRAINING LOSS
def asymmetric_mse(y_true, y_pred):
    '''
    Asymmetric MSE objective for training Lightgbm regressor.
    
    The profit function is non-differentiable => it can only be used for evaluation.
    The asymmetric MSE can be used as a training loss to approximate profit:
     - overpredicting demand by one unit decreases profit by 0.6p
     - underpredicting demand by one unit decreases profit by p
     - hence, overpredicting is 0.6 times less costly
     
    Arguments:
    - y_true (numpy array or list): ground truth (correct) target values.
    - y_pred (numpy array or list): estimated target values.
    
    Returns:
    - gradient matrix
    - hessian matrix
    '''
    
    # asymmetry parameter
    fee_mult = 0.6
    
    # computations
    residual = (y_true - y_pred).astype('float')    
    grad = np.where(residual > 0, -2*residual*fee_mult, -2*residual)
    hess = np.where(residual > 0,  2*fee_mult, 2.0)
    
    # return values
    return grad, hess



##### PROFIT FUNCTION
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

    # remove negative and round
    y_pred = np.where(y_pred > 0, y_pred, 0)
    y_pred = np.round(y_pred).astype('int')

    # sold units
    units_sold = np.minimum(y_true, y_pred)

    # overstocked units
    units_overstock = y_pred - y_true
    units_overstock[units_overstock < 0] = 0

    # profit
    revenue = units_sold * price
    fee     = units_overstock * price * 0.6
    profit  = revenue - fee
    profit  = profit.sum()
    
    # return values
    return profit



##### POSTPROCESSING PREDICTIONS
def postprocess_preds(y_pred):
    '''
    Processess demand predictions outputted by a model.
    
    Arguments:
    - y_pred (numpy array or list): estimated target values.

    Returns:
    - corrected y_pred
    
    Examples:
    
    postprocess_preds(y_pred = np.array([-2.10, 1.15, 10.78]))
    '''

    # demand can not be negative
    y_pred = np.where(y_pred > 0, y_pred, 0)
    
    # demand has to be integer
    y_pred = np.round(y_pred).astype('int')

    # return values
    return y_pred