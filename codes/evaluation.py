import pandas as pd
import numpy as np



###############################
#                             
#        TRAINING LOSS
#                             
###############################

def asymmetric_mse(y_true, 
                   y_pred):
    '''
    Asymmetric MSE objective for training LightGBM regressor.
    
    Arguments:
    - y_true (numpy array or list): ground truth (correct) target values.
    - y_pred (numpy array or list): estimated target values.
    
    Returns:
    - gradient matrix
    - hessian matrix
    '''
    
    # checks
    assert len(y_true) == len(y_pred), 'Predictions and actuals are not the same length'
    
    # calculations
    residual = (y_true - y_pred).astype('float')    
    grad     = np.where(residual > 0, -2*residual, -0.72*residual)
    hess     = np.where(residual > 0,  2.0, 0.72)
    
    return grad, hess



###############################
#                             
#        VALIDATION LOSS
#                             
###############################

def asymmetric_mse_eval(y_true, 
                        y_pred):
    
    '''
    Asymmetric MSE evaluation metric for LightGBM regressor.
    
    Arguments:
    - y_true (numpy array or list): ground truth (correct) target values.
    - y_pred (numpy array or list): estimated target values.
    
    Returns:
    - name of the metric
    - value od the metric
    - whether the metric is maximized
    '''
    
    # checks
    assert len(y_true) == len(y_pred), 'Predictions and actuals are not the same length'
    
    # calculations
    residual = (y_true - y_pred).astype('float')      
    loss     = np.where(residual > 0, 2*residual**2, 0.72*residual**2)
    
    return 'asymmetric_mse_eval', np.mean(loss), False



###############################
#                             
#        PROFIT FUNCTION
#                             
###############################

def profit(y_true, 
           y_pred, 
           price):
    
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
    
    # checks
    assert len(y_true) == len(y_pred), 'Predictions and actuals are not the same length'
    assert len(price)  == len(y_pred), 'Predictions and prices are not the same length'
    
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