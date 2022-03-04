###############################
#                             
#     POSTPROCESS PREDICTIONS
#                             
###############################

import numpy as np

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