# overpredicting demand by one unit decreases profit by 0.6p
# underpredicting demand by one unit decreases profit by p
# hence, underpredicting is 1/0.6 = 1.67 times more costly

# training loss
def asymmetric_mse_train(y_true, y_pred):
    underpredict_mult = 1/0.6
    residual = (y_true - y_pred).astype("float")
    grad     = np.where(residual > 0, -2*residual*underpredict_mult, -2*residual)
    hess     = np.where(residual > 0, 2*underpredict_mult, 2.0)
    return grad, hess

# validation loss
def asymmetric_mse_eval(y_true, y_pred):
    underpredict_mult = 1/0.6
    residual = (y_true - y_pred).astype("float")
    loss     = np.where(residual > 0, (residual**2)*underpredict_mult, residual**2) 
    return "asymmetric_mse", np.mean(loss), False
