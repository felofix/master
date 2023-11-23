import sys
import numpy as np
sys.path.append('../pytorch_code')
import clamped_beam_pytorch as cbp
import optuna
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

fenics_true = np.loadtxt("../data/clamped_beam_fenics.txt", delimiter=',')

def objective(trial):
	lr = trial.suggest_float('lr', 1e-7, 1e-2)
	epochs = trial.suggest_int('epochs', 30000, 40000, step=100)
	network = cbp.solve_clamped_beam_pytorch(4, 32, epochs, lr)
	coor_after = cbp.predict(network)
	mse = mean_squared_error(coor_after, fenics_true)
	
	return mse

"""
study = optuna.create_study()
study.optimize(objective, n_trials=10)

# Finding best parameters and plotting.
best_params = study.best_params
best_lr = best_params['lr']
best_epoch = best_params['epochs']
"""
network = cbp.solve_clamped_beam_pytorch(4, 32, 50000, 1e-4)
coordinates = cbp.predict(network)

plt.scatter(coordinates[:, 0], coordinates[:, 1],label='pinns')
plt.scatter(fenics_true[:, 0], fenics_true[:, 1],label='fenics')
plt.legend()
plt.grid()
plt.show() 
