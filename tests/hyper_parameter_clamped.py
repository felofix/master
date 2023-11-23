import sys
import numpy as np
sys.path.append('../pytorch_code')
import clamped_beam_pytorch as cbp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import seaborn as sns

def plot_heatmap(matrix, row_labels, col_labels, title, cmap="YlGnBu"):
	"""
	Plot a heatmap using Seaborn.

	Parameters:
		matrix (numpy.array): 2D array to be plotted.
		row_labels (list): Labels for the rows.
		col_labels (list): Labels for the columns.
		title (str): Title of the heatmap.
		cmap (str, optional): Color map. Defaults to "YlGnBu".

	Returns:
		None
	"""
	plt.figure(figsize=(10, 8))
	
	# Create a heatmap using Seaborn
	sns.heatmap(matrix, annot=True, cmap=cmap, cbar=True, 
				xticklabels=col_labels, yticklabels=row_labels, 
				fmt=".2e", linewidths=0.5)

	# Setting the title
	plt.title(title, fontsize=18)
	
	# Setting x and y labels
	plt.xlabel('Epochs', fontsize=14)
	plt.ylabel('Learning rates', fontsize=14)

	plt.savefig("plots_clamped_beam/" + title + '.pdf')


fenics_true = np.loadtxt("../data/clamped_beam_fenics.txt", delimiter=',')
optimizers = ['ADAM', 'LBFGS']
activation_functions = {'sigmoid': torch.sigmoid, 'relu': torch.relu,'tanh': torch.tanh}
epochs = [1000, 3000, 5000, 8000, 10000]
learning_rates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

for opt in optimizers:
	for act in activation_functions:
		mses = np.zeros((len(learning_rates), len(epochs)))
		title = opt + "_" + act

		for epo in range(len(epochs)):
			for lea in range(len(learning_rates)):
				network = cbp.solve_clamped_beam_pytorch(4, 32, epochs[epo], 
														 learning_rates[lea], 
														 optimizer_type=opt, 
														 activation_function=activation_functions[act])
				coordinates = cbp.predict(network)
				mses[epo, lea] = mean_squared_error(coordinates, fenics_true)

		plot_heatmap(mses, learning_rates, epochs, title)




