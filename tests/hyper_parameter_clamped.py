import sys
import numpy as np
sys.path.append('../pytorch_code')
import clamped_beam_pytorch as cbp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import seaborn as sns
fenics_true = np.loadtxt("../data/clamped_beam_fenics.txt", delimiter=',')
fenics_before = np.loadtxt("../data/clamped_beam_fenics_before.txt", delimiter=',')

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

def plot_mse_vs_epochs(epochs, mses, title):
	# Set the style of the visualization
	sns.set_theme(style="whitegrid")

	# Create a color palette
	palette = sns.color_palette("husl", 1)

	# Create a line plot of 'mses' vs 'epochs'
	sns.lineplot(x=epochs, y=mses,marker='o', palette=palette, linewidth=2.5, label='MSE ' + title)

	# Set title and labels for axes
	plt.legend()
	plt.xlabel("Epochs", fontsize=14)
	plt.ylabel("Mean Squared Error (MSE)", fontsize=14)


def plot_activation_functions():
	# Tessting different activation functions. 
	activation_functions = {'sigmoid': torch.sigmoid, 'relu': torch.relu,'tanh': torch.tanh}
	epochs = [1000, 3000, 5000, 8000, 10000]
	learning_rates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

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

def plot_epochs():
	# Plotting epochs vs loss for 50k. 
	epochs_range = np.arange(1000, 21000, 1000)
	ADAM_relu = 1e-3
	network, losses = cbp.solve_clamped_beam_pytorch(4, 32, 20000, 
													ADAM_relu,  
													activation_function=torch.relu,
													verbose=True)
	print(f'The best loss for the Relu function was: {np.min(losses)}')
	plot_mse_vs_epochs(epochs_range[2:], losses[2:], "ReLU")
	ADAM_sigmoid = 1e-3
	network, losses = cbp.solve_clamped_beam_pytorch(4, 32, 20000, 
													ADAM_sigmoid,  
													activation_function=torch.sigmoid,
													verbose=True)
	print(f'The best loss for the Sigmoid function was: {np.min(losses)}')
	plot_mse_vs_epochs(epochs_range[2:], losses[2:], "Sigmoid")
	ADAM_tanh = 1e-4
	network, losses = cbp.solve_clamped_beam_pytorch(4, 32, 20000, 
													ADAM_tanh,  
													activation_function=torch.tanh,
													verbose=True)
	print(f'The best loss for the tanh function was: {np.min(losses)}')
	plot_mse_vs_epochs(epochs_range[2:], losses[2:], "Tanh")

	plt.savefig("plots_clamped_beam/losses.pdf")

def plot_real_points():
	"""
	Using the best parameters from the previous two functions we 
	plot against the amount of random points from 1 to 10 for 30000 epochs. 
	"""
	points = np.arange(11)
	mses = np.zeros(len(points))
	epochs = 10000
	lr = 1e-4

	for p in range(len(points)):
		network = cbp.solve_clamped_beam_pytorch(4, 32, epochs, 
												lr,  
												activation_function=torch.tanh,
												verbose=False,
												random_amount = points[p])
		coordinates = cbp.predict(network)
		mses[p] = mean_squared_error(coordinates, fenics_true)

	print(f'The best loss for points was: {np.min(mses)}')

	# Set the style of the visualization
	sns.set_theme(style="whitegrid")

	# Create a color palette
	palette = sns.color_palette("husl", 1)

	# Create a line plot of 'mses' vs 'epochs'
	sns.lineplot(x=points, y=mses,marker='o', palette=palette, linewidth=2.5)

	# Set title and labels for axes
	plt.xlabel("Real points", fontsize=14)
	plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
	plt.savefig("plots_clamped_beam/real_points.pdf")

def plot_best_mse():
	"""
	Using the best parameters from the previous two functions we 
	plot against the amount of random points from 1 to 10 for 30000 epochs. 
	"""
	epochs = 20000
	lr = 1e-4
	network = cbp.solve_clamped_beam_pytorch(4, 32, epochs, 
												lr,  
												activation_function=torch.tanh,
												verbose=False,
												random_amount = 6)
		
	coordinates = cbp.predict(network)
	mse = mean_squared_error(coordinates, fenics_true)

	print(f'The best loss for points was: {mse}')

	plt.scatter(coordinates[:, 0], coordinates[:, 1], s=5, color='red', label='PINNs')
	plt.scatter(fenics_true[:, 0], fenics_true[:, 1], s=5, color='black', label='FEM')
	plt.grid()
	plt.legend()
	plt.savefig("plots_clamped_beam/pinnvsfem.pdf")


if __name__ == '__main__':
	#plot_activation_functions()
	#plot_epochs()
	#plot_real_points()
	#plot_best_mse()








