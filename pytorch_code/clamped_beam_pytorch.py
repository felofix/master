"""
Problem to be solved: Clamped beam in two dimentions.

Input: x and y position.
Output: x and y position displacement.

Notes:
Should I use a meshgrid as an input, or two arrays?
How should I define my boundaries?

To do:
- Get the solver working. ()
- Have my constants somewhere that is more clean. ()
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.autograd import Variable
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error


fenics_before = np.loadtxt("../data/clamped_beam_fenics_before.txt", delimiter=',')
fenics_displacement = np.loadtxt("../data/clamped_beam_fenics_displacement.txt", delimiter=',')
fenics_after = np.loadtxt("../data/clamped_beam_fenics.txt", delimiter=',')

torch.manual_seed(2023)
np.random.seed(2023)

# Constants
lenght = 1; width = 0.2
mu = 1
rho = 1
delta = width/lenght
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma
f = [0, -rho*g]

n_length = 11 # + 1
n_width = 4 # + 1

def solve_clamped_beam_pytorch(n_hid, n_neu, epochs, lr, activation_function = torch.tanh, verbose=False, random_amount = 5):
	"""
	PARAMETERS:

	n_hid = Number of hidden layers.
	n_neu = Number of neurons in each hidden layer.

	"""
	n_inputs =  2   # x and y.
	n_outputs = 2   # displacement in x and y. 

	# Neural network.
	net = Net(n_hid, n_neu, n_inputs, n_outputs, activation_function)
	net = net.to(device)
	mse_cost_function = torch.nn.MSELoss(reduction ='mean') # Mean squared error
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)

	# Boundary conditions. 
	bx = np.linspace(0, lenght, n_length)
	by = np.linspace(0, width, n_width)
	bxij, byij = np.meshgrid(bx, by)
	dirichlet = np.arange(0, n_width * n_length, n_length)
	bxij, byij = bxij.flatten(), byij.flatten()
	bxij, byij = bxij[dirichlet], byij[dirichlet] 

	# Fenics loss points.
	random_indices = np.random.choice(fenics_before.shape[0], random_amount, replace=False)
	random_points= fenics_before[random_indices, :]
	random_displacements = fenics_displacement[random_indices, :]

	losses = []


	with tqdm(total=epochs, desc="Epochs") as epoch_pbar:
		for epoch in range(epochs):
			optimizer.zero_grad()

			# Boundary loss.
			tbx = Variable(torch.from_numpy(bxij.reshape((len(bxij), 1))).float(), requires_grad=False).to(device)
			tby = Variable(torch.from_numpy(byij.reshape((len(byij), 1))).float(), requires_grad=False).to(device)
			zeros_bc = Variable(torch.from_numpy(np.zeros((len(dirichlet), 1))).float(), requires_grad=False).to(device)
			predicted_bc = net([tbx, tby])

			bcx = predicted_bc[:, 0].reshape(len(zeros_bc), 1)
			bcy = predicted_bc[:, 1].reshape(len(zeros_bc), 1)

			mse_ux = mse_cost_function(zeros_bc, bcx)
			mse_uy = mse_cost_function(zeros_bc, bcy)
			mse_bc = mse_ux + mse_uy

			# Collocation points. 
			xc = np.random.uniform(low=0.0001, high=lenght, size=n_length)
			yc = np.random.uniform(low=0.0001, high=width, size=n_width)

			xcij, ycij = np.meshgrid(xc, yc)
			xcij, ycij = xcij.flatten(), ycij.flatten()
			xcij, ycij = xcij.reshape((len(xcij), 1)), ycij.flatten().reshape((len(xcij), 1))

			txc = Variable(torch.from_numpy(xcij).float(), requires_grad=True).to(device)
			tyc = Variable(torch.from_numpy(ycij).float(), requires_grad=True).to(device)
			zeros_collcation = Variable(torch.from_numpy(np.zeros((len(txc), 1))).float(), requires_grad=False).to(device)

			res_x, res_y = navier_cauchy_loss(txc, tyc, net)
			
			mse_xc = mse_cost_function(zeros_collcation, res_x)
			mse_yc = mse_cost_function(zeros_collcation, res_y)
			mse_cc = mse_xc + mse_yc

			# Fenics loss.
			fx = Variable(torch.from_numpy(random_points[:,0].reshape((len(random_points[:,0]), 1))).float(), requires_grad=False).to(device)
			fy = Variable(torch.from_numpy(random_points[:,1].reshape((len(random_points[:,1]), 1))).float(), requires_grad=False).to(device)
			fxd = Variable(torch.from_numpy(random_displacements[:,0].reshape((len(random_displacements[:,0]),))).float(), requires_grad=False).to(device)
			fyd = Variable(torch.from_numpy(random_displacements[:,1].reshape((len(random_displacements[:,1]),))).float(), requires_grad=False).to(device)

			f_loss_x, f_loss_y = fenics_loss(fx, fy, net)
			mse_fe_x = mse_cost_function(fxd, f_loss_x)
			mse_fe_y = mse_cost_function(fyd, f_loss_y)
			mse_fe = mse_fe_x + mse_fe_y

			# For Adam optimizer
			loss = mse_fe + mse_cc + mse_bc*10

			if verbose and epoch % 1000 == 0:
				predicted = predict(net)
				mse = mean_squared_error(fenics_after, predicted)
				losses.append(mse)

			loss.backward()
			optimizer.step()

			epoch_pbar.update(1)

	if not verbose:
		return net
	else:
		return net, losses

class Net(nn.Module):
	def __init__(self, num_hidden_layers, num_neurons, ninputs, noutputs, activation_function):
		"""Initializing the neural network. 
		Trying to learn how the network runs. 
		"""
		super(Net, self).__init__()
		self.num_hidden_layers = num_hidden_layers
		self.num_neurons = num_neurons
		self.ninputs = ninputs
		self.noutputs = noutputs
		self.hidden_layers = nn.ModuleList()
		self.hidden_layers.append(nn.Linear(self.ninputs, self.num_neurons))
		self.activation_function = activation_function

		for hl in range(1, self.num_hidden_layers):
			self.hidden_layers.append(nn.Linear(self.num_neurons, self.num_neurons))

		self.output_layer = nn.Linear(self.num_neurons, self.noutputs)

	def forward(self, inputs):  
		"""Moving the neural network forward. 
		Forward step of the neural network. 
		"""
		layer_inputs = torch.cat(inputs, axis=1) 
		layer = self.activation_function(self.hidden_layers[0](layer_inputs))

		for hl in range(1, self.num_hidden_layers):
			layer = self.activation_function(self.hidden_layers[hl](layer))

		output = self.output_layer(layer) 
		return output

def navier_cauchy_loss(x, y, net):
	u = net([x, y])
	u_x = u[:, 0]
	u_y = u[:, 1]

	u_xx = diff(u_x, x)
	u_yy = diff(u_y, y)
	u_xy = diff(u_x, y)
	u_yx = diff(u_y, x)

	u_y_xx = diff(u_xx, y)
	u_x_yy = diff(u_yy, y)
	u_x_xx = diff(u_xx, x)
	u_y_xy = diff(u_xy, y)
	u_y_yy = diff(u_yy, y)
	u_x_yx = diff(u_yx, x)

	residue_x = (lambda_ + mu)*(u_x_xx + u_x_yy) + mu*(u_x_xx + u_y_xy) + f[0]
	residue_y = (lambda_ + mu)*(u_y_xx + u_y_yy) + mu*(u_x_yx + u_y_yy) + f[1]

	return residue_x, residue_y

def fenics_loss(x, y, net):
	u = net([x, y])
	return u[:, 0], u[:, 1]


def diff(u, d):
	return torch.autograd.grad(u, d, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

def predict(network):
	x = np.linspace(0, lenght, n_length)
	y = np.linspace(0, width, n_width)
	xij, yij = np.meshgrid(x, y)
	xij = xij.reshape((len(xij.flatten()), 1))
	yij = yij.reshape((len(yij.flatten()), 1))

	tx = Variable(torch.from_numpy(xij).float(), requires_grad=False).to(device)
	ty = Variable(torch.from_numpy(yij).float(), requires_grad=False).to(device)
	deform = network([tx, ty]).detach().numpy()

	newx = xij + deform[:, 0].reshape((len(deform[:, 0]), 1))
	newy = yij + deform[:, 1].reshape((len(deform[:, 1]), 1))

	coordinates_after = np.zeros((len(newx), 2))
	coordinates_after[:, 0], coordinates_after[:, 1] = newx.flatten(), newy.flatten()

	return coordinates_after

print(g)