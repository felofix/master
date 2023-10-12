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
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
f = [0, -rho*g*100]

n_length = 10
n_width = 3

def solve_clamped_beam_pytorch(n_hid, n_neu, epochs):
	"""
	PARAMETERS:

	n_hid = Number of hidden layers.
	n_neu = Number of neurons in each hidden layer.

	"""

	n_inputs = 2    # x and y.
	n_outputs = 2   # x and y. 

	# Neural network.
	net = Net(n_hid, n_neu, n_inputs, n_outputs)
	net = net.to(device)
	mse_cost_function = torch.nn.MSELoss() 			# Mean squared error
	optimizer = torch.optim.Adam(net.parameters())  # Can experiment with different optimizers.

	# Boundary conditions. 
	bx = np.linspace(0, lenght, n_length)
	by = np.linspace(0, width, n_width)
	bxij, byij = np.meshgrid(bx, by)
	dirichlet = np.arange(0, n_width * n_length, n_length)
	bxij, byij = bxij.flatten(), byij.flatten()


	for epoch in range(epochs):
		optimizer.zero_grad() # to make the gradients zero

		# Boundary loss.
		tbx = Variable(torch.from_numpy(bxij.reshape((len(bxij), 1))).float(), requires_grad=False).to(device)
		tby = Variable(torch.from_numpy(byij.reshape((len(byij), 1))).float(), requires_grad=False).to(device)
		zeros_bc = Variable(torch.from_numpy(np.zeros((len(dirichlet), 1))).float(), requires_grad=False).to(device)
		predicted_bc = net([tbx, tby])
		bcx = predicted_bc[:, 0][dirichlet].reshape(len(zeros_bc), 1)
		bcy = predicted_bc[:, 1][dirichlet].reshape(len(zeros_bc), 1)

		mse_ux = mse_cost_function(zeros_bc, bcx)
		mse_uy = mse_cost_function(zeros_bc, bcy)
		mse_bc = mse_ux + mse_uy

		# Collocation points. 
		xc = np.random.uniform(low=0.0, high=lenght, size=n_length)
		yc = np.random.uniform(low=0.0, high=width, size=n_width)

		xcij, ycij = np.meshgrid(xc, yc)
		xcij, ycij = xcij.flatten(), ycij.flatten()
		xcij, ycij = xcij.reshape((len(xcij), 1)), ycij.flatten().reshape((len(xcij), 1))

		txc = Variable(torch.from_numpy(xcij).float(), requires_grad=True).to(device)
		tyc = Variable(torch.from_numpy(ycij).float(), requires_grad=True).to(device)
		zeros_collcation = Variable(torch.from_numpy(np.zeros((len(txc), 1))).float(), requires_grad=False).to(device)

		res_x, res_y = navier_cauchy(txc, tyc, net)
		mse_xc = mse_cost_function(zeros_collcation, res_x)
		mse_yc = mse_cost_function(zeros_collcation, res_y)
		mse_cc = mse_xc + mse_yc

		loss = mse_bc + mse_cc

		print(mse_bc)
		print(bcx)
		print(bcy)

		loss.backward() # This is for computing gradients using backward propagation
		optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
	
	return net

class Net(nn.Module):
	def __init__(self, num_hidden_layers, num_neurons, ninputs, noutputs):
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

		for hl in range(1, self.num_hidden_layers):
			self.hidden_layers.append(nn.Linear(self.num_neurons, self.num_neurons))

		self.output_layer = nn.Linear(self.num_neurons, self.noutputs)

	def forward(self, inputs):  
		"""Moving the neural network forward. 
		Forward step of the neural network. 
		"""
		layer_inputs = torch.cat(inputs, axis=1) 
		layer = torch.sigmoid(self.hidden_layers[0](layer_inputs))

		for hl in range(1, self.num_hidden_layers):
			layer = torch.sigmoid(self.hidden_layers[hl](layer))

		output = self.output_layer(layer) 
		return output


def navier_cauchy(x, y, net):
	"""Finds the residual for
	the Navier Cauchy partial differential equation. 
	"""
	u = net([x, y])

	u_x = u[:, 0]
	u_y = u[:, 1]

	u_x1x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
	u_x1y = torch.autograd.grad(u_x, y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
	u_y1y = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
	u_y1x = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
	
	u_x2x = torch.autograd.grad(u_x1x, x, grad_outputs=torch.ones_like(u_x1x), create_graph=True)[0]
	u_y1xy = torch.autograd.grad(u_y1y, x, grad_outputs=torch.ones_like(u_y1y), create_graph=True)[0]
	u_x2y = torch.autograd.grad(u_x1y, y, grad_outputs=torch.ones_like(u_x1y), create_graph=True)[0]

	residue_x = (lambda_ + mu) * (u_x2x + u_y1xy) + mu * (u_x2x + u_x2y) + f[0]

	u_x1yx = torch.autograd.grad(u_x1x, y, grad_outputs=torch.ones_like(u_x1x), create_graph=True)[0]
	u_y2y = torch.autograd.grad(u_y1y, y, grad_outputs=torch.ones_like(u_y1y), create_graph=True)[0]
	u_y2x = torch.autograd.grad(u_y1x, x, grad_outputs=torch.ones_like(u_y1x), create_graph=True)[0]

	residue_y = (lambda_ + mu) * (u_x1yx + u_y2y) + mu * (u_y2x + u_y2y) + f[1]

	return residue_x, residue_y

import matplotlib.pyplot as plt
# Tets
network = solve_clamped_beam_pytorch(5, 5, 1000)

# Boundary conditions. 
x = np.linspace(0, lenght, n_length)
y = np.linspace(0, width, n_width)

for i in x:
	for j in y:
		plt.scatter(i, j, color='Blue')
		xarr = np.array(i).reshape((1,1))
		yarr = np.array(j).reshape((1,1))
		tx = Variable(torch.from_numpy(xarr).float(), requires_grad=False).to(device)
		ty = Variable(torch.from_numpy(yarr).float(), requires_grad=False).to(device)
		deform = network([tx, ty]).detach().numpy()
		newx = i + deform[0,0]
		newy = j + deform[0,1]
		plt.scatter(newx, newy, color='Orange')

plt.grid()
plt.show()



