import fenics as fe
import matplotlib.pyplot as plt
import numpy as np

# Write 'c'onda activate fenicsporject' in terminal. 

# Constants
lenght = 1; width = 0.2
mu = 1
rho = 1
delta = width/lenght
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

n_length = 10
n_width = 3

def solve_clamped_beam_fenics():
	"""
	A two dimentional clamped beam problem. 
	As of now it is a pretty simple solver. 

	"""
	# Mesh and Vector Function Space
	mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(lenght, width), n_length, n_width)
	
	# Get the coordinates of the vertices before deformation. 
	coordinates_before = mesh.coordinates().copy()

	# Creating vector space for basis functions.
	V = fe.VectorFunctionSpace(mesh, "Lagrange", 1)

	# Define boundary condition tolarance.
	tol = 1E-14

	def clamped_boundary(x, on_boundary):
		"""
		Dirichlet boundary condtions. 
		"""
		return on_boundary and x[0] < tol

	bc = fe.DirichletBC(V, fe.Constant((0, 0)), clamped_boundary)

	def epsilon(u):
		# Engineering strain. 
		return 0.5*(fe.nabla_grad(u) + fe.nabla_grad(u).T)
		#return sym(nabla_grad(u))

	def sigma(u):
		# Stress. 
		return lambda_*fe.div(u)*fe.Identity(d) + 2*mu*epsilon(u)

	# Define variational problem
	u = fe.TrialFunction(V)
	d = u.geometric_dimension()  # space dimension
	v = fe.TestFunction(V)
	f = fe.Constant((0, -rho*g))
	T = fe.Constant((0, 0))
	a = fe.inner(sigma(u), epsilon(v))*fe.dx
	L = fe.dot(f, v)*fe.dx + fe.dot(T, v)*fe.ds
	
	# Compute solution
	u = fe.Function(V)      # This is the displacement field, that is how much it moves. 
	fe.solve(a == L, u, bc)

	# Move the mesh according to the displacement field u
	fe.ALE.move(mesh, u)

	# Now mesh.coordinates() will return the new coordinates of the mesh vertices
	coordinates_after = mesh.coordinates()

	return coordinates_before, coordinates_after



















