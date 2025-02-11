from fenics import *
from dolfin import fem, io, mesh, plot
import sys
from scipy import optimize as opt
import numpy as np
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
import matplotlib
import matplotlib.pyplot as plt


#info(NonlinearVariationalSolver.default_parameters(), 1)

plt.rcParams["font.size"] = 10

M = sys.float_info.max
#mesh generation
mesh = UnitSquareMesh(50,50) #, "right/left"
#finite-element-space
V = FunctionSpace(mesh, 'P', 1)
N = V.dim()
dx = Measure('dx',domain=mesh)
#boundary condition
u_D = Expression('0',degree=0)
def boundary(x,on_boundary):
	return on_boundary
bc = DirichletBC(V, u_D, boundary)
eps = 1e-10
set_log_level(LogLevel.ERROR)

#right_side

Omega = "Square"

Horak = {1.1: np.array([3.7586, 3.8702]), 
1.2: np.array([4.5012, 4.6179]),
1.3: np.array([5.2500, 5.3715]),
1.4: np.array([6.0385, 6.1621]),
1.5: np.array([6.8835, 7.0053]),
1.6: np.array([7.7971, 7.9118]),
1.7: np.array([8.7897, 8.8903]),
1.8: np.array([9.8708, 9.9490]),
1.9: np.array([11.050, 11.095]),
2: np.array([12.338, 12.338]),
2.1: np.array([13.684, 13.744]),
2.2: np.array([15.144, 15.282]),
2.3: np.array([16.725, 16.961]),
2.4: np.array([18.438, 18.797]),
2.5: np.array([20.293, 20.802]),
3: np.array([32.107, 33.956]),
4: np.array([74.757, 85.447]),
5: np.array([163.59, 205.08]),
6: np.array([343.77, 477.60]),
8: np.array([1402.1, 2443.4]),
10: np.array([5339.0, 11888])}

def compare_with_Horak(p,val): #see InvIt_square_medial for details
	if p in Horak: 
		hval = Horak[p].copy()
		hval *= 2**p
		return (hval - val)/hval
	p_r = 0.1 * int(10*p)
	if p_r in Horak and p_r + 0.1 in Horak:
		hval = (p - p_r) * Horak[p_r].copy() + (p_r + 0.1 - p) * Horak[p_r + 0.1].copy()
		hval *= 10
		hval *= 2**p
		return (hval - val)/hval
	return [np.nan, np.nan]
	
uprev = Expression('10 * x[0] * x[1] * (x[0] - x[1]) * (x[0] - 1) * (x[1] - 1)', degree=3)
marker = "diagonal"
	
uprev = project(uprev, V)

u0 = uprev.copy(deepcopy=True)

#number of iterations
it = 5
t = "{0}, {1}, it = {2}".format(Omega, marker, it)
FileHorak = open(t + " + Horak.txt", 'a')
FileHorak.write("Triangles num:\t" + str(mesh.num_cells()) + "\n")

#defining positive and negative parts of a function
def uplus(u):
	return conditional(u >= 0, u, 0)
def uminus(u):
	return conditional(u <= 0, -u, 0)
prange = [1.55, 1.54, 1.535]#, 1.5347, 1.53468
#for p in [1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5]:
for p in [1.6, 2, 3.5, 5]:
	if p >= 1.6:
		uprev = u0.copy(deepcopy=True)
	s = "{3}, p = {0}, {2}, it = {1}".format(p, it, marker, Omega)
	File = open(s + ".txt", 'w')
	File.write("NEW p:\t" + str(p) + str("\n"))
	
	#norms, functionals etc.
	def Lp(u):
		return assemble((abs(u)**p) * dx)
	def Wp(u):
		return assemble((grad(u)[0]**2 + grad(u)[1]**2)**(p/2) *  dx)
	def R(u):
		return Wp(u)/Lp(u)
	def Rplus(u):
		q = Lp(uplus(u))
		try:
			return Wp(uplus(u))/q
		except:
			return M
	def Rminus(u):
		q = Lp(uminus(u))
		try:
			return Wp(uminus(u))/q
		except:
			return M
	uprev.vector()[:] /= (Lp(uprev))**(1/p)

	File.write("\tR[u]:\t" + str(R(uprev)) + "\n")
	Wps = [] #W-norms of |u_{k+1} - u_k|
	Lps = [] #Lp-norms of |u_{k+1} - u_k|
	Rs = []
	Rs.append(R(uprev))
	print(p)
	for i in range(it):
		print("\t" + str(i))
		File.write("\nIteration:\t" + str(i) + '\n')
		def phi_alpha(a): #defining a boundary value problem for alpha
			b = (1 - a**p)**(1/p)
			u_ = uprev.copy(deepcopy=True)
			u = TrialFunction(V)
			v = TestFunction(V)
			F = (grad(u)[0]**2 + grad(u)[1]**2)**((p-2)/2) * dot(grad(u), grad(v)) * dx - a * uplus(uprev)**(p-1) * v * dx + b * uminus(uprev)**(p-1) * v * dx #/ (grad(u)[0]**2 + grad(u)[1]**2)**0.5 
			F = action(F, u_)
			du = TrialFunction(V)
			J = (p-2) * (grad(u_)[0]**2 + grad(u_)[1]**2)**((p-4)/2) * dot(grad(u_), grad(v)) * dot(grad(u_), grad(du)) * dx #/ (grad(u_)[0]**2 + grad(u_)[1]**2)**1.5 
			J += (grad(u_)[0]**2 + grad(u_)[1]**2)**((p-2)/2) * dot(grad(v), grad(du)) * dx #/ (grad(u_)[0]**2 + grad(u_)[1]**2)**0.5 
			problem = NonlinearVariationalProblem(F, u_, bc, J)
			solver  = NonlinearVariationalSolver(problem)
			solver.solve()
			u = u_.copy(deepcopy=True)
			u.vector()[:] /= (Lp(u))**(1/p)
			return u
		def rho(a): #defining an equation for alpha to be solved
			u = phi_alpha(a)
			plu = Rplus(u)
			minu = Rminus(u)
			return plu - minu
		optres = opt.root_scalar(rho, method="ridder", bracket=(0, 1)) #solving the equation for alpha
		alpha = optres.root
		File.write("\tOpt res:\t" + str(alpha) + ", " + str(rho(alpha)) + "\n")
		unew = phi_alpha(alpha)
		File.write("\t" + str(Rplus(unew)) + ", " + str(Rminus(unew)) + "\n")
		d = uprev - unew
		Wps.append(Wp(d)**(1/p))
		File.write("\tDev W_0^{1,p}-norm:\t" + str(Wps[-1]) + "\n")
		Lps.append(Lp(d)**(1/p))
		File.write("\tDev L^p-norm:\t" + str(Lps[-1]) + "\n")
		#print("\tResidual:\t" + str(Res(unew)))
		Rs.append(R(unew))
		File.write("\tR[u]:\t" + str(Rs[-1]) + "\n")
		uprev = unew

	fig = plt.figure()
	plot(uprev,mode="warp",linewidth = 0.1, 
                         antialiased = True,
                         edgecolor = 'dimgray', cmap='jet', vmin = uprev.vector().min(), vmax = uprev.vector().max())#cmap='cool' cmap='turbo' cmap='jet', 'seismic'
	for ax in fig.get_axes():
		ax.view_init(elev=19, azim=-116, roll=0)
		ax.set_xlabel(r'$x$')
		ax.set_ylabel(r'$y$')
	#plt.show()
	plt.tight_layout()
	plt.savefig(s + ", func.png") #"home/tmrxn/" + 
	plt.close()
	plt.semilogy(Rs, label=r'$R[u_k]$', color='magenta')
	plt.xlabel(r'$k$')
	plt.legend()
	plt.savefig(s + ", Rs.png")
	plt.close()
	fig, (ax1, ax2) = plt.subplots(2, sharex = True)
	ax1.semilogy(Wps, label=r'$||\nabla(\tilde u_{k+1} - \tilde u_{k})||_{p}$', color='red')
	ax1.legend()
	ax2.semilogy(Lps, label=r'$||\tilde u_{k+1} - \tilde u_{k}||_{p}$', color='green')
	ax2.set(xlabel=r'$k$')
	ax2.legend()
	plt.savefig(s + ", norms.png")
	plt.close()
	FileHorak.write(str(p) + "\t" + str(Rs[-1]) + "\t" + str(compare_with_Horak(p, Rs[-1])[0]) + "\t" + str(compare_with_Horak(p, Rs[-1])[1]) + "\n")
	save_func = open(s + ", resulting u.txt", "w")
	uprev.vector()[:].tofile(save_func, sep=" ")
	save_func.close()