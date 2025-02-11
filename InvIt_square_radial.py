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
mesh = UnitSquareMesh(100,100)
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

uprev = Expression('100 * x[0] * x[1] * (x[0] - 1) * (x[1] - 1) * (0.0625 - (x[0] - 0.5)*(x[0] - 0.5) - (x[1] - 0.5)*(x[1] - 0.5))', degree=6)
marker = "radial NLVS"
	
uprev = project(uprev, V)
#get_func = open("Square, p = 2, radial NLVS, it = 100, resulting u.txt", "r")
#array = np.loadtxt(get_func)
#for i in range(len(array)):
#	uprev.vector()[i] = array[i]
#start = 100
start = 0

u0 = uprev.copy(deepcopy=True)

#number of iterations
it = 100

#defining positive and negative parts of a function
def uplus(u):
	return conditional(u >= 0, u, 0)
def uminus(u):
	return conditional(u <= 0, -u, 0)

for p in [1.7, 2, 3]:
	alpha_star = 2**(-p)
	increase = False # whether |u_{k+1} - u_k| was increasing at the previous iteration
	uprev = u0.copy(deepcopy=True)
	s = "{3}, p = {0}, {2}, it = {1}".format(p, it, marker, Omega)
	s1 = "{3}, p = {0}, {2}, it = ".format(p, it, marker, Omega)
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
	Lps = [] #L-norms of |u_{k+1} - u_k|
	Rs = [] #Rayleigh quotients
	Rs.append(R(uprev))
	print(p)
	for i in range(start,it):
		print("\t" + str(i))
		File.write("\nIteration:\t" + str(i) + '\n')
		def phi_alpha(a): #sovling the boundary value problem for alpha
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
		optres = opt.root_scalar(rho, method="ridder", bracket=(0, 1), x0=alpha_star) #solving the equation for alpha
		alpha = optres.root
		File.write("\tOpt res:\t" + str(alpha) + ", " + str(rho(alpha)) + "\n")
		unew = phi_alpha(alpha)
		File.write("\t" + str(Rplus(unew)) + ", " + str(Rminus(unew)) + "\n")
		d = uprev - unew
		Wps.append(Wp(d)**(1/p))
		File.write("\tDev W_0^{1,p}-norm:\t" + str(Wps[-1]) + "\n")
		Lps.append(Lp(d)**(1/p))
		File.write("\tDev L^p-norm:\t" + str(Lps[-1]) + "\n")
		Rs.append(R(unew))
		File.write("\tR[u]:\t" + str(Rs[-1]) + "\n")
		if (i - start > 1 and Lps[-1] > Lps[-2] and not(increase)) or i == it - 1:
			increase=True
			print("\tLocal minimum: " + str(i-1))
			gr = plot(uprev,mode='contour',linewidths=0.5,colors='k')#colors = 'black' cmap='brg'
			gr = plot(uprev,cmap='jet')#colors = 'black' cmap='brg'
			plt.colorbar(gr)
			plt.tight_layout()
			plt.savefig(s1 + str(i) + ", func.png")
			plt.clf()
			plt.close()
			save_func = open(s1 + str(i + 1) + ", resulting u.txt", "w")
			uprev.vector()[:].tofile(save_func, sep=" ")
			save_func.close()
		uprev = unew
		if increase==True and Lps[-1] <= Lps[-2]:
			increase=False
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