import dolfin as df
import numpy as np

#%% Options

time_step = 1e-4
dt = df.Constant(time_step)
mu  = 1e-2   # kg/m*s
rho = 1.06e3 # kg/m^3
nu = df.Constant(1e4*mu/rho) # cm^2/s
U  = 20 # cm/s
H = np.sqrt(0.4**2+0.1**2) #cm
T  = 1e-2
toll = 1e-14

formulation = 'navier-stokes_SI'
testcase = 'coronary'

print("Reynolds -> %1.4f" %(U*(H/2)/nu))

#%% Creating SubDomains

# Sub domain for no-slip (mark whole boundary, inflow and outflow will overwrite)
class Noslip(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
# Sub domain for inflow (left)
class Inflow(df.SubDomain):
    def inside(self, x, on_boundary):
        return np.abs(x[1]+4*x[0]+6.4) < toll and on_boundary
# Sub domain for outflow_1 (upper_right)
class Outflow1(df.SubDomain):
    def inside(self, x, on_boundary):
        return np.abs(x[1]+2*x[0]-4.8) < toll and on_boundary
# Sub domain for outflow_2 (lower_right)
class Outflow2(df.SubDomain):
    def inside(self, x, on_boundary):
        return np.abs(x[0]-1.2) < toll and x[1] < 0 and on_boundary

#%% Creating Mesh

times = np.arange(0.0, T, step = time_step)

import meshio
msh = meshio.read("coroParam.msh")
meshio.write("coroParam.xml", msh)
mesh = df.Mesh("coroParam.xml")

sub_domains = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(4) # default value is interior

Noslip().mark(sub_domains, 0)
Inflow().mark(sub_domains, 1)
Outflow1().mark(sub_domains, 2)
Outflow2().mark(sub_domains, 3)

#file = df.File("subdomains.xml")
#file << sub_domains

#%% Spaces and Functions

V = df.VectorElement("CG", mesh.ufl_cell(), 2)
Q = df.FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem = df.MixedElement([V,Q])
W = df.FunctionSpace(mesh,W_elem)

(v, q) = df.TestFunctions(W)
(u, p) = df.TrialFunctions(W)
f = df.Constant((0, 0, 0))
w = df.Function(W)
w_old = df.Function(W)

# u(s) = s * (1-s)
# s = sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H
inflow_function = df.Expression(("U*cos_theta*(std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H)*(1-std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H)",
                                 "U*sin_theta*(std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H)*(1-std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H)",
                                 "0"), cos_theta = np.cos(np.arctan(1/4)), sin_theta = np.sin(np.arctan(1/4)),
                                       x0 = -1.4, y0 = -0.8, H = H, U = U, degree=2)

bcs = list()
bcs.append(df.DirichletBC(W.sub(0), df.Constant((0, 0, 0)), sub_domains, 0))
bcs.append(df.DirichletBC(W.sub(0), inflow_function       , sub_domains, 1))
ds = df.Measure('ds', domain=mesh, subdomain_data=sub_domains)


#%% Save output

def save_output(w, t, it):
    (u, p) = w.split()
    u.rename('u', 'u')
    p.rename('p', 'p')
    xdmf_file = df.XDMFFile('data/Coronary/' + formulation + '_' + testcase + '_unsteady_%05d.xdmf' % it)
    xdmf_file.parameters['flush_output'] = True
    xdmf_file.parameters['functions_share_mesh'] = True
    xdmf_file.parameters['rewrite_function_mesh'] = False
    xdmf_file.write(u, t)
    xdmf_file.write(p, t)


#%% Solver
n = df.FacetNormal(mesh)
save_output(w, 0, 0)
for i in range(1, len(times)):
    t = times[i]
    print('*** solving time t = %1.6f ***' % t)
    w_old.assign(w)
    (u_old, p_old) = w_old.split()

    print('Navier-Stokes (semi-implicit)...')
    a = (
        df.inner(u, v)/dt
        + nu*df.inner(df.grad(u), df.grad(v))
        + df.inner(df.grad(u)*u_old, v)
        - df.div(v)*p
        + q*df.div(u)
        )*df.dx + (
        df.dot(p/nu,df.inner(n,v)))*df.ds(2) + (df.dot(p/nu,df.inner(n,v)))*df.ds(3)
    rhs = (
          df.inner(u_old, v)/dt
        + df.inner(f, v))*df.dx
    df.solve(a == rhs, w, bcs)

    save_output(w, t, i)

print('-----------Done------------')
