#!/usr/bin/env python3
import dolfin as df
import numpy as np

#%% Options

nu = 1e0
T  = 1e-2
U  = 1
dt = 1e-4
tol = 1e-14

formulation = 'navier-stokes_SI'
testcase = 'coronary'

#%% Solver

import meshio
msh = meshio.read("coroParam.msh")
meshio.write("coroParam.xml",msh)
mesh = df.Mesh("coroParam.xml")

boundary_mesh = df.BoundaryMesh(mesh, "exterior", True)
boundary_points = boundary_mesh.coordinates()

# ids: 0 for no_slip boundary
#      1 for inflow boundary
#      2 for outflow1 boundary
#      3 for outflow2 boundary

def get_id(x,y,tol):
    if (np.abs(y+4*x+6.4)<tol) : return 1
    if (np.abs(y+2*x-4.8)<tol) : return 2
    if (np.abs(x-1.2)<tol) and (y<0) : return 3
    else : return 0

ids = [get_id(x[0],x[1],tol) for x in boundary_points[:,0:2]]
ids = np.expand_dims(np.array(ids), axis = 1)
np.save("data/Coronary/bpoints.npy",np.hstack((boundary_points,ids)))

V = df.VectorElement("CG", mesh.ufl_cell(), 2)
Q = df.FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem = df.MixedElement([V,Q])
W = df.FunctionSpace(mesh,W_elem)

bcs = list()

def inflow_boundary(x,on_boundary):
    return on_boundary and (np.abs(x[1]+4*x[0]+6.4)<tol)

def outflow1_boundary(x,on_boundary):
    return on_boundary and (np.abs(x[1]+2*x[0]-4.8)<tol)

def outflow2_boundary(x,on_boundary):
    return on_boundary and (np.abs(x[0]-1.2)<tol) and (x[1]<0)

def outflow_boundary(x,on_boundary):
    return outflow1_boundary(x,on_boundary) or outflow2_boundary(x,on_boundary)

def noslip_boundary(x,on_boundary):
    return on_boundary and not(inflow_boundary(x, on_boundary)) and not(outflow_boundary(x, on_boundary))

# u(s) = s * (1-s)
# s = std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H

inflow_function = df.Expression(("cos_theta*(std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H)*(1-std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H)", 
                                  "sin_theta*(std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H)*(1-std::sqrt((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / H)",
                                  "0"), cos_theta = np.cos(np.arctan(1/4)), sin_theta = np.sin(np.arctan(1/4)), 
                                        x0 = -1.4, y0 = -0.8, H = np.sqrt(0.4**2+0.1**2), degree=2)

bcs.append(df.DirichletBC(W.sub(0), df.Constant((0, 0, 0)), noslip_boundary))
bcs.append(df.DirichletBC(W.sub(0), inflow_function       , inflow_boundary))
bcs.append(df.DirichletBC(W.sub(1), df.Constant(0)        , outflow_boundary))

(v, q) = df.TestFunctions(W)
(u, p) = df.TrialFunctions(W)
f = df.Constant((0, 0, 0))
w = df.Function(W)
w_old = df.Function(W)


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


times = np.arange(0.0, T, step = dt)
save_output(w, 0, 0)


for i in range(1, len(times)):
    t = times[i]
    print('*** solving time t = %1.6f ***' % t)
    w_old.assign(w)
    (u_old, p_old) = w_old.split()

    if formulation == 'stokes':
        print('Stokes...')
        a = (
                df.inner(u, v)/df.Constant(dt)
                + df.Constant(nu)*df.inner(df.grad(u), df.grad(v))
                - df.div(v)*p
                + q*df.div(u)
            )*df.dx
        rhs = (
                df.inner(u_old, v)/df.Constant(dt)
                + df.inner(f, v)
            )*df.dx
        df.solve(a == rhs, w, bcs)
        
    elif formulation == 'navier-stokes_SI':
        print('Navier-Stokes (semi-implicit)...')
        a = (
                df.inner(u, v)/df.Constant(dt)
                + df.Constant(nu)*df.inner(df.grad(u), df.grad(v))
                + df.inner(df.grad(u)*u_old, v)
                - df.div(v)*p
                + q*df.div(u)
            )*df.dx
        rhs = (
                df.inner(u_old, v)/df.Constant(dt)
                + df.inner(f, v)
            )*df.dx
        df.solve(a == rhs, w, bcs)

    save_output(w, t, i)

print('-----------Done------------')