#!/usr/bin/env python3
import dolfin as df
import numpy as np
import pandas as pd

#%% Options
L  = 2.0
H  = 2.0
nu = 1e-2

n1 = 30
n2 = 30

formulation = 'stokes'
# formulation = 'navier-stokes'

# testcase = 'channel-flow'
testcase = 'cavity'

#%% Solver
mesh = df.RectangleMesh(df.Point(0, 0), df.Point(L, H), n1, n2)

V = df.VectorElement("CG", mesh.ufl_cell(), 2)
Q = df.FiniteElement("CG", mesh.ufl_cell(), 1)
W = df.FunctionSpace(mesh, V * Q)

bcs = list()
tol = 1e-14

if testcase == 'channel-flow':
    def noslip_boundary(x, on_boundary):
        return on_boundary and (x[1] < tol or x[1] > H - tol)
    def inflow_boundary(x, on_boundary):
        return on_boundary and x[0] < tol
    def outflow_boundary(x, on_boundary):
        return on_boundary and x[0] > L - tol

    inflow_function = df.Expression(("x[1]*(H - x[1])/std::pow(H,2)", "0.0"), H = H, degree=2)

    bcs.append(df.DirichletBC(W.sub(0), df.Constant((0, 0)), noslip_boundary))
    bcs.append(df.DirichletBC(W.sub(0), inflow_function    , inflow_boundary))
    bcs.append(df.DirichletBC(W.sub(1), df.Constant(0)     , outflow_boundary))
elif testcase == 'cavity':
    def noslip_boundary(x, on_boundary):
        return on_boundary and (x[1] < tol or x[0] < tol or x[0] > L - tol)
    def top_boundary(x, on_boundary):
        return on_boundary and x[1] > H - tol

    bcs.append(df.DirichletBC(W.sub(0), df.Constant((0, 0)), noslip_boundary))
    bcs.append(df.DirichletBC(W.sub(0), df.Constant((1, 0)), top_boundary))

(v, q) = df.TestFunctions(W)
(u, p) = df.TrialFunctions(W)
f = df.Constant((0, 0))
w = df.Function(W)

if formulation == 'stokes':
    print('Stokes...')
    a = (
            df.Constant(nu)*df.inner(df.grad(u), df.grad(v))
            - df.div(v)*p
            + q*df.div(u)
        )*df.dx
    rhs = df.inner(f, v)*df.dx
    df.solve(a == rhs, w, bcs)
if formulation == 'navier-stokes':
    print('Navier-Stokes...')
    NS = (
            df.Constant(nu)*df.inner(df.grad(u), df.grad(v))
            + df.inner(df.grad(u)*u, v)
            - df.div(v)*p + q*df.div(u)
            - df.inner(f, v)
        )*df.dx
    residual = df.action(NS, w)
    jacobian = df.derivative(residual,  w)
    problem  = df.NonlinearVariationalProblem(residual, w, bcs, jacobian)
    solver   = df.NonlinearVariationalSolver(problem)
    solver.solve()

(u, p) = w.split()
if testcase == 'cavity':
    # remove mean value
    p_mean = df.assemble(p*df.dx(mesh)) / df.assemble(df.Constant(1.0)*df.dx(mesh))
    p.vector().set_local(p.vector().get_local() - p_mean)

#%% Postprocessing
output_file = 'data/' + formulation + '_' + testcase + '_steady'
u.rename('u', 'u')
p.rename('p', 'p')
xdmf_file = df.XDMFFile(output_file + '.xdmf')
xdmf_file.parameters['flush_output'] = True
xdmf_file.parameters['functions_share_mesh'] = True
xdmf_file.parameters['rewrite_function_mesh'] = False
xdmf_file.write(u, 0)
xdmf_file.write(p, 0)

#%% Export to csv

num_out_points = 100

x_points = np.random.rand(num_out_points) * L
y_points = np.random.rand(num_out_points) * H
u_points = np.array([u(x,y) for x,y in zip(x_points, y_points)])
p_points = np.array([p(x,y) for x,y in zip(x_points, y_points)])

data = pd.DataFrame({'x': x_points,
                     'y': y_points,
                     'ux': u_points[:,0],
                     'uy': u_points[:,1],
                     'p' : p_points})
data.to_csv(output_file + '.csv', index = False)