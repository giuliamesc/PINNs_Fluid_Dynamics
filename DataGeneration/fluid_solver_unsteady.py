#!/usr/bin/env python3
import dolfin as df
import numpy as np

#%% Options
L  = 1.0
H  = 1.0
nu = 1e0
T  = 1e-2
U  = 1

n1 = 100
n2 = 100
dt = 1e-4

# formulation = 'stokes'
#formulation = 'navier-stokes_I'
formulation = 'navier-stokes_SI'

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
    bcs.append(df.DirichletBC(W.sub(0), df.Constant((U, 0)), top_boundary))

(v, q) = df.TestFunctions(W)
(u, p) = df.TrialFunctions(W)
f = df.Constant((0, 0))
w = df.Function(W)
w_old = df.Function(W)


#%% Export to csv

import pandas as pd

data_list = []
num_out_points = 100 #square root of the desired number of points
x_points = np.linspace(0, L , num_out_points)
y_points = np.linspace(0, H , num_out_points)

def create_csv_for_df(w, t):
    (u, p) = w.split()
    u_points = np.array([u(x,y) for y in y_points for x in x_points])
    p_points = np.array([p(x,y) for y in y_points for x in x_points])
    x_tab = np.array([x for y in y_points for x in x_points])
    y_tab = np.array([y for y in y_points for x in x_points])
    t_tab = np.array([t for y in y_points for x in x_points])
    data = pd.DataFrame({'t': t_tab,
                         'x': x_tab,
                         'y': y_tab,
                         'ux': u_points[:,0],
                         'uy': u_points[:,1],
                         'p' : p_points})
    return data

def save_output(w, t, it):
    (u, p) = w.split()
    u.rename('u', 'u')
    p.rename('p', 'p')
    xdmf_file = df.XDMFFile('data/UnsteadyCase/' + formulation + '_' + testcase + '_unsteady_%05d.xdmf' % it)
    xdmf_file.parameters['flush_output'] = True
    xdmf_file.parameters['functions_share_mesh'] = True
    xdmf_file.parameters['rewrite_function_mesh'] = False
    xdmf_file.write(u, t)
    xdmf_file.write(p, t)


times = np.arange(0.0, T, step = dt)
save_output(w, 0, 0)
temp_dataframe = create_csv_for_df(w, 0)
data_list.append(temp_dataframe)

for i in range(1, len(times)):
    t = times[i]
    print('*** solving time t = %1.6f ***' % t)
    # top_function.u_horiz = 0.1*np.sin(10*t)
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
    elif formulation == 'navier-stokes_I':
        print('Navier-Stokes (implicit)...')
        NS = (
                df.inner(u - u_old, v)/df.Constant(dt)
                + df.Constant(nu)*df.inner(df.grad(u), df.grad(v))
                + df.inner(df.grad(u)*u, v)
                - df.div(v)*p + q*df.div(u)
                - df.inner(f, v)
            )*df.dx
        residual = df.action(NS, w)
        jacobian = df.derivative(residual,  w)
        problem  = df.NonlinearVariationalProblem(residual, w, bcs, jacobian)
        solver   = df.NonlinearVariationalSolver(problem)
        solver.solve()
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
    temp_dataframe = create_csv_for_df(w, t)
    data_list.append(temp_dataframe)

dataframe_output_file = 'data/UnsteadyCase/' + formulation + '_' + testcase + '_unsteady_r'
output_dataframe = pd.concat(data_list)
output_dataframe.to_csv(dataframe_output_file + '.csv', index = False)