#!/usr/bin/env python

import cvxpy as cp
import numpy as np
import pdb  

from utils import *

def solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False):
    """
    Solves an SOCP of the form:

    minimize(h^T x)
    subject to:
        ||A_i x + b_i||_2 <= c_i^T x + d_i    for all i
        F x == g

    Args:
        x       - cvx variable.
        As      - list of A_i numpy matrices.
        bs      - list of b_i numpy vectors.
        cs      - list of c_i numpy vectors.
        ds      - list of d_i numpy vectors.
        F       - numpy matrix.
        g       - numpy vector.
        h       - numpy vector.
        verbose - whether to print verbose cvx output.

    Return:
        x - the optimal value as a numpy array, or None if the problem is
            infeasible or unbounded.
    """
    objective = cp.Minimize(h.T @ x)
    constraints = []
    for A, b, c, d in zip(As, bs, cs, ds):
        constraints.append(cp.SOC(c.T @ x + d, A @ x + b))
    constraints.append(F @ x == g)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    if prob.status in ['infeasible', 'unbounded']:
        return None

    return x.value

def grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext):
    """
    Solve the grasp force optimization problem as an SOCP. Handles 2D and 3D cases.

    Args:
        grasp_normals   - list of M surface normals at the contact points, pointing inwards.
        points          - list of M grasp points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).
        wrench_ext      - external wrench applied to the object.

    Return:
        f
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)
    transformations = [compute_local_transformation(n) for n in grasp_normals]

    ########## Your code starts here ##########
    As = []
    bs = []
    cs = []
    ds = []

    x_size = D*M+1
    x = cp.Variable(x_size)



    h = np.zeros((x_size))
    h[-1] = 1

    #first half inequality constraints
    #A
    j = 0
    while j+D <= x_size:
        A = np.zeros((D, x_size))
        begin = j
        end = j + D
        ran = list(range(begin, end))
        A[np.arange(D),ran] = 1
        j = end
        As.append(A)
    
    for _ in range(M):
        #b
        b = np.zeros((D))
        bs.append(b)

        #c
        c = np.zeros((x_size))
        c[-1] = 1
        cs.append(c)

        #d 
        d = np.reshape(np.array([0]), (1,1))
        ds.append(d)
    
    #second half inequality constraints
    #A
    k = 0
    while k+(D-1) <= x_size:
        A = np.zeros((D-1, x_size))
        begin = k
        end = k + D -1
        ran = list(range(begin, end))
        A[np.arange(D-1),ran] = 1
        k = end
        As.append(A)
    
    for i in range(M):
        #b
        b = np.zeros((D-1))
        bs.append(b)

        #c
        curr_idx = D*(i+1) -1
        c = np.zeros((x_size))
        c[curr_idx] = friction_coeffs[i]
        cs.append(c)

        #d 
        d = np.reshape(np.array([0]), (1,1))
        ds.append(d)

    #constructing F matrix

    F = np.zeros((N, x_size))
    t_all = transformations[0]
    for i in range(1, len(transformations)):
        t_all = np.hstack((t_all, transformations[i]))
    t_all = np.hstack((t_all, np.zeros((D,1))))

    pt_all = cross_matrix(points[0]) @ transformations[0]
    for i in range(1, len(transformations)):
        pt_all = np.hstack((pt_all, cross_matrix(points[i]) @ transformations[i]))
    pt_all = np.hstack((pt_all, np.zeros((N-D,1))))


    F = np.vstack((t_all, pt_all))
    g = -1*wrench_ext



    x = solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False)

    # TODO: extract the grasp forces from x as a stacked 1D vector
    f = x[:-1]

    ########## Your code ends here ##########

    # Transform the forces to the global frame
    F = f.reshape(M,D)
    forces = [T.dot(f) for T, f in zip(transformations, F)]

    return forces

def precompute_force_closure(grasp_normals, points, friction_coeffs):
    """
    Precompute the force optimization problem so that force closure grasps can
    be found for any arbitrary external wrench without redoing the optimization.

    Args:
        grasp_normals   - list of M contact normals, pointing inwards from the object surface.
        points          - list of M contact points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).

    Return:
        force_closure(wrench_ext) - a function that takes as input an external wrench and
                                    returns a set of forces that maintains force closure.
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)

    ########## Your code starts here ##########
    # TODO: Precompute the optimal forces for the 12 signed unit external
    #       wrenches and store them as rows in the matrix F. This matrix will be
    #       captured by the returned force_closure() function.
    F = np.zeros((2*N, M*D))
    #print(N)
    #print(M)
    
    for i in range(2*N):
        print(i)
        curr_wrench = np.zeros((N))
        curr_wrench[i//2] = (-1)**i
        print(curr_wrench)
        forces = grasp_optimization(grasp_normals, points, friction_coeffs, curr_wrench)
        forces_all = forces[0]
        for j in range(1, len(forces)):
            forces_all = np.hstack((forces_all, forces[j]))
        F[i, :] = forces_all
    #print(F)
    import pdb
    pdb.set_trace()



    ########## Your code ends here ##########

    def force_closure(wrench_ext):
        """
        Return a set of forces that maintain force closure for the given
        external wrench using the precomputed parameters.

        Args:
            wrench_ext - external wrench applied to the object.

        Return:
            f - grasp forces as a list of M numpy arrays.
        """

        ########## Your code starts here ##########
        # TODO: Compute the force closure forces as a stacked vector of shape (M*D)
        f = np.zeros(M*D)

  
        ########## Your code ends here ##########

        forces = [f_i for f_i in f.reshape(M,D)]
        return forces

    return force_closure
