import cvxpy as cp
import numpy as np

def cross_matrix(x):
    """
    Returns a matrix x_cross such that x_cross.dot(y) represents the cross
    product between x and y.

    For 3D vectors, x_cross is a 3x3 skew-symmetric matrix. For 2D vectors,
    x_cross is a 2x1 vector representing the magnitude of the cross product in
    the z direction.
     """
    D = x.shape[0]
    if D == 2:
        return np.array([[-x[1], x[0]]])
    elif D == 3:
        return np.array([[0., -x[2], x[1]],
                         [x[2], 0., -x[0]],
                         [-x[1], x[0], 0.]])
    raise RuntimeError("cross_matrix(): x must be 2D or 3D. Received a {}D vector.".format(D))

def wrench(f, p):
    """
    Computes the wrench from the given force f applied at the given point p.
    Works for 2D and 3D.

    Args:
        f - 2D or 3D contact force.
        p - 2D or 3D contact point.

    Return:
        w - 3D or 6D contact wrench represented as (force, torque).    
    """
    ########## Your code starts here ##########
    # Hint: you may find cross_matrix(x) defined above helpful. This should be one line of code.
    #print(f.shape)
    w = np.hstack((f, cross_matrix(p) @ f))

    ########## Your code ends here ##########

    return w

def cone_edges(f, mu):
    """
    Returns the edges of the specified friction cone. For 3D vectors, the
    friction cone is approximated by a pyramid whose vertices are circumscribed
    by the friction cone.

    In the case where the friction coefficient is 0, a list containing only the
    original contact force is returned.

    Args:
        f - 2D or 3D contact force.
        mu - friction coefficient.

    Return:
        edges - a list of forces whose convex hull approximates the friction cone.
    """
    # Edge case for frictionless contact
    if mu == 0.:
        return [f]
    # Planar wrenches
    D = f.shape[0]
    if D == 2:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 2
        randvec1 = np.ones((D))
        loc1 = cross_matrix(f).reshape((D,)) #@ randvec1
        loc2 = -loc1
        # print(f)
        # print(randvec1)
        # print(loc1)
        # print(loc1.shape)
        # print(cross_matrix(f).shape)

        e1 = loc1*(1.0/np.linalg.norm(loc1)) * (mu*np.linalg.norm(f)) + f
        e2 = loc2*(1.0/np.linalg.norm(loc2)) * (mu*np.linalg.norm(f)) + f
        edges[0] = e1
        #print(edges)
        edges[1] = e2

        #print(edges)


        ########## Your code ends here ##########

    # Spatial wrenches
    elif D == 3:
        ########## Your code starts here ##########
        edges = [np.zeros(D)] * 4
        randvec1 = np.ones((D))
        loc1 = cross_matrix(f) @ randvec1
        loc2 = cross_matrix(f) @ loc1
        loc3 = -loc1.copy()
        loc4 = -loc2.copy()

        #rescaling_factor = 1.0*/np.linalg.norm(f)

        e1 = loc1*(1.0/np.linalg.norm(loc1)) * (mu*np.linalg.norm(f)) + f
        e2 = loc2*(1.0/np.linalg.norm(loc2)) * (mu*np.linalg.norm(f)) + f
        e3 = loc3*(1.0/np.linalg.norm(loc3)) * (mu*np.linalg.norm(f)) + f
        e4 = loc4*(1.0/np.linalg.norm(loc4)) * (mu*np.linalg.norm(f)) + f

        edges[0] = e1
        edges[1] = e2
        edges[2] = e3
        edges[3] = e4

        
        ########## Your code ends here ##########

    else:
        raise RuntimeError("cone_edges(): f must be 3D or 6D. Received a {}D vector.".format(D))

    return edges

def form_closure_program(F):
    """
    Solves a linear program to determine whether the given contact wrenches
    are in form closure.

    Args:
        F - matrix whose columns are 3D or 6D contact wrenches.

    Return:
        True/False - whether the form closure condition is satisfied.
    """
    ########## Your code starts here ##########
    # Hint: you may find np.linalg.matrix_rank(F) helpful
    # TODO: Replace the following program (check the cvxpy documentation)
    num_dim = F.shape[0]
    num_wrenches = F.shape[1]

    # zeros = np.zeros((num_dim,))
    # ones = np.ones((num_wrenches,))
    k = cp.Variable(num_wrenches)
    objective = cp.Minimize(cp.sum(k))
    constraints = [F @ k == 0]

    for i in range(num_wrenches):
        constraints.append(k[i] >= 1)


    ########## Your code ends here ##########

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.ECOS)

    return prob.status not in ['infeasible', 'unbounded']

def is_in_form_closure(normals, points):
    """
    Calls form_closure_program() to determine whether the given contact normals
    are in form closure.

    Args:
        normals - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.

    Return:
        True/False - whether the forces are in form closure.
    """
    ########## Your code starts here ##########
    # TODO: Construct the F matrix (not necessarily 6 x 7)
    wrench_dim = 0 
    if normals[0].shape[0] == 2:
        wrench_dim = 3
    elif normals[0].shape[0] == 3:
        wrench_dim = 6
    else:
        raise Exception("Normals need to be 2D or 3D")

    num_points = len(points)


    F = np.zeros((wrench_dim,num_points))
    for i in range(num_points):
        F[:, i] = wrench(normals[i], points[i])


    


    ########## Your code ends here ##########

    return form_closure_program(F)

def is_in_force_closure(forces, points, friction_coeffs):
    """
    Calls form_closure_program() to determine whether the given contact forces
    are in force closure.

    Args:
        forces - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.
        friction_coeffs - list of friction coefficients.

    Return:
        True/False - whether the forces are in force closure.
    """
    ########## Your code starts here ##########
    # TODO: Call cone_edges() to construct the F matrix (not necessarily 6 x 7)
    D = forces[0].shape[0]
    num_forces = len(forces)
    wrench_dim = 0
    col_dim = 0
    num_edges_per_cone =0

    if D == 2:
        wrench_dim = 3
        num_edges_per_cone = 2
        #col_dim = num_edges_per_cone*num_forces
    elif D == 3:
        wrench_dim = 6  
        num_edges_per_cone = 4
        #col_dim = num_edges_per_cone*num_forces
    else:
        raise Exception("D either 2 or 3 dimension")
    
    col_dim = 0
    for coeff in friction_coeffs:
        if coeff != 0:
            col_dim += num_edges_per_cone
        else:
            col_dim += 1

    F = np.zeros((wrench_dim, col_dim))
    print(col_dim)
    print(friction_coeffs)
    curr_col = 0
    for i in range(num_forces):
        curr_force = forces[i]
        curr_point = points[i]
        curr_coeff = friction_coeffs[i]
        edges = cone_edges(curr_force, curr_coeff)
        for j, edge_force in enumerate(edges):
            F[:, curr_col] = wrench(edge_force, curr_point) 
            curr_col += 1

    


    ########## Your code ends here ##########

    return form_closure_program(F)
