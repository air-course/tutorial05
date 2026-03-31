import numpy as np
import sympy as sym
def obtain_matrix_form_EL(Tt, Vt, q, dq, t):
    '''
    Uses Euler-Lagrange to obtain nonlinear equations of motion in matrix form:
        M(q)ddq + C(q, dq) + V(q) = 0 
    Note that damping is not included, since EL does not work for nonconservative forces
    Input:
        - Tt: Total kinetic energy
        - Vt: Total potential energy
        - q: Dynamic variables(matrix form)
        - q: Dynamic variables (matrix form)
    Returns: M, C, V
    '''
    n = q.shape[0]
    # Use the Euler-Lagrange equations to obtain the system EOMs
    L_sym = Tt - Vt

    # Gradient w.r.t q
    dL_dq = sym.Matrix([L_sym.diff(qi) for qi in q])
    # Gradient w.r.t dq
    dL_ddq = sym.Matrix([L_sym.diff(dqi) for dqi in dq])
    # Time derivative
    d_dt_dL_ddq = sym.Matrix([expr.diff(t) for expr in dL_ddq])
    # Mass matrix
    M = sym.simplify(sym.hessian(Tt, dq))
    # Coriolis & Centrifugal matrix through christoffel symbols
    C = sym.Matrix.zeros(n, n)
    for i in range(sym.shape(C)[0]):
        for j in range(sym.shape(C)[1]):
            for k in range(len(q)):
                dMij_dqk = M[i, j].diff(q[k])
                dMik_dqj = M[i, k].diff(q[j])
                dMkj_dqi = M[k, j].diff(q[i])

                C[i, j] += 0.5*(dMij_dqk + dMik_dqj + dMkj_dqi)*dq[k]
    # Potential force
    V = sym.simplify(sym.Matrix([Vt]).jacobian(q).T)
    
    return M, C, V