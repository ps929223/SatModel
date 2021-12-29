import numpy as np


'3.1 Linear case'
def Lin3dvar(ub,w,,H,R,B,opt):
    # The solution of the 3DVAR problem in the linear case requires
    # the solution of linear system of equations
    # Here, we utilize the built-in numpy function to do this.
    # Other schemes can be used, instead.

    # A = [[1, 2],    B = [[11, 12],
    #      [3, 4]]         [13, 14]]
    # Element-wise multiplication will yield:
    #
    # A * B = [[1 * 11,   2 * 12],
    #          [3 * 13,   4 * 14]]
    # Matrix multiplication will yield:
    #
    # A @ B  =  [[1 * 11 + 2 * 13,   1 * 12 + 2 * 14],
    #            [3 * 11 + 4 * 13,   3 * 12 + 4 * 14]]

    if opt == 1: # model-space approach
        Bi = np.linalg.inv(B)
        Ri = np.linalg.inv(R)
        A = Bi + (H.T)@Ri@H
        b = Bi@ub + (H.T)@Ri@w
        ua = np.linalg.solve(A,b) # solve a linear system

    elif opt == 2: # model-space incremental approach
        Bi = np.linalg.inv(B)
        Ri = np.linalg.inv(R)
        A = Bi + (H.T)@Ri@H
        b = (H.T)@Ri@(w-H@ub)
        ua = ub + np.linalg.solve(A,b) # solve a linear system

    elif opt == 3: # observation-space incremental approach
        A = R + H@B(H.T)
        b = (w-H@ub)
        ua = ub + B@(H.T)@np.linalg.solve(A,b)

    return ua



'3.2 Nonlinear Case'
def NonLin3dvar(ub,w,,ObsOp,JObsOp,R,B):
    # The solution of the 3DVAR problem in the nonlinear case requires
    # the solution of linear system of equations
    # Here, we utilize the built-in numpy function to do this.
    # Other schemes can be used, instead.

    Bi = np.linalg.inv(B)
    Ri = np.linalg.inv(R)
    ua = np.copy(ub)

    for iter in range(100):
        Dh = JObsOp(ua)
        A = Bi + (Dh.T)@Ri@Dh
        b = Bi@(ub-ua) + (Dh.T)@Ri@(w-ObsOp(ua))
        du = np.linalg.solve(A,b) # solve a linear system
        ua = ua + du

    if np.linalg.norm(du) <= 1e-4:
        break

    return ua


