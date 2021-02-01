import numpy as np
from scipy.spatial.distance import cdist
from concorde.tsp import TSPSolver

def solve_tsp(positions, current_position, fn):

    n_orig = positions.shape[0]
    # dummy node for current location
    positions_current_dummy = np.vstack([current_position, positions])

    # dummy nodes for all other locations
    positions_dummy_full = np.vstack([positions_current_dummy, positions_current_dummy])
    # now rows 0-20 represent the "incoming" distances, 21-41 represent "outgoing" distances

    # costs -- high cost for dummy - non dummy, hight costs coming into starting node

    edm = cdist(positions_dummy_full, positions_dummy_full)
    
    n_full = edm.shape[0]

    penalty = 100
    edm[:n_orig+1,:n_orig+1] = penalty
    edm[n_orig+1:,n_orig+1:] = penalty
    edm[np.arange(n_full), np.arange(n_full)] = 0
    edm[0, n_orig + 2 + np.arange(n_orig)] = penalty
    edm[n_orig + 2 + np.arange(n_orig), 0] = penalty

    edm_extended = np.zeros((n_full + 2, n_full + 2))
    edm_extended[:n_full, :n_full] = edm

    edm_extended[n_full, :n_orig+1] = penalty
    edm_extended[:n_orig+1, n_full] = penalty

    edm_extended[n_full + 1, 1:-2] = penalty
    edm_extended[1:-2, n_full + 1] = penalty

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(edm_extended)
    #plt.show()

    write_tsp(edm_extended, fn)
    solver = TSPSolver.from_tspfile(fn)
    (sol, val, success, found_tour, hit_timebound) = solver.solve()
    sol_rewrapped = np.roll(sol, -np.argmin(sol))
    if sol_rewrapped[1] == n_full+1:
        # reverse
        sol_corrected = sol_rewrapped[4:-1:2] - 1
        sol_corrected = sol_corrected[::-1]
    else:
        # not reversed
        sol_corrected = sol_rewrapped[2:-2:2] - 1
    
    return sol_corrected


def write_tsp(mat, fn):
    header = 'NAME: tracking\nTYPE: TSP\nDIMENSION: %d\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n' % mat.shape[0]
    with open(fn, 'w') as fo:
        fo.write(header)
        nr, nc = mat.shape
        for r in range(nr):
            for c in range(nc):
                fo.write(' %d' % (1000 * mat[r, c]))
            fo.write('\n')
        fo.write('EOF')
            
