import numpy as np
from pyscipopt import Model
from dynamics import unicycle_ddt
from agents import AgentGroup, DefaultTargetParams
from utils import rollout
import matplotlib.pyplot as plt




def generate_assignments(targets_states, target_params, tracker_position):
    position_rollouts = []
    times = []
    #for target in targets.agent_list:
    for target in targets_states:
        ts, ps = rollout(target, target_params, 50, 0.1)
        position_rollouts.append(ps)
        times.append(ts)

    positions_sampled = [0] * len(position_rollouts)
    times_sampled = [0] * len(times)
    for ix in range(len(times_sampled)):
        samples = sample_indices(times[ix], position_rollouts[ix], np.ones(len(times)))
        positions_sampled[ix] = position_rollouts[ix][samples, :]
        times_sampled[ix] = times[ix][samples]

   
    path = None 
    vmax_base = 5
    for ix in range(1,10):
        dag_weight_matrix, partitions = build_pdag(times_sampled, positions_sampled, tracker_position, vmax_base*ix)

        n = dag_weight_matrix.shape[0]
        node_weights = -100*np.ones(n)
        path = pdag_min_paths(dag_weight_matrix, node_weights, partitions, 1)
        print(targets_states)
        if path is not None:
            path = path[0]
            print(path)
            break
        else:
            print('Error: Cannot reach targets')
    
    offsets = np.cumsum([0, 1] + [len(tl) for tl in times_sampled])
    path_positions = []
    path_assignments = []
    for ix in path:
        ix1, ix2 = index_to_lol(ix, offsets)
        path_positions.append(positions_sampled[ix1-1][ix2])
        path_assignments.append(ix1 - 1)

    return (path_assignments, path_positions)


def sample_indices(times, positions, costs):
    """ Sample (time, position) pairs, maybe weighted according to costs 

        For now, we ignore costs and sample with linspace for testing

        Args:
            times: numpy array length n
            positions: nx2 numpy array
            costs: numpy array length n
    """
    n = len(times)
    n_sample = 10.
    stepsize = int(n / n_sample)
    return np.arange(1,n,stepsize)

def index_to_lol(ix, offsets):
    ix1 = np.where(ix < offsets)[0][0] - 1
    ix2 = ix - offsets[ix1]
    assert(lol_to_index(ix1, ix2, offsets) == ix)
    return ix1, ix2

def lol_to_index(ix1, ix2, offsets):
    return offsets[ix1] + ix2 

def build_pdag(times_lists, positions_lists, current_position, vmax):
    """ Build Partitioned DAG from a list of times and positions for each agent

        Args:
            times_lists: A list of lists, representing the timings the sampled locations for each agent
            positions_lists: A list of lists, representing the sampled locations for each agent
            current_position: Current tracker position
    """

    positions_lists_new = [current_position] + positions_lists
    times_lists_new = [np.array([0])] + times_lists
    n = sum([len(tl) for tl in times_lists_new])

    weight_matrix = np.ones((n + 1, n + 1)) * np.inf
    added_vertices = []
    partitions = []

    offsets = np.cumsum([0] + [len(tl) for tl in times_lists_new])
    get_node_ix = lambda tl_ix, ix: lol_to_index(tl_ix, ix, offsets)

    for tl_ix in range(len(times_lists_new)):
        if tl_ix > 0:
            partition = []
        for ix in range(len(times_lists_new[tl_ix])):
            current_node_ix = get_node_ix(tl_ix, ix)
            if tl_ix > 0:
                partition.append(current_node_ix)
            for (cmp_ix, cmp_time, cmp_loc, partition_ix) in added_vertices:
                if tl_ix == partition_ix:
                    continue
                dist = np.linalg.norm(cmp_loc - positions_lists_new[tl_ix][ix])
                dt = times_lists_new[tl_ix][ix] - cmp_time
                vel = dist / abs(dt)
                if vel > vmax:
                    continue
                if dt < 0:
                    weight_matrix[get_node_ix(tl_ix, ix), cmp_ix] = dist#-1
                elif dt > 0:
                    weight_matrix[cmp_ix, get_node_ix(tl_ix, ix)] = dist#-1

            added_vertices.append((get_node_ix(tl_ix, ix), times_lists_new[tl_ix][ix], positions_lists_new[tl_ix][ix], tl_ix))
        if tl_ix > 0:
            partitions.append(partition)

    return weight_matrix, partitions

def pdag_min_paths(edge_weight_matrix, node_weights, partitions, n):
    """ Compute minimum paths in a partitioned DAG

        Args:
            edge_weight_matrix: matrix representing DAG. Assumes (r -> c).
            node_weights: list of weights for each node in the DAG.
            partitions: list of lists, each representing the indices in a partition.

        Returns:
            
    """

    model = Model("PDAG-Min-Paths")
    model.setRealParam('limits/time', 2)
    model.setIntParam('presolving/maxrounds', 0) # -1 is unlimited rounds
    # variable for each edge with cost less than np.inf
    # for variable (i -> j), add cost: (i,j) + j
    # 

    (nr, nc) = edge_weight_matrix.shape
    # first n indices correspond to path "heads"

    #edge_weights_modified = np.zeros((nr + n, nc + n))
    #edge_weights_modified[:-n, :-n] = edge_weight_matrix
    #edge_weights_modified[-n:, :] = 0 # no cost connecting dummy nodes to real nodes
    #edge_weights_modified[:, -n:] = np.inf # cannot connect real node to dummy node
    #(nr_extended, nc_extended) = edge_weights_modified.shape

    
    variable_array = np.zeros(edge_weight_matrix.shape, dtype=object)
    variable_mask = np.zeros(variable_array.shape, dtype=bool)
    objective = 0
    for r in range(nr):
        for c in range(nc):
            if r == c:
                continue
            if edge_weight_matrix[r,c] < np.inf:
                variable_array[r,c] = model.addVar(vtype="I", name="%d,%d" % (r, c))
                variable_mask[r,c] = True
                objective += variable_array[r,c] * (edge_weight_matrix[r,c] + node_weights[c])

                model.addCons(variable_array[r,c] >= 0)
                model.addCons(variable_array[r,c] <= 1)

    model.setObjective(objective, 'minimize')

    #import matplotlib.pyplot as plt
    #mat = np.zeros(variable_mask.shape)
    #mat[variable_mask] = edge_weight_matrix[variable_mask]
    #plt.imshow(mat)
    #plt.show()

    for p in partitions:
        if np.any(variable_mask[:,p]):
            incoming_cons = np.sum(variable_array[:, p])
            model.addCons(incoming_cons <= 1) # incoming per partition <= 1
            model.addCons(np.sum(variable_array[p, :]) <= np.sum(variable_array[:, p])) # outgoing per partition <= incoming
        else:
            return None
        

    for c in range(n, nc):
        if np.any(variable_mask[:, c]) or np.any(variable_mask[c, :]):
            model.addCons(np.sum(variable_array[:, c]) >= np.sum(variable_array[c, :])) # incoming - outgoing consistency

    for r in range(n):
        if np.any(variable_mask[r,:]):
            model.addCons(np.sum(variable_array[r, :]) <= 1) # the "initial" positions have at most one outgoing edge
        else:
            return None


    model.optimize()
    val = model.getObjVal()

    #for r in range(nr):
    #    for c in range(nc):
    #        if variable_mask[r, c]:
    #            print((r,c), model.getVal(variable_array[r, c]))

    heads = []
    for d in range(n):
        for c in range(nc):
            if variable_mask[d, c]:
                val = model.getVal(variable_array[d, c])
                if val > .999 and val < 1.001:
                    heads.append(c)
    paths = []
    for h in heads:
        paths.append(chase_path(h, model, variable_array, variable_mask))

    return paths


def chase_path(h, model, va, vm):
    path = [h]
    n = va.shape[0]
    found_it = True
    while found_it:
        found_it = False
        for c in range(n):
            if vm[h, c]:
                val = model.getVal(va[h, c])
                if val > .999 and val < 1.001:
                    path.append(c)
                    h = c
                    found_it = True
                    break
                
    return path


if __name__ == '__main__':
    tracker_position = np.array([-4, 1])

    targets = AgentGroup(5, [-5,-5,-5], [5,5,5], DefaultTargetParams())

    (path_assignment, path_positions) = generate_assignments(targets.unicycle_state, tracker_position)
    print(path_assignment, path_positions)


    xvals = [arr[0] for arr in path_positions]
    yvals = [arr[1] for arr in path_positions]


    target_x = [t.unicycle_state[0] for t in targets.agent_list]
    target_y = [t.unicycle_state[1] for t in targets.agent_list]

    plt.scatter(target_x, target_y)

    plt.plot(xvals, yvals)
    plt.show()
