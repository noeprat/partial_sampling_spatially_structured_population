import numpy as np

from scipy.stats import binom

from numba import njit, prange, jit #, int_, float_



# useful functions


@jit
def simulate_graph(DG, nb_demes, N, M, s, tmax, initial_node=0):
    """
    Arguments:
     -> DG: ndArray(nb_demes, nb_demes), matrix representing a directed graph
     -> nb_demes: int
     -> N: int, number of individuals per deme
     -> M: int, number of updated individuals per deme
     -> s: float, relative fitness
     -> tmax: int, maximal number of iterations simulated
     -> (optional) initial_node: int = 0, node where the first mutant spawns
    Returns:
     -> fixation: int, 1 if fixation happened before tmax, 0 otherwise
     -> stop_time: int, number of iterations before fixation or extinction
    """
    
    b = True
    t = 1

    i_nodes = np.zeros(nb_demes)  # list of the number of mutants in each node
    i_nodes[initial_node] = 1
    #N_nodes = N * np.ones(nb_demes, dtype=int_)  # population size in each node
    #M_nodes = M * np.ones(nb_demes, dtype=int_)  # update size in each node

    # Creating a directed graph (migration rates)
    

    #trajectories = np.zeros((tmax, nb_demes))
    #trajectories[0, :] = i_nodes

    #while t < tmax and b:
    while b:

        i_nodes_before = i_nodes[::]
        for selected_node in range(nb_demes):

            # Hypergeometrical sampling
            ngood = i_nodes_before[selected_node]
            nbad = N - ngood

            nb_mutants_before_update = np.random.hypergeometric(ngood, nbad, M)

            # Binomial sampling
            x_tilde = sum([DG[k, selected_node]* (i_nodes_before[k] /N) for k in range(nb_demes)])
            #print('x_tilde:', x_tilde)
            prob =  x_tilde* (1+s) / (1 + s* x_tilde)

            if x_tilde >1 or x_tilde<0 :
                print('x_tilde',x_tilde)
                #print('prob', prob)
                #print('sum m_kj, should be equal to one:', sum([DG[k, selected_node] for k in range(nb_demes)]))
                #print('s',s)
            prob = max(min(prob, 1),0)  #clip probability to stay in [0,1]
            n_trials = M
            nb_mutants_after_update = np.random.binomial(n_trials, prob)
            #nb_mutants_after_update = binom.rvs(n_trials, prob)   #does not work with numba

            # Update mutants in the node
            i_nodes[selected_node] = ngood - nb_mutants_before_update + nb_mutants_after_update

        #trajectories[t, :] = i_nodes
        t += 1
        b = sum(i_nodes) < nb_demes*N and (i_nodes > 0).any()

    if sum(i_nodes) == nb_demes*N:
        fixation = 1
    else:
        fixation = 0
    
    stop_time = t  #should be < tmax

    # assert not b

    #if t < tmax:
        #for tt in range(t, tmax):
            #trajectories[tt, :] = trajectories[t - 1, :]

    return fixation, stop_time



@njit(parallel=True)
def simulate_multiple_trajectories_graph(DG, nb_demes, N, M, s, tmax, initial_node=0, nb_trajectories=100):
    """
    Multiple simulations with the same parameters
    Arguments:
     -> DG: ndArray(nb_demes, nb_demes), matrix representing a directed graph
     -> nb_demes: int
     -> N: int, number of individuals per deme
     -> M: int, number of updated individuals per deme
     -> s: float, relative fitness
     -> tmax: int, maximal number of iterations simulated
     -> (optional) initial_node: int = 0, node where the first mutant spawns
     -> (optional) nb_trajectories: int = 100, number of simulation runs
    Returns:
     -> count_fixation: int, number of fixations observed
     -> fixation_seq: ndArray(nb_trajectories), fixation_seq[i] = 1 if a fixation was observed at trajectory i, 0 otherwise
     -> fixation_times: ndArray(nb_trajectories), fixation_times[i] is the number of iterations before fixation (if it was observed) at trajectory i (0 otherwise)
     -> extinction_times: ndArray(nb_trajectories), extinction_times[i] is the number of iterations before extinction (if it was observed) at trajectory i (0 otherwise)
     -> count_runs: int, number of runs (debug, should be equal to nb_trajectories)
    """
    #all_trajectories = np.zeros((int(nb_trajectories), int(tmax)))
    fixation_seq = np.zeros(nb_trajectories)
    count_fixation = 0
    count_runs = 0
    fixation_times = np.zeros(nb_trajectories)
    extinction_times = np.zeros(nb_trajectories)

    for trajectory_index in prange(nb_trajectories):  #parallelized
        #print('trajectory:', trajectory_index)
        fixation, stop_time = simulate_graph(DG, nb_demes, N, M, s, tmax, initial_node)

        count_fixation += fixation
        fixation_seq[trajectory_index] = fixation
        if fixation == 1:
            fixation_times[trajectory_index] = stop_time
            extinction_times[trajectory_index] = np.inf
        else:
            extinction_times[trajectory_index] = stop_time
            fixation_times[trajectory_index] = np.inf

        count_runs += 1
        #fixation_seq[trajectory_index] = fixation
        #all_trajectories[trajectory_index, :] = np.sum(trajectories, axis=1)

    return count_fixation, fixation_seq, fixation_times, extinction_times, count_runs



def sweep_s_graph(DG, nb_demes, N, M, log_s_min, log_s_max, initial_node = 0, nb_trajectories = 100, tmax = 100000, num=10):
    """
    Sweeps a logspace interval of relative fitness values [10**log_s_min, 10**log_s_max]
    Arguments:
     -> DG: ndArray(nb_demes, nb_demes), matrix representing a directed graph
     -> nb_demes: int, number of demes / nodes in the graph
     -> N: int, number of individuals per deme
     -> M: int, number of updated individuals per deme
     -> log_s_min: int,
     -> log_s_max: int,
     -> (optional) initial_node: int = 0, node where the first mutant spawns
     -> (optional) nb_trajectories: int = 100, number of simulation runs
     -> (optional) tmax: int = 100000, maximal number of iterations simulated per trajectory
     -> (optional) num: int = 10, number of points in the interval of relative fitness values
    Returns:
     -> s_range: np.logspace(log_s_min, log_s_max, num=num), interval of s values
     -> fixation_counts: ndArray(num), fixation_counts[i] is the number of fixation counted for s = s_range[i]
     -> run_counts: ndArray(num), fixation_counts[i] is the number of runs counted for s = s_range[i]
     -> all_fixation_times: ndArray(num, nb_trajectories), all_fixation_times[s_index, traj_index] is the fixation time (if it was observed) at trajectory [traj_index] for s = s_range[s_index] (0 otherwise)
     -> all_extinction_times: ndArray(num, nb_trajectories), all_extinction_times[s_index, traj_index] is the extinction time (if it was observed) at trajectory [traj_index] for s = s_range[s_index] (0 otherwise)
     -> all_extinction_bools: ndArray(num, nb_trajectories), all_extinction_bools[s_index, traj_index] is 1 if fixation happened 0 otherwise
    """
    s_range = np.logspace(log_s_min, log_s_max, num=num)
    tmax = 100000

    fixation_counts = np.zeros_like(s_range)
    run_counts = np.zeros_like(s_range)
    all_extinction_times = np.zeros((num, nb_trajectories))
    all_fixation_times = np.zeros((num, nb_trajectories))
    all_fixation_bools = np.zeros((num, nb_trajectories))

    

    for i,s in enumerate(s_range):
        count_fixation, fixation_seq, fixation_times, extinction_times, count_runs = simulate_multiple_trajectories_graph(DG, nb_demes, N, M, s, tmax, initial_node, nb_trajectories)
        fixation_counts[i] = count_fixation
        
        if count_runs != nb_trajectories:
            print('missed runs')
            print('counted runs', count_runs)
            print('nb_trajectories', nb_trajectories)
            fixation_counts[i] = count_fixation * nb_trajectories / count_runs
        else:
            fixation_counts[i] = count_fixation
        
        all_fixation_times[i,:] = fixation_times[:]
        all_extinction_times[i,:] = extinction_times[:]
        all_fixation_bools[i,:] = fixation_seq[:]


        #mean_extinction_times[i] = sum(extinction_times)/(nb_trajectories - count_fixation)
        #if count_fixation > 0:
        #    mean_fixation_times[i] = sum(fixation_times)/count_fixation


    return s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools




