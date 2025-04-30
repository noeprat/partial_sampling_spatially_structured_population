import numpy as np

from scipy.stats import binom

from numba import njit, jit, prange


# useful functions

@jit
def simulate_trajectory(N, M, s, tmax, initial_state=1):
    """
    Arguments:
     -> N: int, number of individuals
     -> M: int, number of updated individuals
     -> s: float, relative fitness
     -> tmax: int, maximal number of iterations simulated
     -> (optional) initial_state: int = 1, nb of mutants at initial state
    Returns:
     -> fixation: int, 1 if fixation happened before tmax, 0 otherwise
     -> stop_time: int, number of iterations before fixation or extinction
    """
    #states = np.zeros(tmax)
    #states[0] = initial_state
    current_state = initial_state
    t=1
    b = True
    #while t<tmax and b:
    while b:
        #perform hypergeometrical sampling
        nb_mutants_before_update = np.random.hypergeometric(current_state, N - current_state, M)

        # perform binomial sampling 
        x = current_state / N
        prob = x*(1+s) / (1+x*s)
        if prob >1 or prob <0:
                print('prob',prob)
                print('s',s)
                prob = min(1, prob)
                prob = max(prob,0)
        n_trials = M
        #nb_mutants_after_update = np.random.binomial(n_trials, prob)
        nb_mutants_after_update = np.random.binomial(n_trials, prob)
        #nb_mutants_after_update = binom.rvs(n_trials, prob)   #does not work with numba
        
        if M==1: #Moran case
            rand = np.random.rand()
            if rand <= x*(1-prob) and current_state:
                current_state -= 1
            elif rand <= x*(1-prob) + prob*(1-x):
                current_state += 1
        
        if M > 1: 
            current_state = current_state - nb_mutants_before_update + nb_mutants_after_update


        b = 0 < current_state and current_state < N
        t+=1
        #states[t] = current_state
    
    #if t<tmax:
        #states[t+1:] = current_state
    
    stop_time = t

    if current_state == N:
        fixation = 1
    else:
        fixation = 0
    return fixation, stop_time

@njit(parallel=True)
def simulate_multiple_trajectories(N, M, s, tmax, nb_trajectories=100, initial_state=1):
    """
    Multiple simulations with the same parameters
    Arguments:
     -> N: int, number of individuals per deme
     -> M: int, number of updated individuals per deme
     -> s: float, relative fitness
     -> tmax: int, maximal number of iterations simulated
     -> (optional) nb_trajectories: int = 100, number of simulation runs
     -> (optional) initial_node: int = 1, nb of initial mutants
    Returns:
     -> count_fixation: int, number of fixations observed
     -> fixation_seq: ndArray(nb_trajectories), fixation_seq[i] = 1 if a fixation was observed at trajectory i, 0 otherwise
     -> fixation_times: ndArray(nb_trajectories), fixation_times[i] is the number of iterations before fixation (if it was observed) at trajectory i (0 otherwise)
     -> extinction_times: ndArray(nb_trajectories), extinction_times[i] is the number of iterations before extinction (if it was observed) at trajectory i (0 otherwise)
     -> count_runs: int, number of runs (debug, should be equal to nb_trajectories)
    """
    #all_trajectories = np.zeros((int(nb_trajectories),int(tmax)))
    #all_trajectories[:,0] = initial_state

    #fixation_seq = np.zeros(nb_trajectories, dtype=bool)

    fixation_seq = np.zeros(nb_trajectories)
    count_fixation = 0
    count_runs = 0
    fixation_times = np.zeros(nb_trajectories)
    extinction_times = np.zeros(nb_trajectories)


    #for trajectory_index in tqdm(range(nb_trajectories)):
    for trajectory_index in prange(nb_trajectories):
        #if trajectory_index%(10**3) == 0:
            #print('trajectory:', trajectory_index)
        fixation, stop_time = simulate_trajectory(N,M,s,tmax,initial_state)

        count_fixation += fixation

        fixation_seq[trajectory_index] = fixation

        if fixation == 1:
            fixation_times[trajectory_index] = stop_time
            extinction_times[trajectory_index] = np.inf
        else:
            extinction_times[trajectory_index] = stop_time
            fixation_times[trajectory_index] = np.inf
        count_runs += 1        
        

    return count_fixation, fixation_seq, fixation_times, extinction_times, count_runs



def sweep_s_wm_sim(N, M, log_s_min, log_s_max, initial_state = 1, nb_trajectories = 100, tmax = 100000, num=10):
    """
    Sweeps a logspace interval of relative fitness values [10**log_s_min, 10**log_s_max]
    Arguments:
     -> N: int, number of individuals per deme
     -> M: int, number of updated individuals per deme
     -> log_s_min: int,
     -> log_s_max: int,
     -> (optional) initial_state: int = 1, number of initial mutants
     -> (optional) nb_trajectories: int = 100, number of simulation runs
     -> (optional) tmax: int = 100000, maximal number of iterations simulated per trajectory
     -> (optional) num: int = 10, number of points in the interval of relative fitness values
    Returns:
     -> s_range: np.logspace(log_s_min, log_s_max, num=num), interval of s values
     -> fixation_counts: ndArray(num), fixation_counts[i] is the number of fixation counted for s = s_range[i]
     -> all_fixation_times: ndArray(num, nb_trajectories), all_fixation_times[s_index, traj_index] is the fixation time (if it was observed) at trajectory [traj_index] for s = s_range[s_index] (infinity otherwise)
     -> all_extinction_times: ndArray(num, nb_trajectories), all_extinction_times[s_index, traj_index] is the extinction time (if it was observed) at trajectory [traj_index] for s = s_range[s_index] (infinity otherwise)
     -> all_extinction_bools: ndArray(num, nb_trajectories), all_extinction_bools[s_index, traj_index] is 1 if fixation happened at trajectory [traj_index] for s = s_range[s_index] 0 otherwise
    """
    s_range = np.logspace(log_s_min, log_s_max, num=num)
    tmax = 100000

    fixation_counts = np.zeros_like(s_range)
    all_extinction_times = np.zeros((num, nb_trajectories))
    all_fixation_times = np.zeros((num, nb_trajectories))
    all_fixation_bools = np.zeros((num, nb_trajectories))

    

    for i,s in enumerate(s_range):
        count_fixation, fixation_seq, fixation_times, extinction_times, count_runs = simulate_multiple_trajectories(N, M, s, tmax, nb_trajectories, initial_state)
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










