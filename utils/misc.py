import numpy as np

import scipy.integrate as integrate


def phi(N,s,rho,x_initial):
    num = 1 - np.exp(-2*N*s*x_initial / (2-rho))
    denom = 1 - np.exp(-2*N*s / (2-rho))
    return num/denom

def store_output(STORE_FIXATION_TIMES, parameters, s_range, fixation_counts, all_extinction_times, all_fixation_times, all_fixation_bools):
    num = len(s_range)
    nb_trajectories = parameters['nb_trajectories']
    if STORE_FIXATION_TIMES:
        output = {
                'parameters': parameters,
                's_range': list(s_range),
                'nb_fixations': list(fixation_counts),
                'all_extinction_times': list([[all_extinction_times[i,j] for i in range(num)] for j in range(nb_trajectories)]) ,
                'all_fixation_times': list([[all_fixation_times[i,j] for i in range(num)] for j in range(nb_trajectories)]),
                'all_fixation_bools': list([[all_fixation_bools[i,j] for i in range(num)] for j in range(nb_trajectories)])
            }
    else:
        output = {
                'parameters': parameters,
                's_range': list(s_range),
                'nb_fixations': list(fixation_counts)
            }

    return output

def compute_avg_fixation_time(N,M,s):
    """
    Well-mixed case, under the diffusion approximation
    """
    rho = M/N
    t = 2 * np.log(2*N*s/(2-rho)) / (s*rho)
    return t

def f_1(y,s,N,rho):
    num = (np.exp(2*s*N*y/(2-rho))-1) * (1 - np.exp(2*s*N * (1-y)/(2-rho)))
    denum = y * (1-y)
    return num / denum

def f_2(y,s,N,rho):
    num = (1 - np.exp(s*N*(1-y)/(2-rho))) * (1 - np.exp(-2*s*N*(1-y)/(2-rho)))
    denum = y * (1-y)
    return num / denum

def compute_avg_extinction_time(N,M,s):
    """
    Well-mixed case, under the diffusion approximation
    """
    rho = M/N
    frac1 = 1/(rho * s * (1 - np.exp(2*s*N*(2-rho))))
    frac2 = (1-np.exp(-2*s/(2-rho)))/(rho*s*(1-np.exp(-2*s*N*(2-rho))) * (1 - np.exp(2*s*(N-1)*(2-rho))))

    int1,_ = integrate.quad(lambda y: f_1(y,s,N,rho), 0, 1/N)
    int2,_ = integrate.quad(lambda y: f_2(y,s,N,rho), 1/N, 1)

    t = frac1 * int1 + frac2*int2
    return t