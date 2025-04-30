import numpy as np

from numba import jit #, int_, float_


@jit
def generate_clique_graph(nb_demes, migration_rate):
    assert 1 - (nb_demes - 1) * migration_rate >= 0
    DG = np.zeros((nb_demes, nb_demes))
    for node1 in range(nb_demes):
        for node2 in range(nb_demes):
            if node1 == node2:
                weight = 1 - (nb_demes - 1) * migration_rate
            else:
                weight = migration_rate
            DG[node1, node2] = weight
    return DG

@jit
def generate_cycle_graph(nb_demes, migration_rate, alpha):
    # creating a directed graph
    assert 1 - (1 + alpha)*migration_rate >= 0
    DG = np.zeros((nb_demes, nb_demes), dtype=float)

    for node in range(nb_demes):
        next_node = (node + 1) % nb_demes
        prev_node = (node - 1) % nb_demes

        # next edge
        DG[node, next_node] = migration_rate

        # prev edge
        DG[node, prev_node] = alpha*migration_rate

        # loop
        DG[node, node] = 1 - (1+alpha)*migration_rate
    return DG

@jit
def generate_star_graph(nb_demes, migration_rate, alpha):
    assert 1 - (nb_demes-1)*alpha*migration_rate >= 0
    assert 1 - migration_rate >= 0
    DG = np.zeros((nb_demes, nb_demes), dtype=float)


    #adding weighted edges for the center of the star


    for node in range(1,nb_demes):
        # outward edges
        DG[0, node]= migration_rate

        # inward edges
        DG[node, 0] = alpha*migration_rate

        # loops
        DG[node, node] = 1 - migration_rate
    
    DG[0, 0] = 1 -(nb_demes - 1)*alpha*migration_rate
    return DG


@jit
def generate_line_graph(nb_demes, migration_rate, alpha):
    assert 1 - (1+alpha)*migration_rate >= 0
    DG = np.zeros((nb_demes, nb_demes), dtype=float)

    # First node

    DG[0,0] = 1 - migration_rate

    DG[0,1] = alpha*migration_rate

    # Last node

    DG[nb_demes - 1, nb_demes - 1] = 1 - alpha * migration_rate

    DG[nb_demes - 1, nb_demes - 2] = migration_rate

    for i in range(1,nb_demes-1):
        # edge to the right
        DG[i, i + 1] = alpha * migration_rate

        # edge to the left
        DG[i, i - 1] = migration_rate

        # loops
        DG[i, i] = 1 - migration_rate * (1+alpha)

    return DG

