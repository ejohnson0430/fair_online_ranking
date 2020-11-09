import numpy as np


def group_exposures(batch, num_groups):
    exposures = np.zeros([num_groups,2])
    for j in range(num_groups):       
        inds = batch[:,1]==j
        exposures[j,1] = np.sum(inds)
        exposures[j,0] = np.sum(1 / np.log2(1 + batch[inds, 0]))
    return exposures

def append_exposures(batch, past_exposures):
    return past_exposures + group_exposures(batch, past_exposures.shape[0])

def get_mean_exposures(past_exposures):
    return past_exposures[:,0] / np.maximum(past_exposures[:,1], 1)

def append_to_ranking(ranking, j, rel):
    return np.vstack((ranking, np.array([len(ranking) + 1, j, rel])))