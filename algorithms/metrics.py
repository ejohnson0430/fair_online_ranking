import numpy as np


def nDCG(batch):
    ranks = batch[:,0]
    relevance = batch[:,2]
    relevance_sorted = np.sort(relevance)[::-1]
    num_actual = np.power(2,relevance)-1
    num_optimal = np.power(2,relevance_sorted)-1
    dcg_actual = np.sum(num_actual/np.log2(ranks+1))
    dcg_optimal = np.sum(num_optimal/np.log2(np.arange(len(ranks))+2))
    return dcg_actual / dcg_optimal

def DDP(past_exposures, min_count=5):
    # return largest difference in mean exposures
    included = past_exposures[:,1] >= min_count
    if np.sum(included) < 2: return 0
    mean_exposures = past_exposures[included,0] / past_exposures[included,1]
    return np.max(mean_exposures) - np.min(mean_exposures)