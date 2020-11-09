import numpy as np
from copy import deepcopy

from .utils import group_exposures, get_mean_exposures, append_exposures, append_to_ranking
from .metrics import DDP


def heuristic(curr_ranking, qs):
    h = np.zeros([qs.shape[0],2])
    Nr = curr_ranking.shape[0]
    Nu = np.sum(qs)
    remaining_ranks = np.arange(Nr, Nr+Nu) + 1
    avg_remaining_exposure = np.mean(1 / np.log2(1 + remaining_ranks))
    exposures = group_exposures(curr_ranking, len(qs))
    for group, remaining_ct in enumerate(qs):
        h[group,1] = remaining_ct + exposures[group,1]
        h[group,0] = remaining_ct * avg_remaining_exposure + exposures[group,0]
    return h

def min_exposure_heuristic(curr_ranking, qs, past_exposures):
    h = heuristic(curr_ranking, qs)
    mean_exposures = get_mean_exposures(past_exposures + h)
    mean_exposures[qs==0] = np.inf
    return np.argmin(mean_exposures)

def can_be_fair(curr_ranking, past_exposures, _qs, ddp_thresh):
    # curr_ranking should be the ranking up to this point
    # past_exposures should be a G x 2 matrix
    # qs is the list of queue sizes
    # ddp_thresh should be obvious
    qs = deepcopy(_qs)
    sim_ranking = deepcopy(curr_ranking)
    while(np.sum(qs) > 0):
        j = min_exposure_heuristic(sim_ranking, qs, past_exposures)
        sim_ranking = append_to_ranking(sim_ranking, j, 0)
        qs[j] -= 1
    sim_exposures = append_exposures(sim_ranking, past_exposures)
    sim_ddp = DDP(sim_exposures)
    return sim_ddp <= ddp_thresh