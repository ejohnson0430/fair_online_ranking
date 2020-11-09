import numpy as np
from copy import deepcopy

from .utils import append_exposures, append_to_ranking
from .metrics import DDP
from .can_be_fair import can_be_fair, min_exposure_heuristic



def fair_rerank(batch, past_exposures, ddp_thresh, num_groups):
    qs = np.zeros(num_groups, dtype=int)
    groups, queue_sizes = np.unique(batch[:,1].astype(int), return_counts=True)
    qs[groups] = queue_sizes
    queues = {}
    for j in groups:       
        queues[j] = np.sort(batch[batch[:,1]==j,2])[::-1]
    curr_ranking = np.zeros([0,3])
    while len(queues) > 1:
        # try top member of each group ordered by relevance
        found = False
        for j in sorted(queues, key=lambda j: queues[j][0], reverse=True):
            sim_ranking = append_to_ranking(curr_ranking, j, queues[j][0])
            sim_qs = deepcopy(qs)
            sim_qs[j] -= 1
            if can_be_fair(sim_ranking, past_exposures, sim_qs, ddp_thresh):
                found = True
                curr_ranking = sim_ranking
                qs = sim_qs
                queues[j] = queues[j][1:]
                if len(queues[j])==0: del queues[j]
                break
        # if heuristic cannot meet constraint, add from group w/ lowest exposure
        if not found:
            j = min_exposure_heuristic(curr_ranking, qs, past_exposures)
            curr_ranking = append_to_ranking(curr_ranking, j, queues[j][0])
            qs[j] -= 1
            queues[j] = queues[j][1:]
            if len(queues[j])==0: del queues[j]
    # add candidates from remaining group
    for j in queues:
        for rel in queues[j]:
            curr_ranking = append_to_ranking(curr_ranking, j, rel)
    return curr_ranking

def fair_online(batches, ddp_thresh, num_groups, past_exposures=None, debug=False):
    processed_batches = [batch.copy() for batch in batches]
    if past_exposures is None: past_exposures = np.zeros([num_groups,2])
    for i, batch in enumerate(processed_batches):
        processed_batches[i] = fair_rerank(batch, past_exposures, ddp_thresh, num_groups)
        past_exposures = append_exposures(processed_batches[i], past_exposures)
        ddp = DDP(past_exposures)
        if debug and ddp > ddp_thresh:
            print(f'FA*IR did not meet constraint on batch {i}: DDP {ddp}')
    return processed_batches