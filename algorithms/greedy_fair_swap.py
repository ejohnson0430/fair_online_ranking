import numpy as np

from .utils import append_exposures, get_mean_exposures
from .metrics import DDP


def gbf_swap(batch, past_exposures, num_groups):
    new_exposures = append_exposures(batch, past_exposures)
    mean_exposures = get_mean_exposures(new_exposures)
    
    qs = np.zeros(num_groups, dtype=int)
    groups, queue_sizes = np.unique(batch[:,1].astype(int), return_counts=True)
    qs[groups] = queue_sizes
    
    temp_me = mean_exposures.copy()
    temp_me[qs==0] = -np.inf
    over_group = np.argmax(temp_me)
    
    temp_me = mean_exposures.copy()
    temp_me[qs==0] = np.inf
    under_group = np.argmin(temp_me)
    
    over_ranks = batch[batch[:,1]==over_group,0].reshape(-1,1)
    under_ranks = batch[batch[:,1]==under_group,0].reshape(-1,1)
    min_over = np.min(over_ranks)
    
    valid_under = under_ranks>min_over
    if np.sum(valid_under)==0: swap_under_rank = np.min(under_ranks)
    else: swap_under_rank = np.min(under_ranks[valid_under])

    valid_over = over_ranks<swap_under_rank
    if np.sum(valid_over)==0: swap_over_rank = np.max([np.min(under_ranks)-1, 1])
    else: swap_over_rank = np.max(over_ranks[valid_over])

    swap_under_ind = np.where(batch[:,0]==swap_under_rank)[0]
    swap_over_ind = np.where(batch[:,0]==swap_over_rank)[0]
    batch[swap_under_ind,0] = swap_over_rank
    batch[swap_over_ind,0] = swap_under_rank
    if swap_over_rank==swap_under_rank: pass #print('swapping first')
    
def gbf(batch, past_exposures, ddp_thresh, num_groups):
    processed = batch.copy()
    new_exposures = append_exposures(processed, past_exposures)
    swap_limit = batch.shape[0]
    num_swaps=0
    while DDP(new_exposures) > ddp_thresh:
        # break if exceeding swap limit
        if num_swaps > swap_limit:
            break
        gbf_swap(processed, past_exposures, num_groups)
        new_exposures = append_exposures(processed, past_exposures)
        num_swaps+=1
    return processed

def gbf_online(batches, ddp_thresh, num_groups, past_exposures=None, debug=False):
    processed_batches = [batch.copy() for batch in batches]
    if past_exposures is None: past_exposures = np.zeros([num_groups,2])
    for i, batch in enumerate(processed_batches):
        processed_batches[i] = gbf(batch, past_exposures, ddp_thresh, num_groups)
        past_exposures = append_exposures(processed_batches[i], past_exposures)
        ddp = DDP(past_exposures)
        if debug and ddp > ddp_thresh:
            print(f'GBF did not meet constraint on batch {i}: DDP {ddp}')
    return processed_batches