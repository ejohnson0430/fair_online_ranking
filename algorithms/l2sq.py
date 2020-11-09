import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import append_exposures, get_mean_exposures, append_to_ranking
from .metrics import DDP, nDCG
from .can_be_fair import can_be_fair, min_exposure_heuristic
from .fair_queues import fair_online


class L2SQ(nn.Module):

    def __init__(self, ddp_thresh, num_groups):
        super(L2SQ, self).__init__()
        self.ddp_thresh = ddp_thresh
        self.num_groups = num_groups
        
        self.num_features = self.num_groups * 18
        self.fc1 = nn.Linear(self.num_features, self.num_features//2)
        self.fc2 = nn.Linear(self.num_features//2, self.num_features//4)
        self.fc3 = nn.Linear(self.num_features//4, num_groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scores = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return torch.exp(scores)
    
    def features(self, queues, curr_ranking, past_exposures):
        # stats for queue / ranked members for each group
        features = np.array([])
        mean_exposures = get_mean_exposures(past_exposures)
        total_count = np.sum(past_exposures[:,1])
        total_count = max(total_count,1)
        features = np.append(features, mean_exposures)
        features = np.append(features, past_exposures[:,1]/total_count)
        for j in range(self.num_groups):
            if j in queues:
                queue = queues[j]
                features = np.append(features, [queue[0], queue.size, np.min(queue), np.max(queue), 
                                                np.mean(queue), np.std(queue)])
            else:
                features = np.append(features, [0]*6)
            ranking_inds = np.where(curr_ranking[:,1]==j)[0]
            if len(ranking_inds) > 0:
                ranks = curr_ranking[ranking_inds,0]
                rel = curr_ranking[ranking_inds,2]
                for stat in [ranks, rel]:
                    features = np.append(features, [stat.size, np.min(stat), np.max(stat), 
                                                    np.mean(stat), np.std(stat)])
            else:
                features = np.append(features, [0]*10)
        return torch.FloatTensor(features)

    def init_state(self, batch):
        # queues is a dict of the sorted list of relevance scores for each group
        # qs is a list of the queue size for each group
        # curr_ranking is an empty array of the ranking dimensions
        curr_ranking = np.zeros([0,3]) 
        queues = {}
        qs = np.zeros(self.num_groups, dtype=int)
        for j in range(self.num_groups):       
            ranking_inds = np.where(batch[:,1]==j)[0]
            qs[j] = len(ranking_inds)
            if len(ranking_inds) > 0:
                queues[j] = np.sort(batch[ranking_inds,2])[::-1]
        return queues, qs, curr_ranking
    
    def test(self, batch, past_exposures, state=None):
        if state: queues, qs, curr_ranking = state
        else: queues, qs, curr_ranking = self.init_state(batch)
        while len(queues) > 1:
            # try top member of each group ordered by model scores
            features = self.features(queues, curr_ranking, past_exposures)
            with torch.no_grad(): scores = self(features)
            found = False
            for j in torch.argsort(scores, descending=True).numpy():
                if j not in queues: continue
                sim_ranking = append_to_ranking(curr_ranking, j, queues[j][0])
                sim_qs = deepcopy(qs)
                sim_qs[j] -= 1
                if can_be_fair(sim_ranking, past_exposures, sim_qs, self.ddp_thresh):
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
                if len(queues[j])==0:
                    del queues[j]
        # add candidates from remaining group
        for j in queues:
            for rel in queues[j]:
                curr_ranking = append_to_ranking(curr_ranking, j, rel)
        return curr_ranking
    
    def test_online(self, batches, past_exposures=None, debug=False):
        processed_batches = [batch.copy() for batch in batches]
        if past_exposures is None: past_exposures = np.zeros([self.num_groups,2])
        for i, batch in enumerate(processed_batches):
            processed_batches[i] = self.test(batch, past_exposures)
            past_exposures = append_exposures(processed_batches[i], past_exposures)
            ddp = DDP(past_exposures)
            if debug and ddp > self.ddp_thresh:
                print(f'L2SQ did not meet constraint on batch {i}: DDP {ddp}')
        return processed_batches
    
    def roll_out(self, batches, past_exposures, beta=1):
        if np.random.uniform() < beta:
            return fair_online(batches, self.ddp_thresh, self.num_groups, past_exposures)
        else:
            return self.test_online(batches, past_exposures)
    
    def train(self, X, params, val_batches=None, dataset=None):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=params['lr'], 
                                     weight_decay=params['weight_decay'])
        beta = params['beta']
        roll_out_length = 3
        best_val_nDCG = 0
        for _ in range(params['epochs']):
            epoch_loss = 0
            for batches in X:
                loss = torch.tensor(0.0)
                for t in range(len(batches)):         
                    # roll in
                    past_exposures = np.zeros([self.num_groups,2])
                    roll_in = self.test_online(batches[:t])
                    for batch in roll_in:
                        past_exposures = append_exposures(batch, past_exposures)           
                    batch_t = batches[t]
                    queues, qs, curr_ranking = self.init_state(batch_t)
                    tensors = []
                    dev_ndcgs = []
                    while len(queues) > 1:
                        found_first = False
                        dev_ndcgs.append({})
                        scores = self(self.features(queues, curr_ranking, past_exposures))
                        tensors.append(scores)
                        # try top member of each group ordered by model scores
                        for j in torch.argsort(scores, descending=True).numpy():
                            if j not in queues: continue
                            sim_ranking = append_to_ranking(curr_ranking, j, queues[j][0])
                            sim_qs = deepcopy(qs)
                            sim_qs[j] -= 1
                            if can_be_fair(sim_ranking, past_exposures, sim_qs, self.ddp_thresh):
                                if not found_first:
                                    found_first = True
                                    model_action = j
                                dev_ranking = append_to_ranking(curr_ranking, j, queues[j][0])
                                dev_queues = deepcopy(queues)
                                dev_queues[j] = dev_queues[j][1:]
                                if len(dev_queues[j])==0: del dev_queues[j]
                                dev_state = (dev_queues, sim_qs, dev_ranking)
                                dev_batch = self.test(None, past_exposures, state=dev_state)
                                dev_exposures = append_exposures(dev_batch, past_exposures)
                                dev_roll_out = self.roll_out(batches[t+1:t+1+roll_out_length], dev_exposures, beta)
                                dev_batches = roll_in + [dev_batch] + dev_roll_out
                                dev_ndcgs[-1][j] = np.sum([nDCG(batch) for batch in dev_batches])
                        # update nondev
                        if not found_first:
                            model_action = min_exposure_heuristic(curr_ranking, qs, past_exposures)
                        curr_ranking = append_to_ranking(curr_ranking, model_action, queues[model_action][0])
                        qs[model_action] -= 1
                        queues[model_action] = queues[model_action][1:]
                        if len(queues[model_action])==0: del queues[model_action]
                    # calculate loss by summing over each choice
                    for i, ndcg_dict in enumerate(dev_ndcgs):
                        if not ndcg_dict: continue
                        best_action = max(ndcg_dict.keys(), key=(lambda k: ndcg_dict[k]))
                        best_ndcg = ndcg_dict[best_action]
                        del ndcg_dict[best_action]
                        for j in ndcg_dict:
                            cost = best_ndcg - ndcg_dict[j]
                            loss += (cost * -torch.log(self.sigmoid(tensors[i][best_action] - tensors[i][j])))
                if loss.item() > 0:
                    epoch_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            val_batches_processed = [self.test_online(batches) for batches in val_batches]
            val_nDCG = np.mean([[nDCG(batch) for batch in batches] for batches in val_batches_processed])
            if val_nDCG > best_val_nDCG:
                best_val_nDCG = val_nDCG
                best_model = deepcopy(self)
        return best_model