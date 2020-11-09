import numpy as np
import torch
import sys
import time

from algorithms.l2sq import L2SQ

def main():
    dataset = str(sys.argv[1])
    epochs = int(sys.argv[2])
    beta = float(sys.argv[3])

    X_train = np.load(f'datasets/{dataset}_train.npy', allow_pickle=True)
    X_val = np.load(f'datasets/{dataset}_val.npy', allow_pickle=True)

    ddp_thresh = 0.1
    if dataset=='stacke': num_groups = 3
    else: num_groups = 4
    l2sq = L2SQ(ddp_thresh, num_groups)

    params = {}
    params['epochs'] = epochs
    params['beta'] = beta
    params['lr'] = 1e-3
    params['weight_decay'] = 0
    
    best_model = l2sq.train(X_train,params,val_batches=X_val,dataset=dataset)
    torch.save(best_model, f'models/{dataset}_beta{beta}_{int(time.time())}.pt')

if __name__ == "__main__":
    main()