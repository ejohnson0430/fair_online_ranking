import numpy as np
import torch
import scipy.stats
import matplotlib.pyplot as plt

from algorithms.utils import append_exposures
from algorithms.metrics import DDP, nDCG
from algorithms.l2sq import L2SQ
from algorithms.fair_queues import fair_online
from algorithms.greedy_fair_swap import gbf_online

import seaborn as sns
sns.set()
sns.set_style(style='white')


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def main():
    num_timesteps = 25
    ddp_thresh = 0.1

    best_models = {
        'synthe': 'models/synthe_beta1.0_epoch16_1597113344.pt',
        'german': 'models/german_beta1.0_epoch16_1597128844.pt',
        'resume': 'models/resume_beta1.0_epoch12_1597171199.pt',
        'airbnb': 'models/airbnb_beta1.0_epoch7_1597161178.pt',
        'stacke': 'models/stacke_beta0.5_epoch11_1597013342.pt'
        }
    
    plot_data = {}
    for dataset in best_models:
        if dataset=='stacke': num_groups = 3
        else: num_groups = 4
            
        best_model = torch.load(best_models[dataset])
        
        test_batches = np.load(f'datasets/{dataset}_val.npy', allow_pickle=True)
        l2sq_batches = [best_model.test_online(batches) for batches in test_batches]
        fair_batches = [fair_online(batches, ddp_thresh, num_groups) for batches in test_batches]
        gbf_batches = [gbf_online(batches, ddp_thresh, num_groups) for batches in test_batches]
        
        plot_data[dataset] = [test_batches, l2sq_batches, fair_batches, gbf_batches]


    fontsize = 20
    labels = ['Unprocessed', 'L2S Queues', 'Fair Queues', 'Greedy Fair Swap']

    fig, ax = plt.subplots(2, 4, figsize=(25, 10), tight_layout=True)

    for a1 in ax:
        for a2 in a1:
            for item in ([a2.title, a2.xaxis.label, a2.yaxis.label] +
                        a2.get_xticklabels() + a2.get_yticklabels()):
                item.set_fontsize(fontsize)
                
    cols = ['German Credit', 'Resume', 'Airbnb', 'Stack Exchange']
    rows = ['nDCG', 'DDP']

    for a, col in zip(ax[0], cols):
        a.set_title(col, fontsize=fontsize)

    for a, row in zip(ax[:,0], rows):
        a.set_ylabel(row, rotation=90)

    for i, algo in enumerate(plot_data):
        if i==0: continue
        i-=1
        for j, method_batches in enumerate(plot_data[algo]):
            ddps=[]
            for batches in method_batches:
                if algo=='stacke': num_groups = 3
                else: num_groups = 4
                past_exposures = np.zeros([num_groups,2])
                ddps.append([])
                for batch in batches:
                    past_exposures = append_exposures(batch, past_exposures)
                    ddps[-1].append(DDP(past_exposures))
            ndcgs = [np.cumsum([nDCG(batch) for batch in batches])/(np.arange(num_timesteps)+1) for batches in method_batches]
            conf_dps = np.array([confidence_interval(t) for t in np.array(ddps).T])
            conf_ndcgs = np.array([confidence_interval(t) for t in np.array(ndcgs).T])

            ax[0,i].errorbar(np.arange(num_timesteps), conf_ndcgs[:,0], yerr=conf_ndcgs[:,1], alpha=1, label=labels[j], color=f'C{j}')
            ax[0,i].set_xticklabels([])
            ax[0,i].legend(loc = 'lower right', prop={'size': 15})
            ax[1,i].errorbar(np.arange(num_timesteps), conf_dps[:,0], yerr=conf_dps[:,1], alpha=1, label=labels[j], color=f'C{j}')
            ax[1,i].set_ylim(0,0.25)

        ax[1,i].plot([ddp_thresh]*num_timesteps, color='black', linestyle='dashed', alpha=0.7)
        
    fig.tight_layout()
    plt.savefig('plots/paper_results.png', dpi = 300)
    plt.show()


if __name__ == "__main__":
    main()