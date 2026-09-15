import argparse
import torch
import numpy as np
import os
import random
from torch_geometric.loader import DataLoader
import torch_geometric
from datasets import QM93D
from utils import *
from al import *


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    torch.use_deterministic_algorithms(True)
    pass

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection_method', default='random', type=str,  help='selection method')
    parser.add_argument('--target', default='mu', type=str,  help='target property')
    parser.add_argument('--backbone', default='spherenet', type=str,  help='backbone model')
    parser.add_argument('--dataset', default='qm9_pyg_25k_0.pt', type=str,  help='.pt dataset')
    parser.add_argument('--device', default=0, type=int,  help='GPU Device')
    parser.add_argument('--cycle', default=0, type=int,  help='AL Cycle')
    parser.add_argument('--expt', default=1, type=int,  help='Eperiment number')
    parser.add_argument('--ADDENDUM', default=1500, type=int,  help='No of data to add')
    parser.add_argument('--epochs', default=500, type=int,  help='No of epochs')
    parser.add_argument('--train_size', default=25000, type=int,  help='total samples in pool (L+U)')
    parser.add_argument('--valid_size', default=10000, type=int,  help='No of samples in valid set')
    parser.add_argument('--init_size', default=5000, type=int,  help='Labeled seed size')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    opt = parse_opt()
    
    device = opt.device
    torch.cuda.set_device(device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    selection_method = opt.selection_method
    backbone = opt.backbone
    
    assert selection_method in set(['coreset', 'random', 'unc_div',  'mqr_ud', 'gradient_conflicts'])
    assert backbone in set(['spherenet', 'dimenetpp'])
    setup_seed(opt.expt)

    if backbone == 'spherenet':
        from models import SphereNet_BGNN as Model
    else:
        from models import DimenetPP_BGNN as Model
    
    print(f'Selection method: {selection_method}')
    device = opt.device
    cycle = opt.cycle
    expt = opt.expt
    ADDENDUM = opt.ADDENDUM
    dataset = opt.dataset
    epochs = opt.epochs
    init_size = opt.init_size
    train_size = opt.train_size
    valid_size = opt.valid_size
    print(f'Dataset: {dataset}')


    dataset = QM93D(root='dataset/', processed_fn=dataset)
    target = opt.target # choose from: mu, alpha, homo, lumo, r2, zpve, U0, U, H, G, Cv
    print(f'Target prediction: {target}')
    dataset.data.y = dataset.data[target]

    print(f'Length of entire dataset: {len(dataset.data.y)}')

    split_idx = dataset.get_idx_split(len(dataset.data.y), init_size=init_size, train_size=train_size,valid_size=valid_size, 
                                      seed=42, method= selection_method, cycle=cycle, expt=expt, ADDENDUM=ADDENDUM)

    train_dataset, valid_dataset, test_dataset, unlab_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']], dataset[split_idx['unlabeled']]
    
    print('train, validaion, test, unlab_dataset:', len(train_dataset), len(valid_dataset), len(test_dataset), len(unlab_dataset))
    print('train: ', train_dataset)

    #compute similarity matrix of unlab samples
    model = Model(energy_and_force=False, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=64, 
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
        num_spherical=3, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True, droprate=0.2
        )
  
    model = model.to(device)
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

    run3d = run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          method=selection_method,
          epochs=epochs, batch_size=32,
          vt_batch_size=64, lr=0.0005, lr_decay_factor=0.5, 
          lr_decay_step_size=15, expt=expt, cycle=cycle, save_dir=f"runs/{selection_method}/run{expt}")  
          #32, 32 for Qmugs; 32, 64 for QM9

    if selection_method == 'random':
        query_indice, time_taken = random_sel(split_idx['train'].tolist(), split_idx['unlabeled'].tolist(), k=ADDENDUM)
        # print('random len query: ', len(query_indices))
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indice))
        
        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')


    elif selection_method == 'coreset':
        print('coreset')
        model.eval()
        query_indices, time_taken = coreset(model, split_idx['train'].tolist(), split_idx['unlabeled'].tolist(),dataset[:train_size], 
                            device=device, k=ADDENDUM)
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')
    
    elif selection_method == 'unc_div':
        print('uncertainty_diversity')
        model.eval()
        query_indices, time_taken = compute_uncertainty_diversity_gpu(model, split_idx['train'].tolist(), split_idx['unlabeled'].tolist(),dataset[:train_size], 
            device=device, k=ADDENDUM, expt=expt, selection_method='unc_div')
        print(f'Len of next batch:------------  {len(query_indices)}')
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')
                
    elif selection_method == 'mqr_ud':
        model.eval()
        query_indices, time_taken = MQR_ud(model, split_idx['unlabeled'].tolist(), labeled_indices=split_idx['train'].tolist(), data=dataset[:train_size], 
                        device=device, k=ADDENDUM, valid_dataset=valid_dataset, expt=expt, cycle=cycle, path=f'runs/{selection_method}/run{expt}/')

        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices.tolist()+split_idx['train'].tolist()))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')

    elif selection_method == 'gradient_conflicts':
        model.eval()
        query_indices, time_taken = gradient_conflicts(model, split_idx['unlabeled'].tolist(), labeled_indices=split_idx['train'].tolist(), data=dataset[:train_size], 
                        device=device, k=ADDENDUM, valid_dataset=valid_dataset, expt=expt, cycle=cycle, path=f'runs/{selection_method}/run{expt}/')

        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices.tolist()+split_idx['train'].tolist()))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')       
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')
    