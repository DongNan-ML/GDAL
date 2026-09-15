
import time
import os
import gc
import pandas as pd
import torch
from torch.optim import Adam, SGD
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import sys
from . import get_average_node_per_molecule
import random
import torch_geometric
from sklearn.metrics import mean_squared_error, r2_score

class run():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        pass
    
    def setup_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch_geometric.seed.seed_everything(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
        torch.use_deterministic_algorithms(True)
        pass
    
  
    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, method='random', epochs=200, batch_size=32, vt_batch_size=32, lr=0.0005,  lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, save_dir='', log_dir='', expt=1, cycle=0): 
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        self.setup_seed(expt)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        best_valid = float('inf')
        best_epoch = 0
        best_metrics = {}
            
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)

        training_results = []
            
        for epoch in range(1, epochs + 1):
            print("\n=====Epoch {}".format(epoch), flush=True)
            
            print('\nTraining...', flush=True)
            train_mae, train_rmse, train_r2 = self.train(model, optimizer, train_loader, loss_func, device)

            print('\nEvaluating...', flush=True)
            valid_mae, valid_rmse, valid_r2 = self.val(model, valid_loader, evaluation, device)

            print(f'Train  - MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, R²: {train_r2:.6f}')
            print(f'Valid  - MAE: {valid_mae:.6f}, RMSE: {valid_rmse:.6f}, R²: {valid_r2:.6f}')

            epoch_result = {
            'epoch': epoch,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'valid_mae': valid_mae,
            'valid_rmse': valid_rmse,
            'valid_r2': valid_r2
            }
            training_results.append(epoch_result)

            if log_dir != '':
                writer.add_scalar('train_mae', train_mae, epoch)
                writer.add_scalar('train_rmse', train_rmse, epoch)
                writer.add_scalar('train_r2', train_r2, epoch)
                writer.add_scalar('valid_mae', valid_mae, epoch)
                writer.add_scalar('valid_rmse', valid_rmse, epoch)
                writer.add_scalar('valid_r2', valid_r2, epoch)
            
            if valid_mae < best_valid:
                best_valid = valid_mae
                best_epoch = epoch
                best_metrics = {
                'valid_mae': valid_mae,
                'valid_rmse': valid_rmse,
                'valid_r2': valid_r2,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2
            }
                if save_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {
                        'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'scheduler_state_dict': scheduler.state_dict(), 
                        'best_valid_mae': best_valid, 
                        'best_metrics': best_metrics, 
                        'num_params': num_params
                    }
                    torch.save(checkpoint, os.path.join(save_dir, f'valid_checkpoint_{cycle}.pt'))

            scheduler.step()

        print(f'\nTraining completed. Best validation MAE: {best_valid} at epoch {best_epoch}')

        if save_dir != '' and os.path.exists(os.path.join(save_dir, f'valid_checkpoint_{cycle}.pt')):
            print('Loading best model for testing...')
            checkpoint = torch.load(os.path.join(save_dir,  f'valid_checkpoint_{cycle}.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print('\nFinal Testing...', flush=True)
        test_mae, test_rmse, test_r2 = self.val(model, test_loader, evaluation, device)
        
        print(f'Final results:')
        print(f'Best validation MAE: {best_valid:.6f} (epoch {best_epoch})')
        print(f'Test results with best model:')
        print(f'MAE: {test_mae:.6f}')
        print(f'RMSE: {test_rmse:.6f}')
        print(f'R²: {test_r2:.6f}')
  
        results_dir = f'runs/{method}/run{expt}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_df = pd.DataFrame(training_results)
        results_csv_path = os.path.join(results_dir, f'training_results_cycle_{cycle}.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f'Training results saved to: {results_csv_path}')

        with open(os.path.join(f'runs/{method}/run{expt}', f'res.txt'), 'a') as f:
            f.write(f'CYCLE: {cycle}   BEST_EPOCH: {best_epoch}   '
                    f'VALID(MAE/RMSE/R2): {best_metrics["valid_mae"]}/{best_metrics["valid_rmse"]}/{best_metrics["valid_r2"]}   '
                    f'TEST(MAE/RMSE/R2): {test_mae}/{test_rmse}/{test_r2}\n')

        if log_dir != '':
            writer.add_scalar('final_test_mae', test_mae, epochs)
            writer.add_scalar('final_test_rmse', test_rmse, epochs)
            writer.add_scalar('final_test_r2', test_r2, epochs)
            writer.close()


    def train(self, model, optimizer, train_loader, loss_func, device):
        model.train()
        loss_accum = 0
        num_samples = 0

        train_preds = []
        train_targets = []
        
        for step, batch_data in enumerate(tqdm(train_loader)):

            if step % 20 == 0 and step > 0:
                torch.cuda.empty_cache()    
            
            #zero grad optimizers
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            _, _, out, _ = model(batch_data)

            loss = loss_func(out, batch_data.y.unsqueeze(1))   
                    
            train_preds.append(out.detach().cpu())
            train_targets.append(batch_data.y.unsqueeze(1).detach().cpu())
    
            loss.backward()  # retain_graph=False is default
            optimizer.step()

            batch_size = batch_data.y.size(0)
            loss_accum += loss.detach().cpu().item() * batch_size
            num_samples += batch_size

            del batch_data, out, loss

        if train_preds:
            train_preds_np = torch.cat(train_preds, dim=0).numpy().flatten()
            train_targets_np = torch.cat(train_targets, dim=0).numpy().flatten()
                
            mse = mean_squared_error(train_targets_np, train_preds_np)
            rmse = np.sqrt(mse)
            r2 = r2_score(train_targets_np, train_preds_np)
            
            del train_preds, train_targets, train_preds_np, train_targets_np

        else:
            rmse, r2 = 0.0, 0.0
        
        torch.cuda.empty_cache()
        
        if num_samples > 0:
            return (loss_accum / num_samples, float(rmse), float(r2))
        else:
            return (0.0, 0.0, 0.0)
        

    def val(self, model, data_loader, evaluation, device):
        model.eval()
        preds_cpu = []
        targets_cpu = []

        with torch.no_grad(): 
            for step, batch_data in enumerate(tqdm(data_loader)):
                batch_data = batch_data.to(device)
                _, _, out, _ = model(batch_data)
                
                preds_cpu.append(out.detach().cpu().numpy())
                targets_cpu.append(batch_data.y.unsqueeze(1).detach().cpu().numpy())

                del out, batch_data
                if step % 20 == 0 and step > 0:
                    torch.cuda.empty_cache()

        preds_np = np.concatenate(preds_cpu, axis=0)
        targets_np = np.concatenate(targets_cpu, axis=0)
    
        input_dict = {"y_true": targets_np, "y_pred": preds_np}
        results = evaluation.eval(input_dict)
        mae = results['mae']
        rmse = results['rmse'] 
        r2 = results['r2']

        del preds_cpu, targets_cpu, preds_np, targets_np, input_dict, results
        torch.cuda.empty_cache()
        gc.collect()

        return mae, rmse, r2