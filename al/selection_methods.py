import time
from tqdm import tqdm
import random
import os
from scipy.spatial.distance import cdist
import scipy.sparse as spa
from fastdist import fastdist
import numpy as np
import osqp
from qpsolvers import solve_qp
from torchmetrics.functional import pairwise_euclidean_distance
import torch
from torch_scatter import scatter_mean
from torch.utils.data import TensorDataset
from .sampler import SubsetSequentialSampler
from utils import get_average_node_per_molecule
from .quantile_training import MLP, train_model
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
import networkx as nx
from typing import Optional
from torch_geometric.loader import DataLoader as GeoDataLoader  
from torch.utils.data import DataLoader as TorchDataLoader 
DataLoader = GeoDataLoader
from sklearn.cluster import KMeans, kmeans_plusplus
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from torch.func import vmap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('ConcreteDropout'):
            m.train()

def random_sel(labeled_indices, unlabeled_indices, k=1500):
    t0 = time.perf_counter()
    indices = random.sample(unlabeled_indices, k) + labeled_indices
    dt = time.perf_counter() - t0
    return indices, dt


def mc_dropout(model, unlabeled_indices, data, device, k=1500, with_diversity=False):

    time_start = time.time()
    pred_matrix = torch.tensor([]).to(device)

    for i in range(20):
        unlabeled_loader = GeoDataLoader(data, batch_size=64, 
                            sampler=SubsetSequentialSampler(unlabeled_indices), # more convenient if we maintain the order of subset
                            pin_memory=True, drop_last=False)

        print('Len_unlabloader: ', len(unlabeled_loader))
        
        #set model to evaluation mode
        model.eval()
        enable_dropout(model)

        with torch.cuda.device(device):
            preds = torch.tensor([]).to(device)

        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(unlabeled_loader)):
                batch_data = batch_data.to(device)
            
                with torch.no_grad():
                    _, _, out, _ = model(batch_data)
                    # print(out.shape)
                preds = torch.cat([preds, out.detach_()], dim=0)

        pred_matrix = torch.cat([pred_matrix, preds], dim=1)
        # print(pred_matrix[:10, :])
    
    
    variance = torch.var(pred_matrix, dim=1)
    if with_diversity:
        return variance

    arg = torch.argsort(variance, descending=True).cpu()
    return torch.tensor(unlabeled_indices)[arg[:k]], (time.time()-time_start)/60

##uncertainty rep using euclidean distance
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def compute_USR(mol_pos):
    # Function to calculate four moments for a given set of values
    def calculate_moments(values):
        mean_value = values.mean()
        variance = values.var()
        skewness = (torch.abs((values - mean_value) ** 3)).mean().sqrt()
        kurtosis = ((values - mean_value) ** 4).mean().sqrt()
        return [mean_value, variance, skewness, kurtosis]

    # Function to calculate the angle between two vectors
    def calculate_angle(a, b):
        cos_angle = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        return angle

    # Calculate the centroid
    centroid = mol_pos.mean(dim=0)

    # Find the first farthest point from the centroid
    distances_to_centroid = torch.norm(mol_pos - centroid, dim=1)
    farthest_point = mol_pos[torch.argmax(distances_to_centroid)]
    closest_point = mol_pos[torch.argmin(distances_to_centroid)]
    farfromfarthest_point = mol_pos[torch.argmax(farthest_point)]
    
    # Calculate the reference vector from centroid to the farthest point
    reference_vector = farthest_point - centroid
    reference_vector1 = closest_point - centroid
    reference_vector2 = farfromfarthest_point - centroid

    # Compute angles with the reference vector for atoms to centroid
    angles1 = [calculate_angle(reference_vector, pt - centroid) for pt in mol_pos]
    angles11 = [calculate_angle(reference_vector1, pt - centroid) for pt in mol_pos]
    angles12 = [calculate_angle(reference_vector2, pt - centroid) for pt in mol_pos]
    
    # Compute angles with the reference vector for all possible atom pairs
    angles2 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector, mol_pos[j] - mol_pos[i])
                angles2.append(angle)
    angles21 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector1, mol_pos[j] - mol_pos[i])
                angles21.append(angle)
    angles22 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector2, mol_pos[j] - mol_pos[i])
                angles22.append(angle)

    # Calculate moments for distances from 4 reference points
    distances_to_point1 = torch.norm(mol_pos - farthest_point, dim=1)
    farthest_point2 = mol_pos[torch.argmax(distances_to_point1)]
    
    distances_to_point2 = torch.norm(mol_pos - centroid, dim=1)
    farthest_point3 = mol_pos[torch.argmin(distances_to_point2)]

    points = [centroid, farthest_point, farthest_point2, farthest_point3]
    moments_all = []
    for point in points:
        distances = torch.norm(mol_pos - point, dim=1)
        moments_all.extend(calculate_moments(distances))
    #print("calculate_moments(distances):", torch.tensor(calculate_moments(distances)))

    # Add moments for both sets of angles
    moments_all.extend(calculate_moments(torch.tensor(angles1)))
    moments_all.extend(calculate_moments(torch.tensor(angles11)))
    moments_all.extend(calculate_moments(torch.tensor(angles12)))
    moments_all.extend(calculate_moments(torch.tensor(angles2)))
    moments_all.extend(calculate_moments(torch.tensor(angles21)))
    moments_all.extend(calculate_moments(torch.tensor(angles22)))
    #print("Shape of moments_all:", torch.tensor(moments_all).shape)

    return torch.tensor(moments_all)

def pairwise_usr_similarity(matrix1, matrix2=None):
    if matrix2 is None:
        matrix2 = matrix1
        
    # Compute distances
    distances = cdist(matrix1.cpu().numpy(), matrix2.cpu().numpy(), 'cityblock') / 4.0 + 1.0
    similarities = 1.0 / distances
    
    # Convert similarities to a torch tensor
    tensor_similarities = torch.tensor(similarities, dtype=torch.float32)
    return tensor_similarities

def compute_uncertainty_diversity(model, labeled_indices, unlabeled_indices, data, device, k=1500,expt=1, selection_method='unc_div'):

    time_start = time.time()
    unlabeled_loader = DataLoader(data, batch_size=64, 
                        sampler=SubsetSequentialSampler(unlabeled_indices), # more convenient if we maintain the order of subset
                        pin_memory=True, drop_last=False)
   
    print('Len_unlabloader: ', len(unlabeled_loader))
    
    #set model to evaluation mode
    model.eval()

    with torch.cuda.device(device):
        #features = torch.tensor([]).to(device)
        features1 = torch.tensor([]).to(device)

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(unlabeled_loader)):
            batch_data = batch_data.to(device)
          
            with torch.no_grad():
                z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
               
            usr_features = []
            for mol_idx in batch.unique():
                mol_pos = pos[batch == mol_idx]
                usr_features.append(compute_USR(mol_pos))
      
            usr_features = torch.stack(usr_features).to(device)
            features1 = torch.cat((features1, usr_features), 0)
            
    unlab_predictions = features1

    similarity_matrix = pairwise_usr_similarity(unlab_predictions)
    similarity_matrix = similarity_matrix/similarity_matrix.max()
    similarity_matrix.fill_diagonal_(30.0)
    
    similarity_matrix = similarity_matrix.cpu().numpy()
    print('similarity_matrix shape: ', similarity_matrix.shape)


    uncertainty = mc_dropout(model, unlabeled_indices, data, device, k=1500, with_diversity=True)
    uncertainty = uncertainty/uncertainty.max()
    uncertainty = uncertainty.cpu().numpy()

    ### setting upper bound lower bounds and constraints
    lb = np.zeros(len(unlabeled_indices))
    ub = np.ones(len(unlabeled_indices))

    A = np.ones(len(unlabeled_indices))
    b = 1.0 * np.array([k])
    
    # un_rep = uncertainty 

    #using osqp
    print(f'Solving qp....')
    #     prob = osqp.OSQP()

    # # Setup workspace and change alpha parameter
    #     prob.setup(diversity_matrix, uncertainty, A, lb, ub, alpha=1.0, warm_starting=True)
    #     res = prob.solve()
    res = solve_qp(P=similarity_matrix, q=uncertainty, G=None, A=A, b=b, lb=lb, ub=ub, solver='osqp', verbose=True)
    ##
    # Solve problem   
    arg = np.argsort(res)
    query_indices = np.array(unlabeled_indices)[arg[-k:]]
    return labeled_indices+query_indices.tolist(), (time.time()-time_start)//60


def compute_uncertainty_diversity_gpu(model, labeled_indices, unlabeled_indices, data, device, k=1500,expt=1, selection_method='unc_div'):

    time_start = time.perf_counter()
    #set model to evaluation mode
    model.eval()

    # load similarity matrix 
    similarity_matrix = torch.load(f'runs/{selection_method}/run{expt}/tensor.pt')
    # similarity_matrix.cpu().numpy()
    
    uncertainty = mc_dropout(model, unlabeled_indices, data, device, k=k, with_diversity=True)
    uncertainty = uncertainty/uncertainty.max()
    uncertainty = uncertainty.cpu().numpy()

    ### setting upper bound lower bounds and constraints
    lb = np.zeros(len(unlabeled_indices))
    ub = np.ones(len(unlabeled_indices))

    A = np.ones((1, len(unlabeled_indices)))
    b = 1.0 * np.array([k])
    
    # un_rep = uncertainty 

    #using osqp
    print(f'Solving qp....')
    P = spa.csc_matrix(similarity_matrix.cpu().numpy())
    q = uncertainty
    G = None 
    h = None


    A_osqp = None
    l_osqp = None
    u_osqp = None
    if G is not None and h is not None:
        A_osqp = G
        l_osqp = np.full(h.shape, -np.infty)
        u_osqp = h
    if A is not None and b is not None:
        A_osqp = A if A_osqp is None else spa.vstack([A_osqp, A], format="csc")
        l_osqp = b if l_osqp is None else np.hstack([l_osqp, b])
        u_osqp = b if u_osqp is None else np.hstack([u_osqp, b])
    if lb is not None or ub is not None:
        lb = lb if lb is not None else np.full(q.shape, -np.infty)
        ub = ub if ub is not None else np.full(q.shape, +np.infty)
        E = spa.eye(q.shape[0])
        A_osqp = E if A_osqp is None else spa.vstack([A_osqp, E], format="csc")
        l_osqp = lb if l_osqp is None else np.hstack([l_osqp, lb])
        u_osqp = ub if u_osqp is None else np.hstack([u_osqp, ub])


    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A_osqp, l=l_osqp, u=u_osqp, alpha=1.0, warm_starting=True)
    res = solver.solve()
    # print(res)
    res = res.x
    # res = solve_qp(P=diversity_matrix, q=uncertainty, G=None, A=A, b=b, lb=lb, ub=ub, solver='osqp', verbose=True)
    # qp_time = time.perf_counter() - qp_time
    
    arg = np.argsort(res)
    
    selected_indices = arg[-k:] ###gives index
    
    ''' ########### Here ############ '''
    unselected_indices = torch.tensor(np.setdiff1d(np.arange(similarity_matrix.shape[0]), selected_indices))
    
    #choose subset of diversity matrix containing unselected indices
    
    similarity_matrix = similarity_matrix[:,unselected_indices]
    similarity_matrix = similarity_matrix[unselected_indices,:]
    
    torch.save(similarity_matrix, f'runs/{selection_method}/run{expt}/tensor.pt')
    
    query_indices = np.array(unlabeled_indices)[arg[-k:]]
    return labeled_indices+query_indices.tolist(), time.perf_counter()-time_start

def compute_uncertainty_diversity_fast(model, labeled_indices, unlabeled_indices, data, device, k=1500,expt=1, selection_method='unc_div'):

    time_start = time.perf_counter()
    #set model to evaluation mode
    model.eval()

    #load similarity matrix 
    similarity_matrix = torch.load(f'runs/{selection_method}/run{expt}/tensor.pt')
    print('similarity_matrix shape: ', similarity_matrix.shape)

    uncertainty = mc_dropout(model, unlabeled_indices, data, device, k=k, with_diversity=True)
    uncertainty = uncertainty/uncertainty.max()
    uncertainty = uncertainty.cpu().numpy()

    ### setting upper bound lower bounds and constraints
    lb = np.zeros(len(unlabeled_indices))
    ub = np.ones(len(unlabeled_indices))

    A = np.ones(len(unlabeled_indices))
    b = 1.0 * np.array([k])
    
    # un_rep = uncertainty 

    #using osqp
    print(f'Solving qp....')

    res = solve_qp(P=similarity_matrix, q=uncertainty, G=None, A=A, b=b, lb=lb, ub=ub, solver='osqp', verbose=True)

    # Solve problem 
    
    arg = np.argsort(res)
    
    selected_indices = arg[-k:] ###gives index
    
    ''' ########### Here ############ '''
    unselected_indices = torch.tensor(np.setdiff1d(np.arange(similarity_matrix.shape[0]), selected_indices))
    
    #choose subset of diversity matrix containing unselected indices
    
    similarity_matrix = similarity_matrix[:,unselected_indices]
    similarity_matrix = similarity_matrix[unselected_indices,:]
    
    torch.save(similarity_matrix, f'runs/{selection_method}/run{expt}/tensor.pt')
    
    query_indices = np.array(unlabeled_indices)[arg[-k:]]
    return labeled_indices+query_indices.tolist(), time.perf_counter()-time_start

# GPU version
def coreset(model, labeled_indices, unlabeled_indices, data, device, k=1500):
    t0 = time.perf_counter()
    unlabeled_loader = GeoDataLoader(data, batch_size=64,
                        sampler=SubsetSequentialSampler(labeled_indices+unlabeled_indices), # more convenient if we maintain the order of subset
                        pin_memory=True, drop_last=False)
    
    #set model to evaluation mode
    model.eval()

    with torch.cuda.device(device):
        features = torch.tensor([]).to(device)

    with torch.no_grad():
        i=0
        for step, batch_data in enumerate(tqdm(unlabeled_loader)):
            batch_data = batch_data.to(device)
          
            with torch.no_grad():
                _, feature_dict, _, _ = model(batch_data)
                node_features = feature_dict['node_features']

            av_node_features = get_average_node_per_molecule(batch_data, node_features[-1], device)
            features = torch.cat((features, av_node_features), 0)

    print(features.shape)

    # Split features for train (labeled) and predictions (unlabeled)
    train_features = features[:len(labeled_indices)]
    pred_features = features[len(labeled_indices):]

    print(f'Train_shape: {train_features.shape}:::: un_shape: {pred_features.shape}')
    '''
    This functions takes feature values of labeled dataset and unlabeled dataset and return unlabeled indices based on the coreset method.
    '''
    
    # Convert indices to torch tensors
    subset_indices = torch.tensor(unlabeled_indices, device=device)

    query_indices = []
    
    # Calculate pairwise distances on GPU using torch functions
    # Distance between unlabeled samples
    unlabeled_pairwise_distance = pairwise_euclidean_distance(pred_features, pred_features)
    # Distance between unlabeled and labeled samples
    distance = pairwise_euclidean_distance(pred_features, train_features)
    
    for i in range(0, k):
        
        # Find minimum distance for each unlabeled point to any labeled point
        min_distances, _ = torch.min(distance, dim=1)
        
        # Find the point with maximum min-distance (farthest point)
        max_idx = torch.argmax(min_distances)
        
        # Get the actual index from our subset
        selected_idx = subset_indices[max_idx].item()  
        
        if selected_idx in labeled_indices:
            print('Already in labeled;;;;stop stop')
            break
            
        # Add the selected index to query set
        query_indices.append(selected_idx)
        
        # Update distances by adding the distances to the newly selected point
        # and removing the selected point from our candidate pool
        new_distances = unlabeled_pairwise_distance[max_idx].unsqueeze(1)
        distance = torch.cat((distance, new_distances), dim=1)
        
        # Remove the selected point from our pool of candidates
        mask = torch.ones(len(subset_indices), dtype=torch.bool, device=device)
        mask[max_idx] = False
        
        subset_indices = subset_indices[mask]
        distance = distance[mask]
        unlabeled_pairwise_distance = unlabeled_pairwise_distance[mask][:, mask]
    
    return labeled_indices+query_indices, time.perf_counter() - t0

def get_embedding(model, dataloader, device):
    #set model to evaluation mode
    model.eval()
    with torch.cuda.device(device):
        features = torch.tensor([]).to(device)
 
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(dataloader)):
            batch_data = batch_data.to(device)
         
            with torch.no_grad():
                _, feature_dict, _, _= model(batch_data)
                node_features = feature_dict['node_features']
 
            av_node_features = get_average_node_per_molecule(batch_data, node_features[-1], device)
            features = torch.cat((features, av_node_features), 0)
   
    return features
 
def get_labels(dataset):
    labels = []
    for d in dataset:
        labels.append(d.y)
    labels = np.concatenate(labels).tolist()
    return torch.tensor(labels).reshape(-1, 1)
 
def get_interval_ranges(k, range_min_value, range_max_value):
    assert range_max_value >= range_min_value, 'The maximum values of predictions must be larger or equal than the minimun values of the predictions.'
    ranges = np.linspace(range_min_value, range_max_value, k+1)
 
    return ranges

def get_predictive_distribution(unlabeled_prediction, Q, K, ranges, range_max_value):
    interval_values = np.zeros(shape=(K), dtype=int)
    for j in np.arange(unlabeled_prediction.shape[0]):
        # Each intervals:
        for i in np.arange(0, K, 1):
            start = ranges[i]
            end = ranges[i+1]
            if start <= unlabeled_prediction[j] < end:
                interval_values[i] += 1
            if unlabeled_prediction[j] == range_max_value and i+1 == K:
                interval_values[i] += 1
                
    assert np.sum(interval_values) == Q-1, 'Warning: The interval split has errors.'

    return interval_values/(Q-1)
    
def MQR_ud(model, unlabeled_indices, labeled_indices, data, device, k=1500, valid_dataset=None, expt=1, cycle=0, path=" "):
    time_start = time.perf_counter()
    input_size = 64
    hidden_sizes = [256]
    Q = 100
    K = 5
 
    model = model.to(device)
    quantile_model = MLP(input_size, hidden_sizes, 1, Q-1)
    quantiles = [q/Q for q in range(1, Q)]
 
    X_pool =  GeoDataLoader(data, batch_size=64,
                        sampler=SubsetSequentialSampler(unlabeled_indices),
                        pin_memory=True, drop_last=False)
 
    labeled_data = GeoDataLoader(data, batch_size=64,
                        sampler=SubsetSequentialSampler(labeled_indices),
                        pin_memory=True, drop_last=False)
 
    valid_loader = GeoDataLoader(valid_dataset, 64, shuffle=False)
 
    labeled_embeddings = torch.squeeze(get_embedding(model, labeled_data, device))
    valid_loader_embeddings = torch.squeeze(get_embedding(model, valid_loader, device))
    pool_embeddings = torch.squeeze(get_embedding(model, X_pool, device))
 
    labeled_subset = [data[i] for i in labeled_indices]
    labeled_labels = get_labels(labeled_subset)
    valid_labels = get_labels(valid_dataset)
   
    mu = labeled_labels.mean()
    sigma = labeled_labels.std().clamp(min=1e-8)

    labeled_labels = (labeled_labels - mu) / sigma
    valid_labels   = (valid_labels   - mu) / sigma
   
    print(f"Embedding shape: {labeled_embeddings.shape}, Labels shape: {labeled_labels.shape}")
   
    labeled_loader = TorchDataLoader(
        TensorDataset(labeled_embeddings, labeled_labels),
        batch_size=32, shuffle=True, drop_last=False
    )
   
    valid_loader_tensor = TorchDataLoader(
        TensorDataset(valid_loader_embeddings, valid_labels),
        batch_size=64, shuffle=False, drop_last=False
    )
 
    quantile_model = train_model(quantile_model, labeled_loader, valid_loader_tensor, path)
 
    utility = []
    quantile_model.eval()
    with torch.no_grad():
        pred = quantile_model(pool_embeddings)
        pred = pred.squeeze(1)
        pred = pred * sigma + mu
        numpy_pred = pred.cpu().numpy()

        pred_label = quantile_model(labeled_embeddings)
        pred_label = pred_label.squeeze(1)
        pred_label = pred_label * sigma + mu
        numpy_pred_label = pred_label.cpu().numpy()

        range_min_value = min(numpy_pred.min(), numpy_pred_label.min())
        range_max_value = max(numpy_pred.max(), numpy_pred_label.max())

        ranges = get_interval_ranges(K, range_min_value, range_max_value)
 
        # Each sample in the system:
        for i in range(numpy_pred.shape[0]):
            prob_original = get_predictive_distribution(numpy_pred[i,], Q, K, ranges, range_max_value)
            utility.append(entropy(prob_original))

    utility = np.array(utility)
    selected_indices, _ = K_means_Plus_Plus(pool_embeddings.cpu().numpy(), k, utility, seed_initial=expt*1000+cycle+999)
    unlabel = np.asarray(unlabeled_indices, dtype=np.int64).ravel()
    selected_indices = unlabel[selected_indices]

    return torch.tensor(selected_indices), (time.perf_counter()-time_start)

def K_means_Plus_Plus(seleceted_system, n_instances, top_all_utility, seed_initial):
        # K means++ 
        centers, center_id = kmeans_plusplus(seleceted_system, n_clusters=n_instances, n_local_trials=None, random_state=seed_initial)
        
        # calculate the distance of each sample to the centers
        distances = cdist(seleceted_system, seleceted_system[center_id])
        cluster_labels = np.argmin(distances, axis=1)
        
        # select the sample in the cluster that with the highest scores
        selected_indices = []
        selected_scores = []
        empty_clusters = []  # recording the empty cluster
        
        for i in range(n_instances):
            # Each cluster:
            cluster_points = np.where(cluster_labels == i)[0]
            if len(cluster_points) > 0:
                # get the sample with the highest score
                cluster_utilities = top_all_utility[cluster_points]
                best_point_idx = cluster_points[np.argmax(cluster_utilities)]
                selected_indices.append(best_point_idx)
                selected_scores.append(top_all_utility[best_point_idx])
            else:
                # record empty cluster
                empty_clusters.append(i)
        
        # # if the empty cluster
        # if empty_clusters:
        #     # print("Empty clusters occur.")
        #     # remaining samples that have not been selected
        #     remaining_indices = list(set(range(len(seleceted_system))) - set(selected_indices))
        #     if remaining_indices:
        #         # select the samples with highest scores in the remaining samples
        #         remaining_utilities = top_all_utility[remaining_indices]
        #         n_additional = len(empty_clusters)  # number of samples need to be selected
                
        #         # select the top-k samples with the highest scores
        #         additional_best_indices = np.argsort(remaining_utilities)[-n_additional:]
        #         for idx in additional_best_indices:
        #             selected_indices.append(remaining_indices[idx])
        #             selected_scores.append(remaining_utilities[idx])
        
        selected_indices = np.array(selected_indices)
        sample_acquisition = np.array(selected_scores)
        
        # check the shape of returned values
        assert len(selected_indices) == n_instances, f"Selected {len(selected_indices)} points, but required {n_instances}"
        
        return selected_indices, sample_acquisition

def gradient_conflicts(model, unlabeled_indices, labeled_indices, data, device, k=1500, valid_dataset=None, expt=1, cycle=0, path=" "):
    time_start = time.perf_counter()
    input_size = 64  # Embedding dimension
    hidden_sizes = [256]  # MQR model size
    Q = 100  # quantiles+1
    K = 5   # intervals
    quantiles = torch.tensor([q / Q for q in range(1, Q)], device=device)  # dimension: [Q-1]
    gt_estimation_method = "median"  # weighted_average or median to estimate the ground truth labels

    # Train MQR-MLP 
    model = model.to(device)
    quantile_model = MLP(input_size, hidden_sizes, 1, Q - 1).to(device)

    X_pool = GeoDataLoader(
        data, batch_size=64,
        sampler=SubsetSequentialSampler(unlabeled_indices),
        pin_memory=True, drop_last=False
    )
    labeled_data = GeoDataLoader(
        data, batch_size=64,
        sampler=SubsetSequentialSampler(labeled_indices),
        pin_memory=True, drop_last=False
    )
    valid_loader = GeoDataLoader(valid_dataset, 64, shuffle=False)

    labeled_embeddings = torch.squeeze(get_embedding(model, labeled_data, device))
    valid_loader_embeddings = torch.squeeze(get_embedding(model, valid_loader, device))
    pool_embeddings = torch.squeeze(get_embedding(model, X_pool, device))  # [N, input_size]

    labeled_subset = [data[i] for i in labeled_indices]
    labeled_labels = get_labels(labeled_subset).to(device)
    valid_labels = get_labels(valid_dataset).to(device)

    mu = labeled_labels.mean().detach()
    sigma = labeled_labels.std().clamp(min=1e-8).detach()
    labeled_labels = (labeled_labels - mu) / sigma
    valid_labels = (valid_labels - mu) / sigma

    labeled_loader = TorchDataLoader(
        TensorDataset(labeled_embeddings, labeled_labels),
        batch_size=32, shuffle=True, drop_last=False
    )
    valid_loader_tensor = TorchDataLoader(
        TensorDataset(valid_loader_embeddings, valid_labels),
        batch_size=64, shuffle=False, drop_last=False
    )

    # Probability and Distribution of intervals 
    quantile_model = train_model(quantile_model, labeled_loader, valid_loader_tensor, path)
    quantile_model = quantile_model.to(device)
    quantile_model.eval()
    with torch.no_grad():
        pred = quantile_model(pool_embeddings).squeeze(1)  # [N, Q-1]
        pred_scaled = pred * sigma + mu
        pred_np = pred_scaled.cpu().numpy()

        pred_label = quantile_model(labeled_embeddings).squeeze(1) * sigma + mu
        pred_label_np = pred_label.cpu().numpy()

        range_min = min(pred_np.min(), pred_label_np.min())
        range_max = max(pred_np.max(), pred_label_np.max())
        ranges = get_interval_ranges(K, range_min, range_max)

    # number of unlabelled samples
    N = pred_np.shape[0]
    # distribution of each possible labels
    prob_distributions = np.zeros((N, K), dtype=np.float32)
    # ground truth labels
    ground_truths = np.zeros((N, K), dtype=np.float32)

    if gt_estimation_method == "weighted_average":
        # For each unlabelled sample:
        for i in range(N):
            prob_distributions[i] = get_predictive_distribution(pred_np[i], Q, K, ranges, range_max)

            interval_id = np.digitize(pred_np[i], ranges, right=False) - 1 # From 0 to the last interval, only include the left endpoint
            interval_id[interval_id == K] = K - 1 # make sure the largest value (right end) belongs to the last interval

            median_preds = np.zeros(K, dtype=np.float32)
            for range_idx in range(K):
                # In each interval:
                idxs = np.where(interval_id == range_idx)[0]
                if idxs.size:
                    median_q_idx = idxs[idxs.size // 2]
                    median_preds[range_idx] = pred_np[i, median_q_idx] # The estimated ground truth label is the median value of each interval
                else:
                    median_preds[range_idx] = pred_np[i, (Q // 2) - 1] # If no samples fall into the interval, use the median value of the entire range
            ground_truths[i] = median_preds

    elif gt_estimation_method == "median":
        # The estimated ground truth label is the median value of the entire range
        median_vals = pred_np[:, (Q // 2) - 1]           # [N]
        ground_truths = np.tile(median_vals[:, None], (1, K))  # [N, K]

    prob_distributions_tensor = torch.tensor(prob_distributions, device=device, dtype=torch.float32)  # [N, K]
    ground_truths_tensor = torch.tensor(ground_truths, device=device, dtype=torch.float32)  # [N, K]

    # Split shared parameters and head parameters of MQR model
    params = dict(quantile_model.named_parameters())
    shared_params = {name: p for name, p in params.items() if name.startswith("layers")}
    head_params = {name: p for name, p in params.items() if name.startswith("final_layer")}
    shared_keys = list(shared_params.keys())

    # flatten all parameters into a single tensor
    def flatten_pytree(pytree):
        return torch.cat([pytree[k].reshape(-1) for k in shared_keys])

    # Calculate loss and use vmap to calculate gradient
    # forward function for MQR model (for the single point)
    def mqr_forward_with_shared(shared_params, x):
        """x: [input_size], return [Q-1] predictions of the single point"""
        all_params = {**shared_params, **head_params}
        out = functional_call(quantile_model, all_params, (x.unsqueeze(0),))
        return out.squeeze(0).squeeze(0)
 
    # calculate loss for the single quantile of the single point
    def single_quantile_loss(shared_params, x_i, gt_i, prob_i, q_idx_tensor):
        """
        q_idx_tensor: single tensor for vmap
        """
        pred_all = mqr_forward_with_shared(shared_params, x_i)  # [Q-1]
        pred_all_raw = pred_all * sigma + mu
        
        # the selected quantile prediction
        pred_q = torch.gather(pred_all_raw, 0, q_idx_tensor.view(1)).squeeze(0)
        # the selected quantile
        q_tau = torch.gather(quantiles, 0, q_idx_tensor.view(1)).squeeze(0)

        if gt_estimation_method == "weighted_average":
            # gt_i, prob_i: [K]
            errors = gt_i - pred_q    # [K]
            quantile_loss = 2 * torch.maximum(
                (q_tau - 1) * errors,
                q_tau * errors
            )  # [K]
            weighted_loss = (quantile_loss * prob_i).sum()
            return weighted_loss
        else:  # "median" case
            # gt_i has the same K values
            gt_scalar = gt_i[0]
            error = gt_scalar - pred_q    
            quantile_loss = 2 * torch.maximum(
                (q_tau - 1) * error,
                q_tau * error
            )
            return quantile_loss

    # Calculate the gradient of shared_params
    grad_fn = grad(single_quantile_loss, argnums=0)
    
    # Vmap to all quantiles (for single sample point)
    def compute_all_quantile_grads(shared_params, x_i, gt_i, prob_i):
        """calculate the gradient of shared_params for all quantiles"""
        q_indices = torch.arange(Q - 1, device=device, dtype=torch.long)
        
        # vmap over q_indices
        batched_grad_fn = vmap(
            lambda q_idx: grad_fn(shared_params, x_i, gt_i, prob_i, q_idx),
            in_dims=0
        )
        
        grads_pytree = batched_grad_fn(q_indices)        
        grad_list = []
        for q_idx in range(Q - 1):
            single_q_grads = {k: v[q_idx] for k, v in grads_pytree.items()}
            grad_vec = flatten_pytree(single_q_grads)
            grad_list.append(grad_vec)
        
        return torch.stack(grad_list, dim=0)  # [Q-1, D_shared]
    
    batched_sample_grad_fn = vmap(
        compute_all_quantile_grads,
        in_dims=(None, 0, 0, 0)
    )
    batch_size = 64  
    all_conflict_scores = []

    for batch_start in tqdm(range(0, N, batch_size), desc="Computing gradients"):
        batch_end = min(batch_start + batch_size, N)
        batch_x = pool_embeddings[batch_start:batch_end]         # [B, input_size]
        batch_gt = ground_truths_tensor[batch_start:batch_end]   # [B, K]
        batch_prob = prob_distributions_tensor[batch_start:batch_end]  # [B, K]

        B = batch_x.shape[0]
        
        try:
            # [B, Q-1, D_shared]
            grad_matrices = batched_sample_grad_fn(shared_params, batch_x, batch_gt, batch_prob)
        except RuntimeError as e:
            print(f"Warning: vmap failed, falling back to loop. Error: {e}")
            grad_matrices = []
            for i in range(B):
                grad_matrix = compute_all_quantile_grads(
                    shared_params, batch_x[i], batch_gt[i], batch_prob[i]
                )
                grad_matrices.append(grad_matrix)
            grad_matrices = torch.stack(grad_matrices, dim=0)

        # Deal with each sample gradient
        for sample_idx in range(B):
            grad_matrix = grad_matrices[sample_idx]  # [Q-1, D_shared]

            g = grad_matrix  # [T, D]
            g_norms = g.norm(p=2, dim=1)  # [T]
            g_dir = F.normalize(g, p=2, dim=1)  # [T, D]
            cos_sim = torch.mm(g_dir, g_dir.T)  # [T, T]

            mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
            cos_pairs = cos_sim[mask]

            # ----  c_ij = (1 - cos) / 2 ----
            angle_term = 0.5 * (1.0 - cos_pairs)

            # ---- Phi ----
            gi_norms_mat = g_norms.unsqueeze(1).expand_as(cos_sim)  # [T, T]
            gj_norms_mat = g_norms.unsqueeze(0).expand_as(cos_sim)  # [T, T]

            denominator = gi_norms_mat**2 + gj_norms_mat**2
            denominator_safe = torch.where(
                denominator < 1e-12,
                torch.full_like(denominator, 1e-8),
                denominator
            )

            phi_mat = 2.0 * gi_norms_mat * gj_norms_mat / denominator_safe
            phi_pairs = phi_mat[mask]

            # ---- d_ij = 1 - Phi ----
            mag_imbalance_ij = 1.0 - phi_pairs

            # ---- s_ij = c_ij + d_ij ----
            s_ij = 0.5 * (angle_term + mag_imbalance_ij)

            conflict_score = torch.topk(s_ij.reshape(-1), 10).values.sum().item()
            all_conflict_scores.append(conflict_score)

    utility = np.array(all_conflict_scores)
    utility = torch.tensor(utility, device=device)
    arg = torch.argsort(utility, descending=True)
    unlabeled_tensor = torch.as_tensor(unlabeled_indices, device=arg.device)
    return unlabeled_tensor[arg[:k]].cpu(), time.perf_counter() - time_start
