# BSD 2-Clause License

# Copyright (c) 2022, Lun Wang
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
    Robust estimator for Federated Learning.
'''
import torch
import argparse
import cvxpy as cvx
# import numpy as np
import cupy as np
from numpy.random import multivariate_normal
from cupy.linalg import eigh
from cupyx.scipy.special import rel_entr
from sklearn.preprocessing import normalize
import math 

MAX_ITER = 100
ITV = 1000

def ex_noregret_(samples, eps=1./12, sigma=1, expansion=20, dis_threshold=0.7):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = len(samples)
    f = int(np.ceil(eps*size))
    metric = krum_(list(samples), f)
    indices = np.argpartition(metric, -f)[:-f]
    samples = samples[indices]
    size = samples.shape[0]
    
    dis_list = []
    for i in range(size):
        for j in range(i+1, size):
            dis_list.append(np.linalg.norm(samples[i]-samples[j]))
    dis_list = np.array(dis_list)
    step_size = 0.5 / (np.amax(dis_list) ** 2)
    size = samples.shape[0]
    feature_size = samples.shape[1]
    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)
    for i in range(int(2 * eps * size)):
        avg = np.average(samples, axis=0, weights=c)
        cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
        eig_val, eig_vec = eigh(cov)
        eig_val = eig_val[0]
        eig_vec = eig_vec.T[0]

        if eig_val * eig_val <= expansion * sigma * sigma:
            return avg

        tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
        c = c * (1 - step_size * tau)

        # The projection step
        ordered_c_index = np.flip(np.argsort(c))
        min_KL = None
        projected_c = None
        for i in range(len(c)):
            c_ = np.copy(c)
            for j in range(i+1):   
                c_[ordered_c_index[j]] = 1./(1-eps)/len(c)
            clip_norm = 1 - np.sum(c_[ordered_c_index[:i+1]])
            norm = np.sum(c_[ordered_c_index[i+1:]])
            if clip_norm <= 0:
                break
            scale = clip_norm / norm
            for j in range(i+1, len(c)):
                c_[ordered_c_index[j]] = c_[ordered_c_index[j]] * scale
            if c_[ordered_c_index[i+1]] > 1./(1-eps)/len(c):
                continue
            KL = np.sum(rel_entr(c, c_))
            if min_KL is None or KL < min_KL:
                min_KL = KL
                projected_c = c_

        c = projected_c
        
    avg = np.average(samples, axis=0, weights=c)
    return avg

def ex_noregret(samples, eps=1./12, sigma=1, expansion=20, itv=ITV):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    samples = np.array(samples)
    feature_shape = samples[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())
    samples_flatten = np.array(samples_flatten)
    feature_size = samples_flatten.shape[1]
    if itv is None:
        itv = int(np.floor(np.sqrt(feature_size)))
    cnt = int(feature_size // itv)
    sizes = []
    for i in range(cnt):
        sizes.append(itv)
    if feature_size % itv:
        sizes.append(feature_size - cnt * itv)

    idx = 0
    res = []
    cnt = 0
    for size in sizes:
        cnt += 1
        res.append(ex_noregret_(samples_flatten[:,idx:idx+size], eps, sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)

def mom_ex_noregret(samples, eps=0.2, sigma=1, expansion=20, itv=ITV, delta=np.exp(-30)):
    bucket_num = int(np.floor(eps * len(samples)) + np.log(1. / delta))
    bucket_size = int(np.ceil(len(samples) * 1. / bucket_num))

    bucketed_samples = []
    for i in range(bucket_num):
        bucketed_samples.append(np.mean(samples[i*bucket_size:min((i+1)*bucket_size, len(samples))], axis=0))
    return ex_noregret(bucketed_samples, eps, sigma, expansion, itv)





def rand_filterL2_(samples, eps=0.2, sigma=1, expansion=20):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = samples.shape[0]
    feature_size = samples.shape[1]
    # print(samples.shape)
    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)
    for i in range(max(2 * int(eps * size), 10)):
        # print(i)
        avg = np.average(samples, axis=0, weights=c)
        cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
        eig_val, eig_vec = eigh(cov)
        eig_val = eig_val[0]
        eig_vec = eig_vec.T[0]
        tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
        # print(eig_val)
        # print(tau.shape)
        # print(cov.shape)
        if eig_val * eig_val <= expansion * sigma * sigma:
            return avg
        
        
        tau_max_idx = np.argmax(tau)
        tau_max = tau[tau_max_idx]
        c = c * (1 - tau/tau_max)

        samples = np.concatenate((samples[:tau_max_idx], samples[tau_max_idx+1:]))
        samples_ = samples.reshape(-1, 1, feature_size)
        c = np.concatenate((c[:tau_max_idx], c[tau_max_idx+1:]))
        c = c / np.linalg.norm(c, ord=1)

    avg = np.average(samples, axis=0, weights=c)
    return avg


def rand_filterL2(samples, eps=0.2, sigma=1, expansion=20, itv=ITV):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    samples = np.array(samples)
    feature_shape = samples[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())
    samples_flatten = np.array(samples_flatten)
    # print(samples_flatten.shape)
    feature_size = samples_flatten.shape[1]
    if itv is None:
        itv = int(np.floor(np.sqrt(feature_size)))
    cnt = int(feature_size // itv)
    sizes = []
    for i in range(cnt):
        sizes.append(itv)
    if feature_size % itv:
        sizes.append(feature_size - cnt * itv)

    idx = 0
    res = []
    print(samples.shape)
    for size in sizes:
        res.append(filterL2_(samples_flatten[:,idx:idx+size], eps, sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)



def filterL2_(samples, eps=0.2, sigma=1, expansion=20):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = samples.shape[0]
    feature_size = samples.shape[1]
    # print(samples.shape)
    samples_ = samples.reshape(size, 1, feature_size)
    # print(samples_.shape)

    c = np.ones(size)
    for i in range(max(2 * int(eps * size), 10)):
        # print(i)
        avg = np.average(samples, axis=0, weights=c)
        cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
        eig_val, eig_vec = eigh(cov)
        eig_val = eig_val[0]
        eig_vec = eig_vec.T[0]
        tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
        # print(eig_val)
        # print(tau.shape)
        # print(cov.shape)
        if eig_val * eig_val <= expansion * sigma * sigma:
            return avg
        
        
        tau_max_idx = np.argmax(tau)
        tau_max = tau[tau_max_idx]
        c = c * (1 - tau/tau_max)

        samples = np.concatenate((samples[:tau_max_idx], samples[tau_max_idx+1:]))
        samples_ = samples.reshape(-1, 1, feature_size)
        c = np.concatenate((c[:tau_max_idx], c[tau_max_idx+1:]))
        c = c / np.linalg.norm(c, ord=1)

    avg = np.average(samples, axis=0, weights=c)
    return avg


def filterL2(samples, eps=0.2, sigma=1, expansion=20, itv=ITV):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    samples = np.array(samples)
    feature_shape = samples[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())
    samples_flatten = np.array(samples_flatten)
    # print(samples_flatten.shape)
    feature_size = samples_flatten.shape[1]
    if itv is None:
        itv = int(np.floor(np.sqrt(feature_size)))
    cnt = int(feature_size // itv)
    sizes = []
    for i in range(cnt):
        sizes.append(itv)
    if feature_size % itv:
        sizes.append(feature_size - cnt * itv)

    idx = 0
    res = []
    print(samples.shape)
    for size in sizes:
        res.append(filterL2_(samples_flatten[:,idx:idx+size], eps, sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)


def power_iteration(mat, iterations, device):
    # mat is a square matrix
    dim = mat.shape[0]
    # initial starting point
    u = torch.randn((dim, 1)).to(device)
    # u = torch.tensor([1.1]*dim).to(device)
    for _ in range(iterations):
        u = mat @ u / torch.linalg.norm(mat @ u) 
        # u = torch.nn.functional.normalize(u,dim=0) 
        # print(u.shape)
    eigenvalue = u.T @ mat @ u
    return eigenvalue, u

def randomized_agg(data, eps_poison=0.1, eps_jl=0.1, eps_pow = 0.1, threshold = 0.1, clean_eigen = 1, device = 'cuda:0'):
    # print(data.shape)
    n = int(data.shape[0])
    feature_shape = data[0].shape
    n_dim = int(np.prod(np.array(feature_shape)))
    # data = torch.reshape(data, (n, n_dim))
    res =  _randomized_agg(data, eps_poison, eps_jl, eps_pow, threshold, clean_eigen, device) # return filtered data
    # print(res.shape)
    # print(data.shape)
    # print(res.shape)
    return res
def _randomized_agg(data, eps_poison=0.1, eps_jl=0.1, eps_pow = 0.1, threshold = 100, clean_eigen = 10000, device = 'cuda:0'):
    torch.manual_seed(1)
    
    n = int(data.shape[0])
    data = data.to(device)
    
    # data is n times d, A is d times k
    d = int(math.prod(data[0].shape))
    data_flatten = data.reshape(n, d)
    data_mean = torch.mean(data_flatten, dim=0)
    data_sd = torch.std(data_flatten, dim=0)
    data_norm = (data_flatten - data_mean)/data_sd
    
    k = min(int(math.log(d)//eps_jl**2), d)
    print(f"d = {d}, k = {k}")
    
    A = torch.randn((d, k)).to(device)
    A = A/(k**0.5)

    Y = data_flatten @ A # n times k
    Y = Y.to(device)
    power_iter_rounds = int(- math.log(4*k)/(2*math.log(1-eps_pow)))
    # power_iter_rounds = d
    clean_eigen = clean_eigen * d/k
    # print(clean_eigen)
    # print(power_iter_rounds)
    curr_wt = torch.tensor([1/n]*n).to(device)
    for _ in range(max(int(eps_poison*n), 10)):
        Y_mean = torch.mean(Y, dim=0)
        Y_sd = torch.std(Y, dim=0)
        Y = (Y - Y_mean)#/Y_sd
        # Y_sq = Y.T @ Y # k times k
        # Y_sq.to(device)
        # Y_cov = torch.cov(Y.T)
        Y_sq = Y.T @Y
        eigenvalue, eigenvector = power_iteration(Y_sq, power_iter_rounds, device)
        if eigenvalue < threshold * clean_eigen:
            break

        # project Y on eigenvector
        proj_Y = (Y @ eigenvector)**2
        # return (proj_Y)
        # print(proj_Y)
        max_proj = max(proj_Y)
        weight = 1 - proj_Y/max_proj
        curr_wt = torch.flatten(curr_wt) * torch.flatten(weight)
        curr_wt = curr_wt / torch.sum(curr_wt)
        # print("ssdsadsdds")
        # print(proj_Y.shape)
        # print(eigenvalue)
        # print(eigenvector)
        # print(eigenvector.shape)
        # print(threshold*clean_eigen)
        # is_inlier = torch.flatten(torch.abs(proj_Y) < threshold*clean_eigen)
        # print(is_inlier)
        # # print(is_inlier)
        # # return eigenvector
        # # print(torch.max(proj_Y))
        # if torch.sum(is_inlier) == Y.shape[0]:
        #     break
        
        # # take the inliers
        # Y = Y[is_inlier]
        # data = data[is_inlier]
        # print(data.shape)
        # print(curr_wt)
        # print(Y.shape)
        # print(curr_wt.shape)
        for rr in range(n):
            Y[rr] = Y[rr]*curr_wt[rr]
    # print(data)
    # perform col wise mean on data

    weighted_avg = curr_wt @ data_flatten
    return weighted_avg.reshape(data[0].shape)
    # return torch.mean(data, dim = 0) * data_sd + data_mean


def mom_filterL2(samples, eps=0.2, sigma=1, expansion=20, itv=ITV, delta=np.exp(-30)):
    bucket_num = int(np.floor(eps * len(samples)) + np.log(1. / delta))
    bucket_size = int(np.ceil(len(samples) * 1. / bucket_num))

    bucketed_samples = []
    for i in range(bucket_num):
        bucketed_samples.append(np.mean(samples[i*bucket_size:min((i+1)*bucket_size, len(samples))], axis=0))

    return filterL2(bucketed_samples, eps, sigma, expansion, itv)

def median(samples):
    return np.median(samples, axis=0)

def trimmed_mean(samples, beta=0.1):
    samples = np.array(samples)
    average_grad = np.zeros(samples[0].shape)
    size = samples.shape[0]
    beyond_choose = int(size * beta)
    samples = np.sort(samples, axis=0)
    samples = samples[beyond_choose:size-beyond_choose]
    average_grad = np.average(samples, axis=0)

    return average_grad

def krum_(samples, f):
    size = len(samples)
    size_ = size - f - 2
    metric = []
    for idx in range(size):
        sample = samples[idx]
        samples_ = samples.copy()
        del samples_[idx]
        dis = np.array([np.linalg.norm(sample-sample_) for sample_ in samples_])
        metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
    return np.array(metric)

def krum(samples, f):
    metric = krum_(samples, f)
    index = np.argmin(metric)
    return samples[index], index

def mom_krum(samples, f, bucket_size=3):
    bucket_num = int(np.ceil(len(samples) * 1. / bucket_size))

    bucketed_samples = []
    for i in range(bucket_num):
        bucketed_samples.append(np.mean(samples[i*bucket_size:min((i+1)*bucket_size, len(samples))], axis=0))
    return krum(bucketed_samples, f=f)[0]

def bulyan_median(arr):
    arr_len = len(arr)
    distances = np.zeros([arr_len, arr_len])
    for i in range(arr_len):
        for j in range(arr_len):
            if i < j:
                distances[i, j] = abs(arr[i] - arr[j])
            elif i > j:
                distances[i, j] = distances[j, i]
    total_dis = np.sum(distances, axis=-1)
    median_index = np.argmin(total_dis)
    return median_index, distances[median_index]

def bulyan_one_coordinate(arr, beta):
    _, distances = bulyan_median(arr)
    median_beta_neighbors = arr[np.argsort(distances)[:beta]]
    return np.mean(median_beta_neighbors)

def bulyan(grads, f, aggsubfunc='trimmedmean'):
    samples = np.array(grads)
    feature_shape = grads[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())

    grads_num = len(samples_flatten)
    theta = grads_num - 2 * f
    # bulyan cannot do the work here when theta <= 0. Actually, it assumes n >= 4 * f + 3
    selected_grads = []
    # here, we use krum as sub algorithm
    if aggsubfunc == 'krum':
        for i in range(theta):
            krum_grads, _ = krum(samples_flatten, f)
            selected_grads.append(krum_grads)
            for j in range(len(samples_flatten)):
                if samples_flatten[j] is krum_grads:
                    del samples_flatten[j]
                    break
    elif aggsubfunc == 'median':
        for i in range(theta):
            median_grads = median(samples_flatten)
            selected_grads.append(median_grads)
            min_dis = np.inf
            min_index = None
            for j in range(len(samples_flatten)):
                temp_dis = np.linalg.norm(median_grads - samples_flatten[j])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    min_index = j
            assert min_index != None
            del samples_flatten[min_index]
    elif aggsubfunc == 'trimmedmean':
        for i in range(theta):
            trimmedmean_grads = trimmed_mean(samples_flatten)
            selected_grads.append(trimmedmean_grads)
            min_dis = np.inf
            min_index = None
            for j in range(len(samples_flatten)):
                temp_dis = np.linalg.norm(trimmedmean_grads - samples_flatten[j])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    min_index = j
            assert min_index != None
            del samples_flatten[min_index]

    beta = theta - 2 * f
    np_grads = np.array([g.flatten().tolist() for g in selected_grads])

    grads_dim = len(np_grads[0])
    selected_grads_by_cod = np.zeros([grads_dim, 1])  # shape of torch grads
    for i in range(grads_dim):
        selected_grads_by_cod[i, 0] = bulyan_one_coordinate(np_grads[:, i], beta)

    return selected_grads_by_cod.reshape(feature_shape)
