'''
@Author: Mingming Ren 任铭铭 
@Date: 2024-11-26 18:49:12 
@Last Modified by:   Mingming Ren ren 
@Last Modified time: 2025-07-06 18:49:12 
'''
import torch
import time
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
MAXSCALE = 5e8

DTYPE = torch.float32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=6)

def iukm(points, normals, indexs, logger, angle=38, iters=5, g=1, b=1):
    points = torch.tensor(points, dtype=DTYPE).to(DEVICE)
    normals = torch.tensor(normals, dtype=DTYPE).to(DEVICE)
    indexs = torch.tensor(indexs, dtype=DTYPE)

    N = len(points)
    gama = g
    beta = b
    t = 0
    alpha = 0.5
    size_n = int(N * alpha)
    while size_n * N >= MAXSCALE:
        alpha -= 0.03
        size_n = int(N*alpha)
    logger.info('The number of points is {}, and cluster centers is {}'.format(N, size_n))
    M = torch.arange(0, N)
    M = torch.randperm(len(M), device=DEVICE)[:size_n]
    max_M = torch.max(M)
    mask = torch.ones_like(M, dtype=torch.bool).to(DEVICE)
    C = len(M)
    C = torch.tensor(C).to(DEVICE)
    alpha = torch.ones(C, dtype=torch.float32, device=DEVICE) / C
    dic = torch.zeros(max_M+1, dtype=torch.int64, device=DEVICE)
    M_ = M.clone()
    for i in range(C):
        dic[M_[i]] = i
    M_n = normals[M, :]
    P_n = points[M, :] 
    p_n_c_d = points.unsqueeze(1) - P_n.unsqueeze(0) 
    p_ik_dot_M_n = torch.clamp(abs(torch.sum(p_n_c_d * M_n, dim=2)), min=0)
    M_n_dot_normals = torch.arccos(torch.abs(torch.sum(normals.unsqueeze(1) * M_n.unsqueeze(0), dim=2)))
    pool = p_ik_dot_M_n +  M_n_dot_normals
    del points, M_n, P_n, p_n_c_d, p_ik_dot_M_n, M_n_dot_normals
    torch.cuda.empty_cache()

    stop = False
    iter = 0
    while not stop:
        log_alpha = torch.log(alpha)
        gama_log_alpha = gama * log_alpha
        pool_ = pool[:, mask]
        res = pool_ - gama_log_alpha
        Z = torch.argmin(res, dim=1)
        if t > 0:
            Z = M[Z]
        counts = torch.bincount(Z, minlength=max_M+1)
        sum_Z = counts[M]
        part_1 = sum_Z / N
        part_2 = (beta/gama) * alpha * (log_alpha - torch.sum(alpha * log_alpha))
        alpha_new = part_1 + part_2

        d = 2
        tmp = (1 / t ** torch.floor(torch.tensor(d / 2 - 1, dtype=torch.float32))).int()
        nita = torch.minimum(torch.tensor(1, dtype=torch.int32), tmp)

        alpha = torch.maximum(alpha, torch.tensor(1e-10, dtype=torch.float32))
        log_alpha_tmp = torch.log(alpha)
        denominator = -1 * torch.max(alpha * torch.sum(log_alpha_tmp))
        denominator = torch.tensor(1e-10, dtype=torch.float32) if torch.abs(denominator) < 1e-10 else denominator

        p1 = torch.sum(torch.exp(-nita * N * torch.abs(alpha_new - alpha)))
        p2 = (1 - torch.max(torch.max(counts) / N)) / denominator
        beta = torch.minimum(p1, p2)
        alpha = alpha_new
        err  = torch.sum(alpha < 1/N)
        C = torch.max(torch.tensor(1, dtype=torch.int32), C - err)

        idxs = torch.where(alpha < 1 / N)[0]
        real_idx = M[idxs]
        loc = dic[real_idx]
        mask[loc] = 0
        M = M[alpha >= 1 / N]
        alpha = alpha[alpha >= 1 / N]
        alpha = alpha / torch.sum(alpha)

        t += 1
        iter += 1
        if iter > iters or C <= 1:
            break

    nn = normals[M,:]
    re = torch.arccos(torch.abs(torch.matmul(nn, nn.T)))
    t_d = angle * 3.14 / 180

    label = torch.zeros(re.shape[0])
    all_res = []
    for i in range(re.shape[0]):
        if label[i] == 1:
            continue
        tmp = torch.nonzero(re[i] < t_d)
        label[tmp] = 1
        all_res.append(M[tmp])

    clazz = []
    for elem in all_res:
        if len(elem) > 1:
            normal__ = normals[elem]
            normal_mean = torch.mean(normal__, axis=0)
            normal_mean = normal_mean / torch.linalg.norm(normal_mean)
            clazz.append(normal_mean.cpu())
        elif len(elem) == 1:
            clazz.append(normals[elem[0]].cpu())
        else:
            continue
    return clazz