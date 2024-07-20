import numpy as np
from src.networks import create_sppmi_mtx, construct_w_pkn, create_sppmi_mtx_torch, construct_w_pkn_torch, opt_p_torch, \
    wshrink_obj_torch
from src.networks import solve_l1l2, opt_p, wshrink_obj, solve_l1l2_torch
from tqdm import tqdm
from scipy.linalg import solve
import src.logger as l
import torch


def st_acn(expression, spatial_network, lamb=0.001, dim=200):
    expression = expression.T
    spatial_network = spatial_network.T

    # Data preparation and Variables init
    data = [spatial_network, expression]
    W = [None] * 2
    for i in range(2):
        data[i] = data[i] / np.tile(np.sqrt(np.sum(data[i] ** 2, axis=0)), (data[i].shape[0], 1))

    W[0] = create_sppmi_mtx(data[0], 1)
    W[1] = create_sppmi_mtx(construct_w_pkn(data[1], 20), 1)

    X = W
    V = len(X)
    N = X[0].shape[0]
    Eh = [np.zeros((dim, N)) for _ in range(V)]
    Yh = [np.zeros((dim, N)) for _ in range(V)]
    Ys = [np.zeros((N, N)) for _ in range(V)]
    Zv = [np.zeros((N, N)) for _ in range(V)]
    T = [np.zeros((N, N)) for _ in range(V)]

    sX = [N, N, V]
    P = [np.zeros((dim, N)) for _ in range(V)]

    mu = 1e-4
    pho = 2
    max_mu = 1e6
    max_iter = 50
    threshold = 1e-6
    pbar = tqdm(range(max_iter))
    for iter_ in pbar:
        B = np.array([]).reshape(0, Yh[0].shape[1])
        d = 0
        for i in range(V):
            XX = X[i] - np.dot(X[i], Zv[i])
            P[i] = opt_p(Yh[i], mu, Eh[i], XX)
            A = np.dot(P[i], X[i])
            Zv[i] = solve(np.dot(A.T, A) + np.eye(N),
                          np.dot(A.T, Yh[i] / mu) + np.dot(A.T, (A - Eh[i])) + T[i] - Ys[i] / mu)
            Zv[i] = (Zv[i] + Zv[i].T) / 2

            G = np.dot(P[i], X[i]) - np.dot(np.dot(P[i], X[i]), Zv[i]) + Yh[i] / mu
            B = np.vstack((B, G))
            E = solve_l1l2(B, lamb / mu)
            Eh[i] = E[d: (i + 1) * dim, :]
            d += dim

        Z_tensor = np.stack(Zv, axis=2)
        Ys_tensor = np.stack(Ys, axis=2)
        G_tensor = Z_tensor + 1 / mu * Ys_tensor
        t_tensor, objk = wshrink_obj(G_tensor, 1 / mu, sX, 0, 3)  #
        T_tensor = t_tensor.reshape(sX)

        for i in range(V):
            Zv[i] = Z_tensor[:, :, i]
            T[i] = T_tensor[:, :, i]
            Ys[i] = Ys_tensor[:, :, i]
        GG = []
        for i in range(V):
            GG.append(np.dot(P[i], X[i]) - np.dot(np.dot(P[i], X[i]), Zv[i]) - Eh[i])
            Yh[i] = Yh[i] + mu * (GG[i])
            Ys[i] = Ys[i] + mu * (Zv[i] - T[i])
        mu = min(pho * mu, max_mu)
        errp = np.zeros(V)
        errs = np.zeros(V)
        for i in range(V):
            errp[i] = np.linalg.norm(GG[i], ord=2)
            errs[i] = np.linalg.norm(Zv[i] - T[i], ord=2)
        max_err = np.max(errp + errs)

        if max_err <= threshold:
            pbar.set_description(f'[st_acn] Iter {iter_}: max_err={max_err:.4e} < thresh={threshold:.4e}. Breaking...')
            break

        formatted_errs = [f'{err:.4e}' for err in errs]
        pbar.set_description(f'[st_acn] Iter {iter_}: err={formatted_errs}, max_err={max_err:.4e}, thresh={threshold:.4e}')

    Z_all = np.zeros((N, N))
    for i in range(V):
        Z_all = Z_all + (np.abs(Zv[i]) + np.abs(Zv[i].T))
    Z_all = Z_all / V

    return Z_all


def st_acn_gpu(expression, spatial_network, threshold=1e-6, lamb=0.001, dim=200, device='cpu'):
    expression = expression.T
    spatial_network = spatial_network.T
    data = [spatial_network, expression]
    W = [None] * 2
    for i in range(2):
        data[i] = data[i] / np.tile(np.sqrt(np.sum(data[i] ** 2, axis=0)), (data[i].shape[0], 1))

    W[0] = create_sppmi_mtx(data[0], 1)
    W[1] = create_sppmi_mtx(construct_w_pkn(data[1], 20), 1)

    W[0] = torch.from_numpy(W[0]).to(torch.float32).to(device)
    W[1] = torch.from_numpy(W[1]).to(torch.float32).to(device)

    V = len(W)
    N = W[0].shape[0]
    Eh = [torch.zeros(dim, N).to(device) for _ in range(V)]
    Yh = [torch.zeros(dim, N).to(device) for _ in range(V)]
    Ys = [torch.zeros(N, N).to(device) for _ in range(V)]
    Zv = [torch.zeros(N, N).to(device) for _ in range(V)]
    T = [torch.zeros(N, N).to(device) for _ in range(V)]
    P = [torch.zeros(dim, N).to(device) for _ in range(V)]

    sX = [N, N, V]

    mu = torch.tensor(1e-4, dtype=torch.float32, device=device)
    pho = torch.tensor(2., dtype=torch.float32, device=device)
    max_mu = torch.tensor(1e6, dtype=torch.float32, device=device)
    max_iter = 50

    pbar = tqdm(range(max_iter))
    for iter_ in pbar:
        B = torch.empty(0, Yh[0].shape[1]).to(device)
        d = 0
        for i in range(V):
            XX = W[i] - torch.mm(W[i], Zv[i])
            P[i] = opt_p_torch(Yh[i], mu, Eh[i], XX)
            A = torch.mm(P[i], W[i])
            Zv[i] = torch.linalg.solve(torch.mm(A.t(), A) + torch.eye(N, device=device),
                          torch.mm(A.T, Yh[i] / mu) + torch.mm(A.t(), (A - Eh[i])) + T[i] - Ys[i] / mu)
            Zv[i] = (Zv[i] + Zv[i].t()) / 2

            G = torch.mm(P[i], W[i]) - torch.mm(torch.mm(P[i], W[i]), Zv[i]) + Yh[i] / mu
            B = torch.cat((B, G), dim=0)
            E = solve_l1l2_torch(B, lamb / mu)
            Eh[i] = E[d: (i + 1) * dim, :]
            d += dim

        Z_tensor = torch.stack(Zv, dim=2)
        Ys_tensor = torch.stack(Ys, dim=2)
        G_tensor = Z_tensor + 1 / mu * Ys_tensor

        t_tensor, objk = wshrink_obj_torch(G_tensor, 1 / mu, sX, False, 3)  #
        T_tensor = t_tensor.reshape(sX)

        for i in range(V):
            Zv[i] = Z_tensor[:, :, i]
            T[i] = T_tensor[:, :, i]
            Ys[i] = Ys_tensor[:, :, i]
        GG = []
        for i in range(V):
            GG.append(torch.mm(P[i], W[i]) - torch.mm(torch.mm(P[i], W[i]), Zv[i]) - Eh[i])
            Yh[i] = Yh[i] + mu * (GG[i])
            Ys[i] = Ys[i] + mu * (Zv[i] - T[i])
        mu = torch.min(pho * mu, max_mu)
        errp = torch.zeros(V, device=device)
        errs = torch.zeros(V, device=device)

        for i in range(V):
            errp[i] = torch.norm(GG[i], p=2)
            errs[i] = torch.norm(Zv[i] - T[i], p=2)
        max_err = torch.max(errp + errs)

        if max_err <= threshold:
            pbar.set_description(f'[st_acn] Iter {iter_}: max_err={max_err:.4e} < thresh={threshold:.4e}. Breaking...')
            break

        formatted_errs = [f'{err:.4e}' for err in errs]
        pbar.set_description(f'[st_acn] Iter {iter_}: err={formatted_errs}, max_err={max_err:.4e}, thresh={threshold:.4e}')

    Z_all = torch.zeros(N, N).to(device)
    for i in range(V):
        Z_all = Z_all + (torch.abs(Zv[i]) + torch.abs(Zv[i].t()))
    Z_all = Z_all / V
    return Z_all
