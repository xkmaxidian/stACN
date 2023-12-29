import os.path
import cluster
import numpy as np
from src.networks import create_sppmi_mtx,construct_w_pkn
from src.networks import solve_l1l2, opt_p, wshrink_obj
from tqdm import tqdm
from  scipy.linalg import solve
import logger as l
import load_data
from scipy.io import loadmat
from load_data import load_by_pkl, dump_by_pkl

def st_acn(expression, spatial_network, lamb=0.001, dim=200,is_reload=False ,output_path="output"):
    expression = expression.T
    spatial_network = spatial_network.T

    # Data preparation and Variables init
    data = [spatial_network, expression]
    W = [None] * 2

    # 归一化
    w_dump = os.path.join(output_path,"w_dump.pkl")
    if is_reload:
        W ,ok_W =  load_by_pkl(w_dump)

    if not ok_W:
        for i in range(2):
            data[i] =  data[i] / np.tile(np.sqrt(np.sum(data[i] ** 2, axis=0)), (data[i].shape[0], 1))
            W[i] = create_sppmi_mtx(construct_w_pkn(data[i], 20), 1)
        dump_by_pkl(W,w_dump)

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
    thresh = 1e-6

    for iter_ in tqdm(range(max_iter)):
        B = np.array([]).reshape(0, Yh[0].shape[1])
        d = 0
        l.logger.info(f'[st_acn]iter = {iter_} calculate E')
        for i in range(V):
            XX = X[i] -  np.dot(X[i],Zv[i])
            P[i] = opt_p(Yh[i], mu, Eh[i], XX)
            A = np.dot(P[i] , X[i])
            Zv[i] = solve(np.dot(A.T , A) + np.eye(N), np.dot(A.T , Yh[i] / mu) + np.dot(A.T , (A - Eh[i])) + T[i] - Ys[i] / mu)
            Zv[i] = (Zv[i] + Zv[i].T) / 2

            G = np.dot( P[i] , X[i]) -  np.dot(np.dot(P[i] , X[i]) , Zv[i]) + Yh[i] / mu
            B = np.vstack((B, G))
            E = solve_l1l2(B, lamb / mu)
            Eh[i] = E[d: (i+1) * dim,:]
            d += dim
        l.logger.info(f'[st_acn]iter = {iter_} calculate E end')

        Z_tensor = np.stack(Zv, axis=2)
        Ys_tensor = np.stack(Ys, axis=2)
        l.logger.info(f'[st_acn]iter = {iter_} wshrink_obj')
        G_tensor = Z_tensor + 1/mu * Ys_tensor
        t_tensor,objk = wshrink_obj(G_tensor, 1 / mu, sX, 0, 3) #
        T_tensor = t_tensor.reshape(sX)

        for i in range(V):
            Zv[i] = Z_tensor[:, :, i]
            T[i] = T_tensor[:, :, i]
            Ys[i] = Ys_tensor[:, :, i]
        GG = []
        for i in range(V):
            GG.append( np.dot(P[i] , X[i]) - np.dot(np.dot(P[i] , X[i]) , Zv[i]) - Eh[i])
            Yh[i] = Yh[i] + mu * (GG[i])
            Ys[i] = Ys[i] + mu * (Zv[i] - T[i])
        mu = min(pho * mu, max_mu)
        errp = np.zeros(V)
        errs = np.zeros(V)
        l.logger.info(f'[st_acn]iter = {iter_} calculate errs')
        for i in range(V):
            errp[i] = np.linalg.norm(GG[i], ord=2)
            errs[i] = np.linalg.norm(Zv[i] - T[i],ord=2)
        max_err = np.max(errp + errs)

        if max_err <=  thresh:
            l.logger.info(f'[st_acn]iter = {iter_} max_err={max_err} < {thresh} break')
            break

        formatted_errs = [f'{err:.4e}' for err in errs]
        l.logger.info(f'[st_acn]iter = {iter_}  err={formatted_errs} max_err={max_err:.4e} thresh{thresh:.4e} ')

    Z_all = np.zeros((N, N))
    for i in range(V):
        Z_all = Z_all + (np.abs(Zv[i]) + np.abs(Zv[i].T))
    Z_all = Z_all / V

    return Z_all

if __name__ == '__main__':

    mat = loadmat('D:/MATLAB/bin/MVC_SC/MCSL-LTC-Code/scTData/151675.mat')
    data = mat['data']
    gt = mat['gt'].T
    spatial_network = data[0][0][:3592].astype(np.float32)
    expression = data[0][1][:3592].astype(np.float32)
    output_data_path = "./Output/"
    log_path = os.path.join(output_data_path, 'log')
    l.initlog(log_path, )
    l.logger.info(f'[call] run st_acn_master')
    Z_all = st_acn(expression, spatial_network, gt, lamb=0.001, dim=100)
    l.logger.info(f'[call] run st_acn_master done')
    input_data_path = "D:/E/Work_ML/UGIMC/Data/"
    section_id = '151675'
    AnnData = load_data.load_data_for_h5(input_data_path, section_id)
    ground_truth = load_data.load_data_for_ground_truth(input_data_path, section_id, AnnData)
    cluster.cluster(Z_all, ground_truth)








