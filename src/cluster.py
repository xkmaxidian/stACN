import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import igraph as ig
import leidenalg
from natsort import natsorted
import pandas as pd
import logger as l

# 聚类
def cluster(Z_all, ground_truth):
    l.logger.info("[cluster] begin")
    learned_graph_from_matlab = Z_all
    sources, targets = learned_graph_from_matlab.nonzero()
    ans_weight = learned_graph_from_matlab[sources, targets]
    g = ig.Graph(directed=True)
    g.add_vertices(learned_graph_from_matlab.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    g.es['weight'] = ans_weight

    partition_type = leidenalg.RBConfigurationVertexPartition
    # clustering proper
    partition_kwargs = {'weights': np.array(g.es['weight']).astype(np.float64), 'n_iterations': -1, 'seed': 42,
                        'resolution_parameter': 1.1}

    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    # store output into adata.obs
    groups = np.array(part.membership)
    leiden_label = pd.Categorical(
        values=groups.astype('U'),
        categories=natsorted(map(str, np.unique(groups))),
    )
    ari = adjusted_rand_score(ground_truth, leiden_label)
    l.logger.info(f"[cluster] done ari={ari}")
