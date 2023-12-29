import os
import logger as l
import scanpy as sc
import pandas as pd
import pickle

def load_data_for_h5(input_data_path: str, section_id: str):
    # 拼装输入数据路径
    input_dir = os.path.join(input_data_path, section_id)
    l.logger.info(f"[LoadData]input dir {input_dir} ")
    # 加载 filtered_feature_bc_matrix.h5

    file_name_for_h5 = 'filtered_feature_bc_matrix.h5'
    file_name_for_filtered_feature_bc_matrix = f'{section_id}_{file_name_for_h5}'
    l.logger.info(f'[LoadData] {file_name_for_filtered_feature_bc_matrix} loading ...')

    AnnData = sc.read_visium(path=input_dir, count_file=file_name_for_filtered_feature_bc_matrix)
    AnnData.var_names_make_unique()
    l.logger.info(f'[LoadData] {file_name_for_filtered_feature_bc_matrix} load success')

    # 数据预处理
    sc.pp.filter_genes(AnnData, min_cells=10)
    sc.pp.highly_variable_genes(AnnData, flavor='seurat_v3', n_top_genes=3000)
    hvg_filter = AnnData.var['highly_variable']
    sc.pp.normalize_total(AnnData, inplace=True)
    AnnData = AnnData[:, hvg_filter]
    l.logger.info(f'[LoadData] {file_name_for_filtered_feature_bc_matrix} pre-process success')
    return AnnData


def load_data_for_ground_truth(input_data_path: str, section_id: str, anndata):
    truth_name = 'truth.txt'
    ann_df_file_path = os.path.join(input_data_path, section_id, f'{section_id}_{truth_name}')

    l.logger.info(f'[load_data_for_truth] {input_data_path}')

    ann_df = pd.read_csv(ann_df_file_path, sep='\t', header=None,
                         index_col=0)
    ann_df.columns = ['Ground Truth']
    anndata.obs['Ground Truth'] = ann_df.loc[anndata.obs_names, 'Ground Truth']
    ground_truth = pd.Categorical(anndata.obs['Ground Truth']).codes

    l.logger.info(f'[load_data_for_truth] {input_data_path} success')
    return ground_truth


def load_by_pkl(src_obj_file_path):
    if not os.path.exists(src_obj_file_path):
        return None, False
    else:
        with open(src_obj_file_path,"rb") as file:
            obj = pickle.load(file)
    return obj, True

def dump_by_pkl(obj,dst_obj_file_path):
    with open(dst_obj_file_path, "wb") as file:
        pickle.dump(obj, file)
