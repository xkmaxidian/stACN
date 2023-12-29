# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os
import warnings
import logger as l
import st_acn
import pickle
from load_data import load_data_for_h5, load_by_pkl, load_data_for_ground_truth, dump_by_pkl
import cluster

warnings.filterwarnings("ignore")
# 配置日志记录
from src import enhence

input_data_path: str = ""
section_id: str = ""
output_data_path: str = ""
log_path: str = ""
is_reload: bool = False


def parse_args():
    # common parser
    parser = argparse.ArgumentParser(description='example')
    # 添加位置参数
    parser.add_argument('input_data_path', type=str, help='path of input data')
    parser.add_argument('section_id', type=str, help="section_id")
    # 添加可选参数
    parser.add_argument('--output_data_path', type=str, help='path of input data', default="output", required=False)
    parser.add_argument('--is_reload', type=bool, help="is_reload if true reload last data", default=False,
                        required=False)

    # 解析命令行参数
    args = parser.parse_args()
    return args


# 算法核心逻辑
def call(low_dim_x, cell_spatial, ground_truth, output_path: str, is_reload: bool):
    l.logger.info(f'[call] begin ')

    z_all_dump = os.path.join(output_path, "z_all.pkl")
    if is_reload:
        l.logger.info(f'[call] loading z_all from source path {output_path}...')
        z_all, ok_z_all = load_by_pkl(z_all_dump)
        if ok_z_all:
            l.logger.info(f'[call] loading z_all from source path {output_path} done')
        l.logger.info(f'[call]loading z_all from source path {output_path} failed')

    if not ok_z_all:
        l.logger.info(f'[call] run st_acn_master begin...')
        z_all = st_acn.st_acn(low_dim_x, cell_spatial.A, output_path=output_path, is_reload=is_reload)
        dump_by_pkl(z_all, z_all_dump)
        l.logger.info(f'[call] run st_acn_master done')

    l.logger.info(f'[call] run cluster begin...')
    cluster.cluster(z_all, ground_truth)
    l.logger.info(f'[call] run cluster done')


def process_data(input_data_path: str, section_id: str, output_path: str, is_reload: bool):
    l.logger.info(f'[process_data] begin ')
    low_dim_x_dump = os.path.join(output_path, "low_dim_x.pkl")
    enhanced_adata_dump = os.path.join(output_path, 'enhanced_adata.pkl')
    cell_spatial_dump = os.path.join(output_path, 'cell_spatial.pkl')
    ground_truth_dump = os.path.join(output_path, 'ground_truth.pkl')

    if is_reload:
        l.logger.info(f'[process_data] try load from pkl path={output_path}')
        low_dim_x, ok_low_dim_x = load_by_pkl(low_dim_x_dump)
        enhanced_adata, ok_enhanced_adata = load_by_pkl(enhanced_adata_dump)
        cell_spatial, ok_cell_spatial = load_by_pkl(cell_spatial_dump)
        ground_truth, ok_ground_truth = load_by_pkl(ground_truth_dump)

        if ok_low_dim_x and ok_enhanced_adata and ok_cell_spatial and ok_ground_truth:
            l.logger.info(f'[process_data] try load from pkl path={output_path} success')
            return low_dim_x, enhanced_adata, cell_spatial, ground_truth
    l.logger.info(f'[process_data] try load from pkl path={output_path} failed')

    #  加载数据
    l.logger.info(f'[process_data] loading data section_id={section_id} from source path {input_data_path} begin...')
    AnnData = load_data_for_h5(input_data_path, section_id)
    l.logger.info(f'[process_data] loading data section_id={section_id} from source path {input_data_path} done')

    l.logger.info(f'[process_data] enhence_data begin...')
    enhanced_adata, cell_spatial = enhence.enhence_data(AnnData)
    dump_by_pkl(enhanced_adata, enhanced_adata_dump)
    dump_by_pkl(cell_spatial, cell_spatial_dump)
    l.logger.info(f'[process_data] enhence_data done')
    dump_by_pkl(enhanced_adata_dump, enhanced_adata_dump)

    # load truth
    l.logger.info(
        f'[process_data] loading  ground_truth section_id={section_id} from source path {input_data_path} begin...')
    ground_truth = load_data_for_ground_truth(input_data_path, section_id, AnnData)
    l.logger.info(
        f'[process_data] loading  ground_truth section_id={section_id} from source path {input_data_path} done')
    low_dim_x = enhanced_adata.obsm['X_pca']
    dump_by_pkl(low_dim_x, low_dim_x_dump)

    l.logger.info(f'[process_data] done ')
    return low_dim_x, enhanced_adata, cell_spatial, ground_truth


# main func
def main():
    # 解析命令行参数
    args = parse_args()

    # 读取参数
    input_data_path = args.input_data_path
    section_id = args.section_id
    output_data_path = args.output_data_path
    is_reload = args.is_reload
    log_path = os.path.join(output_data_path, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path, mode=0o644, exist_ok=True)

    # 初始化日志
    l.initlog(log_path)

    # 打印启动日志
    l.logger.info(f'main args={args}')

    # 准备数据
    low_dim_x, enhanced_adata, cell_spatial, ground_truth = process_data(input_data_path, section_id, output_data_path,
                                                                         is_reload)

    # call
    call(low_dim_x, cell_spatial, ground_truth, output_data_path, is_reload)


if __name__ == '__main__':
    main()
