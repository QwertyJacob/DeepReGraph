import pandas as pd
import gdown
import os

def run_tensorboard_google_colab(log_dir='tensorboard_logs/'):
    '''Tensorboard is an interactive dashboard that helps visualizing results for various runs of a ML model:
        The following code will activate it on this Google Colab environment
    '''


    LOG_DIR = log_dir
    os.makedirs(LOG_DIR, exist_ok=True)
    get_ipython().system_raw(
        'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
            .format(LOG_DIR))



def download_data_google_colab(dest_path='/content/DeepReGraph/'):
    print('Downloading Datasets..')

    print('Downloading cCRE activity datasets...')
    gdown.download('11Bn5R5XcF0xGw31ghNPWrmTo65hgnwMx', dest_path+'cCRE_variational_mean_reduced.csv', quiet=True)

    print('Downloading Gene Expression datasets...')
    gdown.download('1Az2Z1JCjUVZt1J6e9m05BM24BTO3gUDr', dest_path+'tight_var_log_fpkm_GE_ds', quiet=True)

    print('Downloading BasePair Distance information...')
    gdown.download('1-k5WkFFPidRasxKIC6CrBa0LyRAT0WZb', dest_path+'Link_Matrix_piece_0.csv', quiet=True)
    gdown.download('1-p0jwhaoiFHw5XfWTplgpee3QFv4myph', dest_path+'Link_Matrix_piece_1.csv', quiet=True)
    gdown.download('1-xHQ9z2FI27Jx2et_WMqJLUFNpygWv83', dest_path+'Link_Matrix_piece_2.csv', quiet=True)
    gdown.download('1-zVjEDSD18qkyOD54oxZaDbpzGxT9GEV', dest_path+'Link_Matrix_piece_3.csv', quiet=True)
    gdown.download('101oEGWsHzDmAewWxOHjlzlsek_Z52jXf', dest_path+'Link_Matrix_piece_4.csv', quiet=True)

    # We assemble it together:
    link_ds_part_0 = pd.read_csv(dest_path + 'Link_Matrix_piece_0.csv', index_col=0)
    link_ds_part_1 = pd.read_csv(dest_path + 'Link_Matrix_piece_1.csv', index_col=0)
    link_ds_part_2 = pd.read_csv(dest_path + 'Link_Matrix_piece_2.csv', index_col=0)
    link_ds_part_3 = pd.read_csv(dest_path + 'Link_Matrix_piece_3.csv', index_col=0)
    link_ds_part_4 = pd.read_csv(dest_path + 'Link_Matrix_piece_4.csv', index_col=0)
    link_ds = pd.concat([link_ds_part_0, link_ds_part_1, link_ds_part_2, link_ds_part_3, link_ds_part_4])

    print('Downloading Supplementary datasets...')
    # Downloading kmeans clustering of genes'
    gdown.download('1FVZ3_eqDBQCxYK8bR0vzQCrAQzmJNvcJ', dest_path+'kmeans_clustered_genes_4.csv', quiet=True)
    # Downloading kmeans clustering of genes'
    gdown.download('14JzI8L9ThVP28UCO5FYfmYeNC2w_e70W', dest_path+'agglomerative_clust_cCRE_8.csv', quiet=True)

    return link_ds