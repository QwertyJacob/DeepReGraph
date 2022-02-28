
'''

# colab Notebook
datapath = "/content/drive/MyDrive/GE_Datasets/"
reports_path= '/content/drive/MyDrive/RL_developmental_studies/Reports/tight_var_data/'
results_path=  "/content/drive/MyDrive/GE_Datasets/results/"
primitive_ccre_ds_path = '/content/drive/MyDrive/RL_developmental_studies/Reports/cCRE Clustering/variable_k/agglomerative_clust_cCRE_8.csv'
'''

# Elis computer
datapath = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\GE_Datasets_PUBLIC_FOR_PAPER\\'
'''
# Personal computer
datapath = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\GE_Datasets_PUBLIC_FOR_PAPER\\'
'''
LOG_DIR = 'local_runs/'

import pandas as pd

link_ds = pd.read_csv(datapath + '/Link_Matrix.tsv', sep='\t')
link_ds.columns = ['EnsembleID', 'cCRE_ID', 'Distance']
link_ds['EnsembleID'] = link_ds['EnsembleID'].apply(lambda x: x.strip())
link_ds['cCRE_ID'] = link_ds['cCRE_ID'].apply(lambda x: x.strip())


##COPY TO NOTEBOOK FROM HERE!!!###

from adaGAE import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



adaGAE_object = initiliaze_DeepReGraph('my_new_model',
                       device=device,
                       link_ds=link_ds,
                       datapath=datapath,
                       chr_to_filter=[12,13,14,15,16,17,18])


manual_run(adaGAE_object, max_epoch=1, init_sparsity=151, sparsity_increment=0,
           init_alpha_D=1, final_alpha_D=1,
           init_alpha_G=.1, final_alpha_G=.1,
           init_alpha_ATAC=.1, final_alpha_ATAC=.1,
           init_alpha_ACET=.1, final_alpha_ACET=.1,
           init_alpha_METH=.1, final_alpha_METH=.1,
           init_alpha_Z=0, final_alpha_Z=0,
           init_attractive_loss_weight=.9, final_attractive_loss_weight=.9,
           init_repulsive_loss_weight=.1, final_repulsive_loss_weight=.1,
           max_iter=150,
           init_wk_ATAC=1,
           final_wk_ATAC=1,
           init_wk_ACET=0,
           final_wk_ACET=0,
           init_wk_METH=0,
           final_wk_METH=0)
