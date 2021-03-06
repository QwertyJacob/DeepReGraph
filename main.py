

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

from DeepReGraph import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



adaGAE_object = initialize_DeepReGraph('my_new_model',
                       link_ds=link_ds,
                        device=device,
                       datapath=datapath,
                       chr_to_filter=[12,13,14,15,16,17,18])


manual_run(adaGAE_object, L=1, init_sparsity=151, sparsity_increment=0,
           init_omega_BP=1, final_omega_BP=1,
           init_alpha_G=.1, final_alpha_G=.1,
           init_alpha_ATAC=.1, final_alpha_ATAC=.1,
           init_alpha_ACET=.1, final_alpha_ACET=.1,
           init_alpha_METH=.1, final_alpha_METH=.1,
           init_alpha_Z=0, final_alpha_Z=0,
           init_attractive_loss_weight=.9, final_attractive_loss_weight=.9,
           init_repulsive_loss_weight=.1, final_repulsive_loss_weight=.1,
           max_iter=150,
           omega_ATAC=1,
           omega_ACET=0,
           omega_METH=0)
