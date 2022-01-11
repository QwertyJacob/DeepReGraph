'''
# Elis computer
datapath = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\GE_Datasets_2\\'
reports_path = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\RL_developmental_studies\\Reports\\tight_var_data\\'
'''

# Personal computer
datapath = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\GE_Datasets_2\\'
reports_path = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\Shared with Me\\RL_developmental_studies\\Reports\\tight_var_data\\'
LOG_DIR = 'local_runs/'


##COPY TO NOTEBOOK FROM HERE!!!###

from adagae import *
import torch
from torch.utils.tensorboard import SummaryWriter
from data_reporting import *


###########
## HYPER-PARAMS
###########


plt.rcParams["figure.figsize"] = (10, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
genes_to_pick = 100

learning_rate = 5 * 10 ** -3


if __name__ == '__main__':


    modelname = '/1'
    tensorboard = SummaryWriter(LOG_DIR + modelname)

    wk_atac = 0.05
    wk_acet = 0.05
    wk_meth = 0.05
    init_sparsity = 100

    X, ge_count, ccre_count, distance_matrices, links, ccre_ds, kendall_matrix, ge_class_labels = \
        data_preprocessing(datapath, reports_path, genes_to_pick,
                           wk_atac=wk_atac, wk_acet=wk_acet, wk_meth=wk_meth, device=device)

    gae = AdaGAE(X,ge_count,ccre_count,distance_matrices,
                 links,kendall_matrix,init_sparsity,ge_class_labels,
                 tensorboard,device=device,datapath = datapath)

    # STEP 1, VERSION 1:

    manual_run(gae, max_epoch=10, init_sparsity=100, sparsity_increment=10,
               init_alpha_D=0, final_alpha_D=0,
               init_alpha_G=1, final_alpha_G=.1,
               init_alpha_ATAC=1, final_alpha_ATAC=.1,
               init_alpha_ACET=1, final_alpha_ACET=.1,
               init_alpha_METH=1, final_alpha_METH=.1,
               init_alpha_Z=0, final_alpha_Z=1,
               init_attractive_loss_weight=0.1, final_attractive_loss_weight=500,
               init_repulsive_loss_weight=1000, final_repulsive_loss_weight=0,
               max_iter=15)

    # STEP 1, VERSION 2:

    manual_run(gae, max_epoch=10, init_sparsity=100, sparsity_increment=10,
               init_alpha_D=1, final_alpha_D=1,
               init_alpha_G=0, final_alpha_G=0,
               init_alpha_ATAC=0, final_alpha_ATAC=0,
               init_alpha_ACET=0, final_alpha_ACET=0,
               init_alpha_METH=0, final_alpha_METH=0,
               init_alpha_Z=0, final_alpha_Z=1,
               init_attractive_loss_weight=0.1, final_attractive_loss_weight=1000,
               init_repulsive_loss_weight=1000, final_repulsive_loss_weight=0,
               max_iter=15)

    tensorboard.close()
    #fixed_spars_run(gae)

