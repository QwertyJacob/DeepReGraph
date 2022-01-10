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
import matplotlib.pyplot as plt



###########
## HYPER-PARAMS
###########


plt.rcParams["figure.figsize"] = (10, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
genes_to_pick = 100

learning_rate = 5 * 10 ** -3


if __name__ == '__main__':


    modelname = '/2'
    tensorboard = SummaryWriter(LOG_DIR + modelname)

    wk_atac = 0.05
    wk_acet = 0.05
    wk_meth = 0.05
    init_sparsity = 10

    X, ge_count, ccre_count, distance_matrices, links, kendall_matrix, ge_class_labels = \
        data_preprocessing(datapath, reports_path, genes_to_pick,
                           wk_atac=wk_atac, wk_acet=wk_acet, wk_meth=wk_meth, device=device)

    gae = AdaGAE(X,ge_count,ccre_count,distance_matrices,
                 links,kendall_matrix,init_sparsity,ge_class_labels,
                 tensorboard,device=device,datapath = datapath)

    manual_run(gae,
               max_epoch=10,
               init_sparsity=init_sparsity,
               sparsity_increment=30,
               init_gbf=0,
               final_gbf=0.5,
               init_RQ_loss_weight=0,
               final_RQ_loss_weight=0,
               init_attractive_loss_weight=0.1,
               final_attractive_loss_weight=3,
               init_repulsive_loss_weight=1,
               final_repulsive_loss_weight=0.1,
               init_lambda_attractive=0.5,
               final_lambda_attractive=0.5,
               init_lambda_repulsive=0.5,
               final_lambda_repulsive=0.5,
               init_agg_repulsive=0,
               final_agg_repulsive=0,
               max_iter=70
               )
    #fixed_spars_run(gae)

