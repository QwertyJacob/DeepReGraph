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


    modelname = '/new_champion'
    tensorboard = SummaryWriter(LOG_DIR + modelname)

    X, ge_count, ccre_count, distance_matrices, links, kendall_matrix, ge_class_labels = data_preprocessing(datapath, reports_path, genes_to_pick, device)

    gae = AdaGAE(X,
                 ge_count,
                 ccre_count,
                 distance_matrices,
                 links,
                 kendall_matrix,
                 ge_class_labels,
                 tensorboard,
                 device=device,
                 datapath = datapath)

    manual_run(gae,
               max_epoch=10,
               init_sparsity=100,
               sparsity_increment=1,
               init_alpha_D=0.5,
               final_alpha_D=0.5,
               init_attractive_loss_weight=0.1,
               final_attractive_loss_weight=3,
               init_repulsive_loss_weight=1,
               final_repulsive_loss_weight=0.1,
               max_iter=70
               )
    #fixed_spars_run(gae)

