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
genes_to_pick = 20

learning_rate = 5 * 10 ** -3


if __name__ == '__main__':


    modelname = '/champion'
    tensorboard = SummaryWriter(LOG_DIR + modelname)

    X, ge_count, ccre_count, links, kendall_matrix, ge_class_labels = data_preprocessing(datapath, reports_path, genes_to_pick, device)

    gae = AdaGAE(X,
                 ge_count,
                 ccre_count,
                 links,
                 kendall_matrix,
                 ge_class_labels,
                 tensorboard,
                 device=device,
                 datapath = datapath,
                 learning_rate=learning_rate)

    #manual_run(gae)
    #fixed_spars_run(gae)

