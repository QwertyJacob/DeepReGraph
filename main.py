'''
The following code is a modified version of the code available at https://github.com/hyzhang98/AdaGAE
which is an implementation of the paper "Adaptive Graph Auto-Encoder for General Data Clustering"
available at https://ieeexplore.ieee.org/document/9606581
Modifications were made by Jesus Cevallos to adapt to the application problem.
'''
#################
# For Colab

# from google.colab import drive
# drive.mount('/content/DIAGdrive')
# !pip install umap-learn[plot]

# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# !unzip ngrok-stable-linux-amd64.zip

# import os
# LOG_DIR = '/content/DIAGdrive/MyDrive/GE_Datasets/official_logs/'
# os.makedirs(LOG_DIR, exist_ok=True)
# get_ipython().system_raw(
#    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
#    .format(LOG_DIR)
# )

# get_ipython().system_raw('./ngrok http 6006 &')

# ! curl -s http://localhost:4040/api/tunnels | python3 -c \
#    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"


# from tqdm.notebook import tqdm
# datapath = "/content/DIAGdrive/MyDrive/GE_Datasets/"

# reports_path= '/content/DIAGdrive/MyDrive/RL_developmental_studies/Reports/'
#################

from tqdm import tqdm as tqdm

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


## Fixed Hyper params


eval = False
pre_trained = False
gcn = False
clusterize = True
use_kendall_matrix = True

learning_rate = 5 * 10 ** -3
init_genomic_C = 3e5
init_genomic_slope = 0.4
rep_agg_loss_weight = 1

if __name__ == '__main__':

    current_sparsity = init_sparsity = 10

    init_gbf = 0.5
    final_gbf = 0.1

    init_attractive_loss_weight = 0.1
    final_attractive_loss_weight = 2

    init_repulsive_loss_weight = 2
    final_repulsive_loss_weight = 0.1

    init_RQ_loss_weight = .1
    final_RQ_loss_weight = 1.5

    init_agg_repulsive = 0
    final_agg_repulsive = 1

    init_lambda_attractive = final_lambda_attractive = 0.6
    init_lambda_repulsive = final_lambda_repulsive = 0.5

    modelname = '/champion'
    tensorboard = SummaryWriter(LOG_DIR + modelname)

    sparsity_increment = 10

    max_iter = 15
    max_epoch = 40

    epoch = 0
    gae = AdaGAE(tensorboard,
                 device=device,
                 pre_trained=False,
                 genes_to_pick=genes_to_pick,
                 datapath = datapath,
                 reports_path=reports_path)

    #manual_run()
    fixed_spars_run(gae)

