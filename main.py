
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
datapath = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\GE_Datasets_2\\'
reports_path = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\RL_developmental_studies\\Reports\\tight_var_data\\'
primitive_ccre_ds_path = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\RL_developmental_studies\\Reports\\cCRE Clustering\\variable_k\\agglomerative_clust_cCRE_8.csv'
'''

LOG_DIR = 'local_runs/'


##COPY TO NOTEBOOK FROM HERE!!!###

from adagae import *
import torch
import pickle

from torch.utils.tensorboard import SummaryWriter


###########
## HYPER-PARAMS
###########


plt.rcParams["figure.figsize"] = (10, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

genes_to_pick = 0
learning_rate = 5 * 10 ** -3
init_sparsity = 100
genomic_C = 3e5
genomic_slope = 0.4

link_ds = pd.read_csv(datapath + '/Link_Matrix.tsv', sep='\t')
link_ds.columns = ['EnsembleID', 'cCRE_ID', 'Distance']
link_ds['EnsembleID'] = link_ds['EnsembleID'].apply(lambda x: x.strip())
link_ds['cCRE_ID'] = link_ds['cCRE_ID'].apply(lambda x: x.strip())


X, G, ge_count, ccre_count, distance_matrices, slopes, gen_dist_score, ccre_ds, ge_class_labels, ccre_class_labels, gene_ds= \
    data_preprocessing(link_ds, genes_to_pick,
                       device=device, genomic_C = genomic_C, genomic_slope = genomic_slope, chr_to_filter=[12,13,14,15,16,17,18])



