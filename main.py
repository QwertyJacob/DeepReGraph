
'''

# colab Notebook
ccre_primitive_clusters_path = '/content/DIAGdrive/MyDrive/RL_developmental_studies/Reports/cCRE Clustering/'
'''

'''
# Elis computer
datapath = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\GE_Datasets_2\\'
reports_path = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\RL_developmental_studies\\Reports\\tight_var_data\\'
'''

# Personal computer
datapath = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\GE_Datasets_2\\'
reports_path = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\RL_developmental_studies\\Reports\\tight_var_data\\'
primitive_ccre_ds_path = 'C:\\Users\\Jesus\\odrive\\Diag GDrive\\RL_developmental_studies\\Reports\\cCRE Clustering\\variable_k\\agglomerative_clust_cCRE_8.csv'
LOG_DIR = 'local_runs/'


##COPY TO NOTEBOOK FROM HERE!!!###

from adagae import *
import torch
from torch.utils.tensorboard import SummaryWriter


###########
## HYPER-PARAMS
###########


plt.rcParams["figure.figsize"] = (10, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
genes_to_pick = 0

learning_rate = 5 * 10 ** -3
wk_atac = 0.0
wk_acet = 0.0
wk_meth = 0.0
init_sparsity = 100

if __name__ == '__main__':



    X, G, ge_count, ccre_count, distance_matrices, gen_dist_score, ccre_ds, kendall_matrix, ge_class_labels, ccre_class_labels= \
        data_preprocessing(datapath, reports_path, primitive_ccre_ds_path, genes_to_pick,
                           wk_atac=wk_atac, wk_acet=wk_acet, wk_meth=wk_meth, device=device, chr_to_filter=[15,16,17,18])



    modelname = '/chromosome_15-18'
    tensorboard = SummaryWriter(LOG_DIR + modelname)


    gae = AdaGAE(X, G, ge_count, ccre_count, distance_matrices,
                 gen_dist_score, kendall_matrix, init_sparsity, ge_class_labels, ccre_class_labels,
                 tensorboard, device=device, datapath = datapath)


    # mode 1:
    # STEP 1:

    manual_run(gae, max_epoch=10, init_sparsity=100, sparsity_increment=10,
               init_alpha_D=1, final_alpha_D=1,
               init_alpha_G=0, final_alpha_G=0,
               init_alpha_ATAC=0, final_alpha_ATAC=0,
               init_alpha_ACET=0, final_alpha_ACET=0,
               init_alpha_METH=0, final_alpha_METH=0,
               init_alpha_Z=0, final_alpha_Z=0.5,
               init_attractive_loss_weight=0, final_attractive_loss_weight=500,
               init_repulsive_loss_weight=100, final_repulsive_loss_weight=0,
               max_iter=15)

    # gae.differential_sparsity = True
    # STEP 2:

    manual_run(gae, max_epoch=10, init_sparsity=100, sparsity_increment=20,
               init_alpha_D=0, final_alpha_D=0,
               init_alpha_G=0, final_alpha_G=0,
               init_alpha_ATAC=0, final_alpha_ATAC=0,
               init_alpha_ACET=0, final_alpha_ACET=0,
               init_alpha_METH=0, final_alpha_METH=0,
               init_alpha_Z=1, final_alpha_Z=1,
               init_attractive_loss_weight=0, final_attractive_loss_weight=100,
               init_repulsive_loss_weight=0, final_repulsive_loss_weight=0,
               max_iter=15)


    # mode 2
    # gae.differential_sparsity = True
    # manual_run(gae, max_epoch=10, init_sparsity=100, sparsity_increment=20,
    #            init_alpha_D=0, final_alpha_D=0,
    #            init_alpha_G=1, final_alpha_G=0,
    #            init_alpha_ATAC=0.5, final_alpha_ATAC=0,
    #            init_alpha_ACET=0.5, final_alpha_ACET=0,
    #            init_alpha_METH=0.5, final_alpha_METH=0,
    #            init_alpha_Z=0, final_alpha_Z=1,
    #            init_attractive_loss_weight=0, final_attractive_loss_weight=100,
    #            init_repulsive_loss_weight=100, final_repulsive_loss_weight=0,
    #            max_iter=15)
    #

    # alpha_SYMMs = [0.5, 0]
    # alpha_Ds = [0, 1]
    # alpha_value_index = 0
    # alphaD = alpha_Ds[alpha_value_index]
    # alphaG = alpha_SYMMs[alpha_value_index]
    # alphaATAC = alpha_SYMMs[alpha_value_index]
    # alphaACET = alpha_SYMMs[alpha_value_index]
    # alphaMETH = alpha_SYMMs[alpha_value_index]
    #
    # def toogle_alphas():
    #     global alphaD, alphaG, alphaATAC, alphaACET, alphaMETH
    #     global alpha_value_index
    #
    #     alpha_value_index += 1
    #
    #     alphaD = alpha_Ds[alpha_value_index%2]
    #     alphaG = alpha_SYMMs[alpha_value_index%2]
    #     alphaACET = alpha_SYMMs[alpha_value_index%2]
    #     alphaATAC = alpha_SYMMs[alpha_value_index%2]
    #     alphaMETH = alpha_SYMMs[alpha_value_index%2]
    #
    #
    #
    # curr_spars = 200
    # max_epochs = 10
    # iters_per_epoch = 45
    # init_alphaZ = 0.3
    # final_alphaZ = 1
    # init_attr = 1000
    # final_attr = 1000
    # init_rep = 100
    # final_rep = 100
    # T = max_epochs * iters_per_epoch
    # for epochs in range(0,max_epochs):
    #
    #     toogle_alphas()
    #     alphaZ = 1
    #     rep = gae.get_dinamic_param(init_rep, final_rep, T)
    #     attr_force = gae.get_dinamic_param(init_attr, final_attr, T)
    #     gae.run_1_epoch(current_sparsity=curr_spars, alpha_D=alphaD, attractive_loss_weight=attr_force,
    #                     repulsive_loss_weight=rep, alpha_G=alphaG, alpha_ATAC=alphaATAC, alpha_ACET=alphaACET,
    #                     alpha_METH=alphaMETH, alpha_Z=alphaZ, max_iter=iters_per_epoch)
    #     gae.plot_classes()
    #
    #     tensorboard.close()
    #     #fixed_spars_run(gae)

