'''
The following code is a modified version of the code available at https://github.com/hyzhang98/AdaGAE
which is an implementation of the paper "Adaptive Graph Auto-Encoder for General Data Clustering"
available at https://ieeexplore.ieee.org/document/9606581
Modifications were made by Jesus Cevallos to adapt to the application problem.
'''



import torch
import cProfile
import pstats
from functools import wraps
import pandas as pd
import numpy as np
import io
import math
import PIL.Image
from torchvision.transforms import ToTensor
# import hdbscan
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.utils import _safe_indexing
from sklearn import linear_model
#import umap
#import umap.plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#######################
# HELPER FUNCTIONS#####
#######################

def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner


def get_hybrid_feature_matrix(link_ds, ccre_ds):
    ge_values = link_ds.reset_index().drop_duplicates('EnsembleID')[['Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5',
                                                                     'Heart_E13_5', 'Heart_E14_5', 'Heart_E15_5',
                                                                     'Heart_E16_5', 'Heart_P0']].values
    ge_count = ge_values.shape[0]
    ge_values_new = np.zeros((ge_values.shape[0], 32))
    ge_values_new[:, 0:8] = ge_values

    ccre_activity = ccre_ds.set_index('cCRE_ID').values
    ccre_count = ccre_activity.shape[0]
    ccre_activity_new = np.zeros((ccre_activity.shape[0], 32))
    ccre_activity_new[:, 8:32] = ccre_activity
    return torch.Tensor(np.concatenate((ge_values_new, ccre_activity_new))).cpu(), ge_count, ccre_count


def get_distance_matrices(X, ge_count):
    print('Computing euclidean distance matrices')

    cpu_X = X.cpu()

    gene_exp = cpu_X[:ge_count, 0:8]
    met = cpu_X[ge_count:, 8:16]
    acet = cpu_X[ge_count:, 16:24]
    atac = cpu_X[ge_count:, 24:32]

    ge_distances = distance(gene_exp.t(), gene_exp.t())
    ge_distances = torch.max(ge_distances, torch.t(ge_distances))
    #abs scaling
    ge_distances /= ge_distances.max()

    met_distances = distance(met.t(), met.t())
    met_distances = torch.max(met_distances, torch.t(met_distances))
    met_distances /= met_distances.max()

    acet_distances = distance(acet.t(), acet.t())
    acet_distances = torch.max(acet_distances, torch.t(acet_distances))
    acet_distances /= acet_distances.max()

    atac_distances = distance(atac.t(), atac.t())
    atac_distances = torch.max(atac_distances, torch.t(atac_distances))
    atac_distances /= atac_distances.max()

    print('Euclidean distance matrices computed')

    return ge_distances, atac_distances, acet_distances, met_distances


def get_kendall_matrix(X, ge_count, ccre_count, wk_atac=0.05, wk_acet=0.05, wk_meth=0.05):

    numpy_X = X.cpu().numpy()

    time_steps = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    gene_exp = numpy_X[:ge_count, 0:8]
    met = numpy_X[ge_count:, 8:16]
    acet = numpy_X[ge_count:, 16:24]
    atac = numpy_X[ge_count:, 24:32]

    print('computing kendall matrix...')

    gene_exp_slopes = []
    reg = linear_model.LinearRegression()
    for gene_exp_row in gene_exp:
        reg.fit(time_steps.reshape(-1, 1), gene_exp_row)
        Y_pred = reg.predict(time_steps.reshape(-1, 1))
        gene_exp_slopes.append(reg.coef_[0])
        # plt.scatter(time_steps.reshape(-1,1),gene_exp_row)
        # plt.plot(time_steps.reshape(-1,1), Y_pred, color='red')
        # plt.show()

    met_slopes = []
    reg = linear_model.LinearRegression()
    for met_row in met:
        reg.fit(time_steps.reshape(-1, 1), met_row)
        Y_pred = reg.predict(time_steps.reshape(-1, 1))
        met_slopes.append(reg.coef_[0])
        # plt.scatter(time_steps.reshape(-1,1),met_row)
        # plt.plot(time_steps.reshape(-1,1), Y_pred, color='red')
        # plt.show()

    acet_slopes = []
    reg = linear_model.LinearRegression()
    for acet_row in acet:
        reg.fit(time_steps.reshape(-1, 1), acet_row)
        Y_pred = reg.predict(time_steps.reshape(-1, 1))
        acet_slopes.append(reg.coef_[0])
        # plt.scatter(time_steps.reshape(-1,1),acet_row)
        # plt.plot(time_steps.reshape(-1,1), Y_pred, color='red')
        # plt.show()

    atac_slopes = []
    reg = linear_model.LinearRegression()
    for atac_row in atac:
        reg.fit(time_steps.reshape(-1, 1), atac_row)
        Y_pred = reg.predict(time_steps.reshape(-1, 1))
        atac_slopes.append(reg.coef_[0])
        # plt.scatter(time_steps.reshape(-1,1),atac_row)
        # plt.plot(time_steps.reshape(-1,1), Y_pred, color='red')
        # plt.show()

    gene_exp_slopes = np.array(gene_exp_slopes)
    gene_exp_slopes = np.sign(gene_exp_slopes)
    # gene_exp_slopes = (gene_exp_slopes - gene_exp_slopes.min()) / (gene_exp_slopes.max()- gene_exp_slopes.min())

    atac_slopes = np.array(atac_slopes)
    atac_slopes = np.sign(atac_slopes)
    # atac_slopes = (atac_slopes - atac_slopes.min()) / (atac_slopes.max()- atac_slopes.min())

    met_slopes = np.array(met_slopes)
    met_slopes = np.sign(met_slopes)
    # met_slopes = (met_slopes - met_slopes.min()) / (met_slopes.max()- met_slopes.min())

    acet_slopes = np.array(acet_slopes)
    acet_slopes = np.sign(acet_slopes)
    # acet_slopes = (acet_slopes - acet_slopes.min()) / (acet_slopes.max()- acet_slopes.min())

    dim = ge_count + ccre_count
    kendall_matrix = torch.zeros(dim, dim)


    ccre_slopes = ((wk_atac * atac_slopes) + (wk_acet * acet_slopes) - (wk_meth * met_slopes))

    ccre_trend_upright_submatrix = np.repeat(ccre_slopes.reshape(-1, 1), ge_count).reshape(ccre_count, -1).transpose()
    gene_trend_upright_submatrix = np.repeat(gene_exp_slopes.reshape(1, -1), ccre_count).reshape(ge_count, -1)
    kendall_matrix[:ge_count, ge_count:] = torch.Tensor(gene_trend_upright_submatrix + ccre_trend_upright_submatrix)

    gene_trend_downleft_submatrix = np.repeat(gene_exp_slopes.reshape(-1, 1), ccre_count).reshape(-1,ccre_count).transpose()
    ccre_trend_downleft_submatrix = np.repeat(ccre_slopes.reshape(1, -1), ge_count).reshape(ccre_count, -1)
    kendall_matrix[ge_count:, :ge_count] = torch.Tensor(gene_trend_downleft_submatrix + ccre_trend_downleft_submatrix)
    kendall_matrix.abs_()
    #row-wise scaling
    kendall_matrix /= (kendall_matrix.max(dim=1)[0] + 1e-10).reshape(-1,1)

    print('kendall matrix computed...')

    return kendall_matrix


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    # result = result.max(torch.zeros(result.shape).cuda())
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    # result = torch.max(result, result.t())
    return result


def get_normalized_adjacency_matrix(weights):
    # We don't create self loops with 1 (nor with any value)
    # because we want the embeddings to adaptively learn
    # the self-loop weights.
    # W = torch.eye(weights.shape[0]).cuda() + weights
    # degree = torch.sum(W, dim=1).pow(-0.5)
    # return (W * degree).t()*degree
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree


def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)


def fast_genomic_distance_to_similarity(link_matrix, c, d):
    '''
    see https://www.desmos.com/calculator/bmxxh8sqra
    TODO play aroun'
    '''
    return 1 / (((link_matrix / c) ** (10 * d)) + 1)


def get_genomic_distance_matrix(link_ds, add_self_loops_genomic):
    genes = link_ds.index.unique().tolist()
    ccres = link_ds.cCRE_ID.unique().tolist()
    entity_number = len(genes + ccres)

    entities_df = pd.DataFrame(genes + ccres, columns=['EntityID'])
    entities_df['entity_index'] = range(0, entity_number)
    entities_df.set_index('EntityID', inplace=True)

    dense_A = np.zeros((entity_number, entity_number))
    dense_A.fill(np.inf)
    if add_self_loops_genomic:
        np.fill_diagonal(dense_A, 0)
    print('processing genomic distances...')

    for index, row in link_ds.reset_index().iterrows():
        gene_idx = entities_df.loc[row.EnsembleID][0]
        ccre_idx = entities_df.loc[row.cCRE_ID][0]
        dense_A[gene_idx, ccre_idx] = row.Distance
        dense_A[ccre_idx, gene_idx] = row.Distance

    return dense_A


def load_data(datapath, num_of_genes=0, tight=True, chr_to_filter=None):

    if tight:
        var_log_ge_ds = pd.read_csv(datapath + 'tight_var_log_fpkm_GE_ds')
    else:
        var_log_ge_ds = pd.read_csv(datapath + 'var_log_fpkm_GE_ds')

    X = var_log_ge_ds[['Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5',
                       'Heart_E13_5', 'Heart_E14_5', 'Heart_E15_5', 'Heart_E16_5', 'Heart_P0']].values
    '''
    Peyman's mean substraction: (For each row)
    '''
    X = np.stack([data_row - data_row.mean() for data_row in X])

    working_genes_ds = var_log_ge_ds.reset_index()[['EnsembleID', 'Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5',
                                                    'Heart_E13_5', 'Heart_E14_5', 'Heart_E15_5', 'Heart_E16_5',
                                                    'Heart_P0']].copy()

    working_genes_ds[['Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5',
                      'Heart_E13_5', 'Heart_E14_5', 'Heart_E15_5', 'Heart_E16_5', 'Heart_P0']] = X

    # if tight:
    #   ccre_ds = pd.read_csv(datapath + 'tight_cCRE_variational_mean_reduced.csv')
    # else:
    #   ccre_ds = pd.read_csv(datapath + 'cCRE_variational_mean_reduced.csv')

    ccre_ds = pd.read_csv(datapath + 'cCRE_variational_mean_reduced.csv')

    if chr_to_filter != None:
        filtered_ccre_ds = pd.DataFrame()
        for chr_number in chr_to_filter:
            filtered_ccre_ds = pd.concat(
                [filtered_ccre_ds, ccre_ds[ccre_ds['cCRE_ID'].str.startswith('chr' + str(chr_number))]])

        ccre_ds = filtered_ccre_ds
    ####################LINK MATRIX #####################################################
    link_ds = pd.read_csv(datapath + '/Link_Matrix.tsv', sep='\t')
    link_ds.columns = ['EnsembleID', 'cCRE_ID', 'Distance']
    link_ds['EnsembleID'] = link_ds['EnsembleID'].apply(lambda x: x.strip())
    link_ds['cCRE_ID'] = link_ds['cCRE_ID'].apply(lambda x: x.strip())

    if num_of_genes == 0:
        var_ge_list = working_genes_ds['EnsembleID'].tolist()
    else:
        var_ge_list = working_genes_ds['EnsembleID'].tolist()[:num_of_genes]

    link_ds = link_ds[link_ds['EnsembleID'].isin(var_ge_list)]
    link_ds = link_ds[link_ds['cCRE_ID'].isin(ccre_ds['cCRE_ID'].tolist())]
    link_ds = link_ds.set_index('EnsembleID').join(
        working_genes_ds[['EnsembleID', 'Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5', 'Heart_E13_5',
                          'Heart_E14_5', 'Heart_E15_5', 'Heart_E16_5', 'Heart_P0']].set_index('EnsembleID'))

    ccres = link_ds.cCRE_ID.unique().tolist()
    ccre_ds = ccre_ds[ccre_ds['cCRE_ID'].isin(ccres)]

    return link_ds, ccre_ds


def get_primitive_gene_clusters(reports_path,link_ds):
    gene_primitive_clusters_path = reports_path
    kmeans_ds = pd.read_csv(gene_primitive_clusters_path + 'kmeans_clustered_genes_4.csv')
    primitive_ge_clustered_ds = kmeans_ds.set_index('EnsembleID').drop('Unnamed: 0', axis=1)
    primitive_ge_clustered_ds.columns = ['primitive_cluster']
    gene_ds = link_ds.reset_index().drop_duplicates('EnsembleID').set_index('EnsembleID')
    prim_gene_ds = gene_ds.join(primitive_ge_clustered_ds)['primitive_cluster'].reset_index()

    return np.array(prim_gene_ds.primitive_cluster.to_list())


def data_preprocessing(datapath, reports_path, genes_to_pick, device,
                       wk_atac=0.05, wk_acet=0.05, wk_meth=0.05, add_self_loops_genomic=False):
    ## Data preprocessing:

    link_ds, ccre_ds = load_data(datapath, genes_to_pick)

    X, ge_count, ccre_count = get_hybrid_feature_matrix(link_ds, ccre_ds)

    links = get_genomic_distance_matrix(link_ds, add_self_loops_genomic)

    ge_class_labels = ['genes_' + str(ge_cluster_label) for ge_cluster_label in
                       get_primitive_gene_clusters(reports_path, link_ds)]

    print('Analyzing ', ge_count, ' genes and ', ccre_count, ' ccres for a total of ', ge_count + ccre_count,
          ' elements.')
    print('cCREs over Gene ratio is ', ccre_count / ge_count)
    kendall_matrix = get_kendall_matrix(X, ge_count, ccre_count, wk_atac=wk_atac, wk_acet=wk_acet, wk_meth=wk_meth)

    D_G, D_ATAC, D_ACET, D_MET = get_distance_matrices(X, ge_count)

    distance_matrices = [D_G, D_ATAC, D_ACET, D_MET]

    X /= torch.max(X)
    X = torch.Tensor(X).to(device)

    return X, ge_count, ccre_count, distance_matrices, links, ccre_ds, kendall_matrix, ge_class_labels


#####
# ADAGAE OBJECT
########

SPARSITY_LABEL: str = 'Sparsity'
GENE_SPARSITY_LABEL: str = 'Gene_Sparsity'
SLOPE_LABEL: str = 'GeneticSlope'
REPULSIVE_CE_TERM: str = 'Repulsive_CE_loss'
ATTRACTIVE_CE_TERM: str = 'Attractive_CE_loss'
RQ_QUOTIENT_LOSS: str = 'RQ Quotient Loss'
RP_AGGRESSIVE_LOSS: str = 'Rep Aggressive Loss'
RP_AGGRESSIVE_LOSS_WEIGHT: str = 'Rep Aggressive Loss Weight'
TOTAL_LOSS_LABEL: str = 'Total_Loss'
RQ_LOSS_WEIGHT: str = 'RQ_Loss_weight'
GENETIC_BALANCE_FACTOR_LABEL: str = 'GeneticBalanceFactor'
GENOMIC_C_LABEL: str = 'GenomicC'
REPULSIVE_CE_LOSS_WEIGHT_LABEL: str = 'RepulsiveCELossWeight'
ATTRACTIVE_CE_LOSS_WEIGHT_LABEL: str = 'AttractiveCELossWeight'
GE_CC_SCORE_TAG: str = 'GeneCCScore'
CCRE_CC_SCORE_TAG: str = 'CCRECCScore'
HETEROGENEITY_SCORE_TAG: str = 'HeterogeneityScore'
EMBEDDING_DIAMETER: str = 'EmbeddingDiameter'
DISTANCE_TO_KNN_TAG: str = 'Mean Distance to knn'
REWARD_TAG: str = 'Reward'
UMAP_CLASS_PLOT_TAG: str = 'UMAPClassPlot'
UMAP_CLUSTER_PLOT_TAG: str = 'UMAPClusterPlot'
CLUSTER_NUMBER_LABEL: str = 'ClusterNumber'
GENE_CLUSTERING_COMPLETENESS_TAG: str = 'GeneClusteringCompleteness'
CCRE_CLUSTERING_COMPLETENESS_TAG: str = 'CCREClusteringCompleteness'
DISTANCE_SCORE_TAG: str = 'DistanceScore'
LAMBDA_REPULSIVE_LABEL: str = 'Lambda Repulsive'
LAMBDA_ATTRACTIVE_LABEL: str = 'Lambda Attractive'
LAMBDA_RQ_LABEL: str = 'Lambda RQ'


class AdaGAE_NN(torch.nn.Module):

    def __init__(self,
                 data_matrix,
                 device,
                 pre_trained,
                 pre_trained_state_dict,
                 pre_computed_embedding,
                 gcn,
                 layers,
                 learning_rate,
                 datapath
                 ):
        super(AdaGAE_NN, self).__init__()
        self.device = device
        self.embedding_dim = layers[-1]
        self.embedding = None
        self.mid_dim = layers[1]
        self.input_dim = layers[0]
        self.data_matrix = data_matrix.to(device)
        self.gcn = gcn
        if self.gcn:
            self.W1 = get_weight_initial([self.input_dim, self.mid_dim])
            self.W2 = get_weight_initial([self.mid_dim, self.embedding_dim])
        else:
            # basic GNN model (hamilton's book)
            self.W1_neigh = get_weight_initial([self.input_dim, self.mid_dim])
            self.W1_self = get_weight_initial([self.input_dim, self.mid_dim])
            self.W1_bias = get_weight_initial([self.data_matrix.shape[0], self.mid_dim])
            self.W2_neigh = get_weight_initial([self.mid_dim, self.embedding_dim])
            self.W2_self = get_weight_initial([self.mid_dim, self.embedding_dim])
            self.W2_bias = get_weight_initial([self.data_matrix.shape[0], self.embedding_dim])

        if pre_trained:
            self.load_state_dict(torch.load(datapath + pre_trained_state_dict, map_location=torch.device(self.device)))
            self.embedding = torch.load(datapath + pre_computed_embedding, map_location=torch.device(self.device))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, norm_adj_matrix):
        if self.gcn:
            embedding = norm_adj_matrix.mm(self.data_matrix.matmul(self.W1))
            embedding = torch.relu(embedding)
            self.embedding = norm_adj_matrix.mm(embedding.matmul(self.W2))
        else:
            # basic GNN model (hamilton's book)
            embedding_1 = norm_adj_matrix.mm(self.data_matrix.matmul(self.W1_neigh))
            embedding_1 += self.data_matrix.matmul(self.W1_self) + self.W1_bias
            embedding_1 = torch.relu(embedding_1)

            embedding = (norm_adj_matrix.matmul(embedding_1)).matmul(self.W2_neigh)
            embedding += embedding_1.matmul(self.W2_self) + self.W2_bias
            self.embedding = torch.relu(embedding)

        distances = distance(self.embedding.t(), self.embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)
        return recons_w + 10 ** -10


class AdaGAE():

    def __init__(self,
                 X,
                 ge_count,
                 ccre_count,
                 distance_matrices,
                 links,
                 kendall_matrix,
                 init_sparsity,
                 ge_class_labels,
                 tensorboard,
                 device=None,
                 pre_trained=False,
                 pre_trained_state_dict='models/combined_adagae_z12_initk150_150epochs',
                 pre_computed_embedding='models/combined_adagae_z12_initk150_150epochs_embedding',
                 global_step=0,
                 layers=None,
                 gcn=False,
                 init_genomic_slope=0.4,
                 init_genomic_C=3e5,
                 init_gbf=0.5,
                 init_RQ_loss_weight=0.1,
                 init_agg_repulsive=1,
                 init_attractive_loss_weight=0.1,
                 init_repulsive_loss_weight=1,
                 init_lambda_repulsive=0.5,
                 init_lambda_attractive=0.5,
                 clusterize=True,
                 learning_rate = 5 * 10 ** -3,
                 datapath="/content/DIAGdrive/MyDrive/GE_Datasets/",
                 init_alpha_ATAC=1,
                 init_alpha_ACET=1,
                 init_alpha_METH=1,
                 differential_sparsity=True,
                 gene_ccre_dist_factor=1):

        super(AdaGAE, self).__init__()

        self.ge_count = ge_count
        self.ccre_count = ccre_count
        self.D_G = distance_matrices[0]
        self.D_ATAC = distance_matrices[1]
        self.D_ACET = distance_matrices[2]
        self.D_METH = distance_matrices[3]
        self.init_alpha_ATAC = init_alpha_ATAC
        self.init_alpha_ACET = init_alpha_ACET
        self.init_alpha_METH = init_alpha_METH
        self.links = links
        self.kendall_matrix = kendall_matrix
        self.ge_class_labels = ge_class_labels
        self.device = device
        if self.device is None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.X = X
        self.pre_trained = pre_trained
        self.pre_trained_state_dict = pre_trained_state_dict
        self.pre_computed_embedding = pre_computed_embedding
        self.global_step = global_step
        self.global_ccres_over_genes_ratio = (self.ccre_count / self.ge_count)
        self.gene_ccre_dist_factor=gene_ccre_dist_factor
        self.differential_sparsity = differential_sparsity
        input_dim = X.shape[1]
        if layers is None:
            self.layers = [input_dim, 12, 2]
        self.init_sparsity = init_sparsity
        self.gcn = gcn
        self.init_genomic_slope =init_genomic_slope
        self.init_genomic_C = init_genomic_C
        self.init_gbf = init_gbf
        self.init_RQ_loss_weight = init_RQ_loss_weight
        self.init_agg_repulsive = init_agg_repulsive
        self.init_attractive_loss_weight = init_attractive_loss_weight
        self.init_repulsive_loss_weight = init_repulsive_loss_weight
        self.init_lambda_repulsive = init_lambda_repulsive
        self.init_lambda_attractive = init_lambda_attractive
        self.tensorboard = tensorboard
        self.clusterize = clusterize
        self.learning_rate = learning_rate
        self.datapath = datapath

        self.reset()

        classes = ['gene', 'ccre']
        self.class_label_array = np.array([classes[0]] * self.ge_count + [classes[1]] * self.ccre_count)


    def reset(self):
        self.iteration = 0
        self.gae_nn = None
        torch.cuda.empty_cache()
        self.gae_nn = AdaGAE_NN(self.X,
                                self.device,
                                self.pre_trained,
                                self.pre_trained_state_dict,
                                self.pre_computed_embedding,
                                self.gcn,
                                self.layers,
                                self.learning_rate,
                                self.datapath).to(self.device)
        self.current_prediction = None
        self.current_sparsity = self.prev_sparsity = self.init_sparsity
        self.current_gene_sparsity = math.ceil(self.current_sparsity / self.global_ccres_over_genes_ratio)
        if self.current_gene_sparsity == 0: self.current_gene_sparsity += 1
        self.current_genomic_slope = self.init_genomic_slope
        self.current_genomic_C = self.init_genomic_C
        self.current_genetic_balance_factor = self.init_gbf
        self.current_rq_loss_weight = self.init_RQ_loss_weight
        self.current_rep_agg_loss_weight = self.init_agg_repulsive
        self.current_cluster_number = math.ceil((self.ge_count + self.ccre_count) / self.current_sparsity)
        self.compute_S_D()
        self.init_adj_matrices()
        self.current_attractive_loss_weight = self.init_attractive_loss_weight
        self.current_repulsive_loss_weight = self.init_repulsive_loss_weight
        self.current_lambda_repulsive = self.init_lambda_repulsive
        self.current_lambda_attractive = self.init_lambda_attractive
        if not self.pre_trained: self.init_embedding()


    def init_adj_matrices(self):
        if not eval:
            self.tensorboard.add_scalar(SPARSITY_LABEL, self.current_sparsity, self.global_step)
            self.tensorboard.add_scalar(GENE_SPARSITY_LABEL, self.current_gene_sparsity, self.global_step)
        # adj is A tilded, it is the symmetric modification of the p distribution
        # raw_adj is the p distribution before the symetrization.
        if self.pre_trained:
            self.compute_P(self.gae_nn.embedding.cpu())
        else:
            self.compute_P(self.X.cpu(), first_time=True)

        self.norm_adj = get_normalized_adjacency_matrix(self.adj)
        self.norm_adj = self.norm_adj.to_sparse()

        self.adj = self.adj.cpu()
        self.raw_adj = self.raw_adj.cpu()
        # notice we could also put norm_adj into the cpu here...
        torch.cuda.empty_cache()


    def init_embedding(self):
        with torch.no_grad():
            # initilalizes self.gae_nn.embedding:
            self.gae_nn.to(self.device)
            _ = self.gae_nn(self.norm_adj.to(self.device))
            _ = None
            torch.cuda.empty_cache()


    def update_graph(self):
        self.compute_P(self.gae_nn.embedding.cpu())
        self.adj = self.adj.detach()
        self.raw_adj = self.raw_adj.detach()
        # threshold = 0.5
        # connections = (recons > threshold).type(torch.IntTensor).cuda()
        # weights = weights * connections
        self.norm_adj = get_normalized_adjacency_matrix(self.adj)

    def build_loss(self, recons):

        self.adj = self.adj.to(self.device)
        self.raw_adj = self.raw_adj.to(self.device)

        size = self.X.shape[0]
        loss = 0

        '''
        # notice that recons is actually the q distribution.
        # and that raw_adj is the p distribution. (before the symmetrization)
        '''

        # The following acts as an ATTRACTIVE force for the embedding learning:
        attractive_CE_term = -(self.raw_adj * torch.log(recons + 10 ** -10))

        # attractive_CE_term[:self.ge_count] *= self.current_lambda_attractive
        # attractive_CE_term[self.ge_count:] *= (1 - self.current_lambda_attractive)

        attractive_CE_term = attractive_CE_term.sum(dim=1)
        attractive_CE_term = attractive_CE_term.mean()
        self.tensorboard.add_scalar(ATTRACTIVE_CE_TERM, attractive_CE_term.item(), self.global_step)


        # If we use the Fuzzy Cross Entropy. (Binary cross entropy)
        # We could create a REPULSIVE force
        # https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668
        repulsive_CE_term = -(1 - self.raw_adj) * torch.log(1-recons)


        #repulsive_CE_term[:self.ge_count] *= self.current_lambda_repulsive
        #repulsive_CE_term[self.ge_count:] *= (1 - self.current_lambda_repulsive)

        repulsive_CE_term = repulsive_CE_term.sum(dim=1)
        repulsive_CE_term = repulsive_CE_term.mean()
        self.tensorboard.add_scalar(REPULSIVE_CE_TERM, repulsive_CE_term.item(), self.global_step)


        # The RQ loss acts as an attractive force for the embedding:
        # It strengthens element-wise similarities
        degree = self.adj.sum(dim=1)
        laplacian = torch.diag(degree) - self.adj


        # Notice that torch.trace return the sum of the elements in the diagonal of the input matrix.
        rayleigh_quotient_loss = torch.trace(
            self.gae_nn.embedding.t().matmul(laplacian).matmul(self.gae_nn.embedding)) / size
        self.tensorboard.add_scalar(RQ_QUOTIENT_LOSS, rayleigh_quotient_loss.item(), self.global_step)


        repulsive_aggressive_loss = torch.sum((1-self.raw_adj) * recons) / size
        self.tensorboard.add_scalar(RP_AGGRESSIVE_LOSS, repulsive_aggressive_loss.item(), self.global_step)

        # If we dont consider the repulsive fuzzy CE term, the following would
        # be exactly equation (11) in the AdaGAE paper.
        loss += self.current_attractive_loss_weight * attractive_CE_term
        loss += self.current_repulsive_loss_weight * repulsive_CE_term
        loss += self.current_rq_loss_weight * rayleigh_quotient_loss
        loss += self.current_rep_agg_loss_weight * repulsive_aggressive_loss
        self.tensorboard.add_scalar(TOTAL_LOSS_LABEL, loss.item(), self.global_step)

        self.adj.to('cpu')
        self.raw_adj.to('cpu')

        return loss

    def cal_clustering_metric(self):

        ge_cc_raw, ccre_cc_raw, ge_clust_completeness, ccre_clust_completeness = self.get_raw_score()
        mean_heterogeneity = self.get_mean_heterogeneity()
        distance_score = self.get_mean_distance_scores()
        return ge_cc_raw, ccre_cc_raw, mean_heterogeneity, ge_clust_completeness, ccre_clust_completeness, distance_score

    def get_mean_distance_scores(self):
        distance_score_matrix = self.S_D[:self.ge_count, self.ge_count:]
        cluster_labels = np.unique(self.current_prediction)
        distance_scores = []

        if -1 in cluster_labels:
            cluster_labels = cluster_labels[1:]

        for k in cluster_labels:

            gene_cluster_mask = (self.current_prediction == k)[:self.ge_count]
            ccre_cluster_mask = (self.current_prediction == k)[self.ge_count:]
            current_distance_score_matrix = distance_score_matrix[gene_cluster_mask, :]
            current_distance_score_matrix = current_distance_score_matrix[:, ccre_cluster_mask]
            # If we have an "only ccres" or "only genes" cluster, we put distance score directly to zero
            distance_score = 0

            if current_distance_score_matrix.shape[0] != 0 and current_distance_score_matrix.shape[1] != 0:
                # Notice that when the clusters are bigger, then it will be more difficult
                # to reach a good distance score. That is why we now give a  normalization factor:
                # cluster_dim = current_distance_score_matrix.shape[0] + current_distance_score_matrix.shape[1]
                # distance_score = current_distance_score_matrix.mean() * (cluster_dim ** 0.3)
                distance_score = current_distance_score_matrix.mean()

            distance_scores.append(distance_score)

        if len(distance_scores) == 0:
            return 0
        else:
            return sum(distance_scores) / len(distance_scores)

    def get_mean_heterogeneity(self):
        cluster_heterogeneities = []
        le_classes = np.unique(self.current_prediction)
        if -1 in le_classes:
            le_classes = le_classes[1:]

        for cluster in le_classes:
            cluster_points = _safe_indexing(self.class_label_array, self.current_prediction == cluster)
            cluster_gene_count = np.count_nonzero(cluster_points == 'gene')
            cluster_ccre_count = np.count_nonzero(cluster_points == 'ccre')
            current_ccres_over_genes_ratio = cluster_ccre_count / 1 + cluster_gene_count
            heterogeneity_drift = abs(current_ccres_over_genes_ratio - self.global_ccres_over_genes_ratio)
            current_heterogeneity = 1 / (1 + heterogeneity_drift)
            cluster_heterogeneities.append(current_heterogeneity)

        if len(cluster_heterogeneities) == 0:
            return 0
        else:
            mean_heterogeneity = sum(cluster_heterogeneities) / len(cluster_heterogeneities)
            return mean_heterogeneity


    def get_mean_cluster_conciseness(self, data_points, labels):

        cluster_concisenesses = []
        cluster_labels = np.unique(labels)

        for k in cluster_labels:

            cluster_k_components = data_points[labels == k]
            centroid_k = np.mean(cluster_k_components, axis=0)
            dispersion_vectors_k = (cluster_k_components - centroid_k) ** 2

            gene_dispersions_k = np.sum(dispersion_vectors_k, axis=1)
            gene_diameter_k = gene_dispersions_k.max()

            if gene_diameter_k == 0:
                mean_scaled_gene_dispersion_k = 0

            else:
                scaled_gene_dispersions_k = gene_dispersions_k / gene_diameter_k
                mean_scaled_gene_dispersion_k = scaled_gene_dispersions_k.mean()
            cluster_concisenesses.append(1 / (1 + mean_scaled_gene_dispersion_k))

        if len(cluster_concisenesses) == 0:
            return 0

        else:
            return sum(cluster_concisenesses) / len(cluster_concisenesses)


    def get_raw_score(self):

        ges = self.X.detach().cpu().numpy()[:self.ge_count][:, :8]
        gene_labels = self.current_prediction[:self.ge_count]
        scattered_genes = 0
        if -1 in np.unique(gene_labels):
            scattered_gene_indexes = np.where(gene_labels == -1)
            scattered_genes = len(scattered_gene_indexes[0])
            clustered_genes = np.delete(ges, scattered_gene_indexes, 0)
            valid_gene_labels = np.delete(gene_labels, scattered_gene_indexes)
            ge_cc = self.get_mean_cluster_conciseness(clustered_genes, valid_gene_labels)
        else:
            ge_cc = self.get_mean_cluster_conciseness(ges, gene_labels)

        ge_clustering_completeness = 1 - scattered_genes / self.ge_count

        ccre_as = self.X.detach().cpu().numpy()[self.ge_count:][:, 8:]
        ccre_labels = self.current_prediction[self.ge_count:]
        scattered_ccres = 0
        if -1 in np.unique(ccre_labels):
            scattered_ccre_indexes = np.where(ccre_labels == -1)
            scattered_ccres = len(scattered_ccre_indexes[0])
            clustered_ccres = np.delete(ccre_as, scattered_ccre_indexes, 0)
            valid_ccre_labels = np.delete(ccre_labels, scattered_ccre_indexes)
            ccre_ch = self.get_mean_cluster_conciseness(clustered_ccres, valid_ccre_labels)
        else:
            ccre_ch = self.get_mean_cluster_conciseness(ccre_as, ccre_labels)

        ccre_clustering_completeness = 1 - scattered_ccres / self.ccre_count

        return ge_cc, ccre_ch, ge_clustering_completeness, ccre_clustering_completeness


    def step(self, action):

        self.global_step += 1
        self.iteration += 1

        action = action.detach().to('cpu').numpy()

        self.current_rq_loss_weight = action[0]
        self.tensorboard.add_scalar(RQ_LOSS_WEIGHT, float(self.current_rq_loss_weight), self.global_step)

        self.prev_sparsity = self.current_sparsity
        self.current_sparsity = int(action[1])
        self.current_gene_sparsity = math.ceil(self.current_sparsity / self.global_ccres_over_genes_ratio)
        self.tensorboard.add_scalar(SPARSITY_LABEL, self.current_sparsity, self.global_step)
        self.tensorboard.add_scalar(GENE_SPARSITY_LABEL, self.current_gene_sparsity, self.global_step)

        self.prev_gbf = self.current_genetic_balance_factor
        self.current_genetic_balance_factor = action[2]
        self.tensorboard.add_scalar(GENETIC_BALANCE_FACTOR_LABEL, float(self.current_genetic_balance_factor),
                               self.global_step)

        self.current_attractive_loss_weight = action[3]
        self.tensorboard.add_scalar(ATTRACTIVE_CE_LOSS_WEIGHT_LABEL, self.current_attractive_loss_weight, self.global_step)

        self.current_lambda_attractive = action[4]
        self.tensorboard.add_scalar(LAMBDA_ATTRACTIVE_LABEL, self.current_lambda_attractive, self.global_step)

        self.current_repulsive_loss_weight = action[5]
        self.tensorboard.add_scalar(REPULSIVE_CE_LOSS_WEIGHT_LABEL, self.current_repulsive_loss_weight, self.global_step)

        self.current_lambda_repulsive = action[6]
        self.tensorboard.add_scalar(LAMBDA_REPULSIVE_LABEL, self.current_lambda_repulsive, self.global_step)

        self.current_rep_agg_loss_weight = action[7]
        self.tensorboard.add_scalar(RP_AGGRESSIVE_LOSS_WEIGHT, self.current_rep_agg_loss_weight, self.global_step)

        self.update_graph()
        self.current_cluster_number = math.floor((self.ge_count + self.ccre_count) / self.current_sparsity)
        self.tensorboard.add_scalar(CLUSTER_NUMBER_LABEL, self.current_cluster_number, self.global_step)

        self.gae_nn.optimizer.zero_grad()

        # recons is the q distribution.
        recons = self.gae_nn(self.norm_adj)
        loss = self.build_loss(recons)

        torch.cuda.empty_cache()
        loss.backward()
        self.gae_nn.optimizer.step()

        return loss


    def evaluate(self):

        gene_cc_score, ccre_cc_score, heterogeneity_score, ge_comp, ccre_comp, distance_score = 0, 0, 0, 0, 0, 0

        if self.current_cluster_number < 30:

            gene_cc_score, ccre_cc_score, heterogeneity_score, ge_comp, ccre_comp, distance_score = self.clustering()
            self.tensorboard.add_scalar(GE_CC_SCORE_TAG, gene_cc_score, self.global_step)
            self.tensorboard.add_scalar(CCRE_CC_SCORE_TAG, ccre_cc_score, self.global_step)
            self.tensorboard.add_scalar(HETEROGENEITY_SCORE_TAG, heterogeneity_score, self.global_step)
            self.tensorboard.add_scalar(GENE_CLUSTERING_COMPLETENESS_TAG, ge_comp, self.global_step)
            self.tensorboard.add_scalar(CCRE_CLUSTERING_COMPLETENESS_TAG, ccre_comp, self.global_step)
            self.tensorboard.add_scalar(DISTANCE_SCORE_TAG, distance_score, self.global_step)

        scaled_ge_cc_score = gene_cc_score
        scaled_ccre_cc_score = ccre_cc_score
        scaled_distance_score = distance_score * 100

        reward = (0.25 * scaled_ge_cc_score) + \
                 (0.25 * scaled_ccre_cc_score) + \
                 (0.25 * scaled_distance_score)

        self.tensorboard.add_scalar(REWARD_TAG, reward, self.global_step)

        return reward


    def get_dinamic_param(self, init_value, final_value, T):
        return init_value + ((final_value - init_value) * (1 / (1 + math.e ** (-1 * (self.iteration - (T/ 2)) / (T/10)))))

    def CAN_precomputed_dist(self, distances):

        element_count = distances.shape[0]

        # symmetrize:
        distances = torch.max(distances, torch.t(distances))

        sorted_distances, _ = distances.sort(dim=1)

        if self.differential_sparsity:

            # distance to the k-th nearest neighbor ONLY GENES:
            top_k_genes = sorted_distances[:self.ge_count, self.current_gene_sparsity]
            top_k_genes = torch.t(top_k_genes.repeat(element_count, 1)) + 10 ** -10
            # summatory of the nearest k distances ONLY GENES:
            sum_top_k_genes = torch.sum(sorted_distances[:self.ge_count, 0:self.current_gene_sparsity], dim=1)
            sum_top_k_genes = torch.t(sum_top_k_genes.repeat(element_count, 1))

            # numerator of equation 20 in the paper ONLY GENES
            T_genes = top_k_genes - distances[:self.ge_count, ]

            # equation 20 in the paper. notice that self.current_sparsity = k. ONLY GENES
            #weights_genes = torch.div(T_genes, self.current_gene_sparsity * top_k_genes - sum_top_k_genes)
            weights_genes = T_genes / top_k_genes


            # distance to the k-th nearest neighbor ONLY CCRES:
            top_k_ccres = sorted_distances[self.ge_count:, self.current_sparsity]
            top_k_ccres = torch.t(top_k_ccres.repeat(element_count, 1)) + 10 ** -10

            # summatory of the nearest k distances ONLY CCRES:
            sum_top_k_ccres = torch.sum(sorted_distances[self.ge_count:, 0:self.current_sparsity], dim=1)
            sum_top_k_ccres = torch.t(sum_top_k_ccres.repeat(element_count, 1))

            # numerator of equation 20 in the paper ONLY CCRES
            T_ccres = top_k_ccres - distances[self.ge_count:, ]

            # equation 20 in the paper. notice that self.current_sparsity = k. ONLY CCRES
            #weights_ccres = torch.div(T_ccres, self.current_sparsity * top_k_ccres - sum_top_k_ccres)
            weights_ccres = T_ccres / top_k_ccres

            weights = torch.cat((weights_genes, weights_ccres))

        else:

            sorted_distances, _ = distances.sort(dim=1)
            top_k = sorted_distances[:, self.current_sparsity]
            top_k = torch.t(top_k.repeat(element_count, 1)) + 10 ** -10

            sum_top_k = torch.sum(sorted_distances[:, 0:self.current_sparsity], dim=1)
            sum_top_k = torch.t(sum_top_k.repeat(element_count, 1))

            T = top_k - distances

            weights = torch.div(T, self.current_sparsity * top_k - sum_top_k)

        weights = weights.relu()

        weights = (weights + weights.t()) / 2

        return weights

    def compute_S_D(self):

        # Notice that the link distance matrix has already self loop weight information
        S_D = fast_genomic_distance_to_similarity(self.links, self.current_genomic_C,
                                                       self.current_genomic_slope)

        S_D = torch.tensor(S_D) * self.kendall_matrix

        # Even if it is not 100% correct, we do a row-wise scaling here because
        # genomic bp distance scores are very weak
        S_D /= (S_D.max(dim=1)[0] + 1e-10).reshape(-1,1)

        self.S_D = S_D


    def compute_P(self, prev_embedding, first_time=False):

        tras_prev_embedding = prev_embedding.t().detach()

        element_count = tras_prev_embedding.shape[1]

        if first_time:
            self.D_Z = torch.full((element_count, element_count),self.gene_ccre_dist_factor)
            self.D_Z[:self.ge_count,:self.ge_count] = self.D_G
            denominator = (self.init_alpha_ACET + self.init_alpha_METH + self.init_alpha_ATAC)
            D_CCRES = ((self.init_alpha_METH *  self.D_METH) +
                       (self.init_alpha_ACET * self.D_ACET) +
                       (self.init_alpha_ATAC * self.D_ATAC)) / denominator
            self.D_Z[self.ge_count:,self.ge_count:] = D_CCRES
            self.S_Z = self.CAN_precomputed_dist(self.D_Z)

        elif self.prev_sparsity != self.current_sparsity:
            self.D_Z = distance(tras_prev_embedding, tras_prev_embedding)
            self.D_Z = torch.max(self.D_Z, torch.t(self.D_Z))
            # abs scaling
            self.D_Z /= self.D_Z.max()
            self.S_Z = self.CAN_precomputed_dist(self.D_Z)

        # After computing the distances being based on current embedding,
        # we add weight to some edges of the graph
        # based on the original graph information.


        self.adj = ((1 - self.current_genetic_balance_factor) * self.S_Z) \
                   + self.current_genetic_balance_factor * self.S_D

        # row-wise normalization
        self.adj /= (self.adj.sum(dim=1) + 1e-10).reshape([-1, 1])

        # UN-symmetric connectivity distribution
        self.raw_adj = self.adj.clone()

        # Symmetrization of the connectivity distribution
        self.adj = (self.adj + self.adj.t()) / 2

        self.raw_adj = self.raw_adj.float().to(self.device)
        self.adj = self.adj.float().to(self.device)



    def clustering(self):

        cpu_embedding = self.gae_nn.embedding.detach().cpu().numpy()
        km = KMeans(n_clusters=self.current_cluster_number).fit(cpu_embedding)
        self.current_prediction = km.predict(cpu_embedding)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples)
        # self.current_prediction = clusterer.fit_predict(cpu_embedding)

        return self.cal_clustering_metric()


    def plot_clustering(self, bi_dim_embedding=None, title=None):

        if self.layers[-1] > 2:
            if bi_dim_embedding == None:
                bi_dim_embedding = umap.UMAP(
                    n_neighbors=30,
                    min_dist=0
                ).fit_transform(self.gae_nn.embedding.detach().cpu().numpy())
        else:
            bi_dim_embedding = self.gae_nn.embedding.detach().cpu()

        cmap = cm.get_cmap('viridis', max(self.current_prediction) + 1)
        markers = ["s", "o", "$f$", "v", "^", "<", ">", "p", "$L$", "x"]

        for cluster in range(0, max(self.current_prediction) + 1):
            cluster_points = _safe_indexing(bi_dim_embedding, self.current_prediction == cluster)
            cluster_marker = markers[cluster % len(markers)]
            plt.scatter(cluster_points[:, 0],
                        cluster_points[:, 1],
                        marker=cluster_marker,
                        color=cmap.colors[cluster],
                        label='Cluster' + str(cluster))
        plt.legend()
        if title != None:
            plt.title(title)
        if not eval:
            self.send_image_to_tensorboard(plt, UMAP_CLUSTER_PLOT_TAG)
        plt.show()


    def plot_classes(self, bi_dim_embedding=None, only_ccres=False, title=None):

        if self.layers[-1] > 2:
            if bi_dim_embedding == None:
                bi_dim_embedding = umap.UMAP(
                    n_neighbors=30,
                    min_dist=0
                ).fit_transform(self.gae_nn.embedding.detach().cpu())
        else:
            bi_dim_embedding = self.gae_nn.embedding.detach().cpu()

        class_labels = np.array(self.ge_class_labels + ['ccres'] * self.ccre_count)
        classes = np.unique(class_labels)

        classplot_alphas = [0.3, 1, 1, 1, 1]
        classplot_markers = ["o", "^", "v", "v", "^"]
        classplot_colors = ['aquamarine', 'r', 'g', 'g', 'r']
        classplot_sizes = [10, 40, 40, 40, 40]
        for idx, elem_class in enumerate(classes):
            cluster_points = _safe_indexing(bi_dim_embedding, class_labels == elem_class)
            cluster_marker = classplot_markers[idx]
            cluster_color = classplot_colors[idx]
            cluster_marker_size = classplot_sizes[idx]
            plt.scatter(cluster_points[:, 0],
                        cluster_points[:, 1],
                        marker=cluster_marker,
                        color=cluster_color,
                        label=elem_class,
                        alpha=classplot_alphas[idx],
                        s=cluster_marker_size)
            if only_ccres:
                break
        plt.legend()
        if title != None:
            plt.title(title)
        if not eval:
            self.send_image_to_tensorboard(plt, UMAP_CLASS_PLOT_TAG)
        plt.show()


    def send_image_to_tensorboard(self, plt, tag):
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).squeeze(0)
        self.tensorboard.add_image(tag, image, self.global_step)


    def perform_spectral_clusering_on_adj_matrix(self):
        clustering = SpectralClustering(n_clusters=self.current_cluster_number,
                                        affinity='precomputed_nearest_neighbors', n_neighbors=self.current_sparsity)
        clustering.fit(self.adj.detach().cpu().numpy())
        self.current_prediction = clustering.labels_


    def run_1_epoch(self,current_sparsity=200,gbf=0,rq_loss_weight=0,attractive_loss_weight=1,
                    repulsive_loss_weight=1,lambda_attractive=0.5,lambda_repulsive=0.5,agg_repulsive=0,max_iter=15):

        self.epoch_losses = []

        for i in range(max_iter):

            dummy_action = torch.Tensor([rq_loss_weight,
                                         current_sparsity,
                                         gbf,
                                         attractive_loss_weight,
                                         lambda_attractive,
                                         repulsive_loss_weight,
                                         lambda_repulsive,
                                         agg_repulsive]).to(self.device)

            loss = self.step(dummy_action)

            self.epoch_losses.append(loss.item())



        return sum(self.epoch_losses) / len(self.epoch_losses)



def save(gae, epoch, datapath, modelname):
    torch.save(gae.gae_nn.state_dict(), datapath + 'models' + modelname + '_model_' + str(epoch) + '_epochs')
    torch.save(gae.gae_nn.embedding, datapath + 'models' + modelname + '_embedding_' + str(epoch) + '_epochs')


#@profile(output_file='profiling_adagae')
def manual_run(gae,
               max_epoch=10,
               init_sparsity=200,
               sparsity_increment=40,
               init_gbf=0.7,
               final_gbf=0,
               init_RQ_loss_weight=0.1,
               final_RQ_loss_weight=1,
               init_attractive_loss_weight=0.1,
               final_attractive_loss_weight=1,
               init_repulsive_loss_weight=1,
               final_repulsive_loss_weight=0.1,
               init_lambda_attractive=0.5,
               final_lambda_attractive=0.5,
               init_lambda_repulsive=0.5,
               final_lambda_repulsive=0.5,
               init_agg_repulsive=1,
               final_agg_repulsive=0.1,
               max_iter=15):

    current_sparsity = init_sparsity
    epoch=0

    while epoch < max_epoch:

        epoch += 1

        T = max_epoch * max_iter

        current_genetic_balance_factor = gae.get_dinamic_param(init_gbf, final_gbf, T)
        current_rq_loss_weight = gae.get_dinamic_param(init_RQ_loss_weight, final_RQ_loss_weight, T)

        current_attractive_loss_weight = gae.get_dinamic_param(init_attractive_loss_weight,
                                                               final_attractive_loss_weight, T)

        current_repulsive_loss_weight = gae.get_dinamic_param(init_repulsive_loss_weight,
                                                               final_repulsive_loss_weight, T)

        current_lambda_attractive = gae.get_dinamic_param(init_lambda_attractive,
                                                         final_lambda_attractive, T)

        current_lambda_repulsive = gae.get_dinamic_param(init_lambda_repulsive,
                                                        final_lambda_repulsive, T)

        current_aggresive_rep_loss_weight = gae.get_dinamic_param(init_agg_repulsive,
                                                        final_agg_repulsive, T)

        gae.epoch_losses = []

        for i in range(max_iter):

            dummy_action = torch.Tensor([current_rq_loss_weight,
                                         current_sparsity,
                                         current_genetic_balance_factor,
                                         current_attractive_loss_weight,
                                         current_lambda_attractive,
                                         current_repulsive_loss_weight,
                                         current_lambda_repulsive,
                                         current_aggresive_rep_loss_weight]).to(gae.device)

            loss = gae.step(dummy_action)

            gae.epoch_losses.append(loss.item())

        mean_loss = sum(gae.epoch_losses) / len(gae.epoch_losses)

        reward = gae.evaluate()

        print('epoch:%3d,' % epoch,
              'reward:%3d,' % reward,
              'mean loss:%3d,' % mean_loss,
              'gbf: %6.3f' % current_genetic_balance_factor,
              'CE_attr_w: %6.3f' % current_attractive_loss_weight,
              'CE_rep_w: %6.3f' % current_repulsive_loss_weight,
              'RQ_w: %6.3f' % current_rq_loss_weight,
              'AR_w: %6.3f' % current_aggresive_rep_loss_weight,
              'spars:%3d, ' % gae.current_sparsity,
              'curr_clust_num:%3d,' % gae.current_cluster_number )

        if gae.layers[-1] > 2:
          if epoch%10==0:
            gae.plot_classes()
        else:
          gae.plot_classes()

        current_sparsity += sparsity_increment


    print('gae.current_cluster_number', gae.current_cluster_number)



def adagae_run(action_index, actions_array, adagae_object, curr_sparsity):

  curr_action = actions_array[action_index]
  print('Curr_action: [GBF: ',curr_action[0], ', ATTR: ',curr_action[1], ', REP: ', curr_action[2], ']')
  adagae_object.run_1_epoch(current_sparsity = curr_sparsity,
                            gbf = curr_action[0],
                            attractive_loss_weight = curr_action[1],
                            repulsive_loss_weight = curr_action[2],
                            )
  adagae_object.plot_classes()
  return adagae_object.evaluate()
