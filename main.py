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
datapath = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\GE_Datasets\\'

reports_path = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\RL_developmental_studies\\Reports\\tight_var_data\\'

LOG_DIR = 'local_runs/'

##COPY TO NOTEBOOK FROM HERE!!!###


import torch
from torch.utils.tensorboard import SummaryWriter
import cProfile
import pstats
from functools import wraps
import pandas as pd
import numpy as np
import io
import math
import PIL.Image
from torchvision.transforms import ToTensor
import hdbscan
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.utils import _safe_indexing

import umap
import umap.plot
import matplotlib.pyplot as plt

######
### SOME CONSTATNS
######

plt.rcParams["figure.figsize"] = (10, 10)

# generate a list of markers and another of colors
markers = ["s", "o", "$f$", "v", "^", "<", ">", "p", "$L$", "x"]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'aquamarine', 'tab:gray', 'xkcd:sky blue']
sizes = [32, 36, 39, 34, 37, 38, 32, 33, 35, 37]


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


def get_genomic_distance_matrix(link_ds):
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

    for index, row in tqdm(link_ds.reset_index().iterrows()):
        gene_idx = entities_df.loc[row.EnsembleID][0]
        ccre_idx = entities_df.loc[row.cCRE_ID][0]
        dense_A[gene_idx, ccre_idx] = row.Distance
        dense_A[ccre_idx, gene_idx] = row.Distance

    return dense_A


def get_primitive_clusters(link_ds, ccre_ds):
    gene_primitive_clusters_path = reports_path + 'GE clustering/'
    kmeans_ds = pd.read_csv(gene_primitive_clusters_path + 'variable_k/kmeans_clustered_genes_5.csv')
    primitive_ge_clustered_ds = kmeans_ds.set_index('EnsembleID').drop('Unnamed: 0', axis=1)
    primitive_ge_clustered_ds.columns = ['primitive_cluster']
    gene_ds = link_ds.reset_index().drop_duplicates('EnsembleID').set_index('EnsembleID')
    prim_gene_ds = gene_ds.join(primitive_ge_clustered_ds)['primitive_cluster'].reset_index()

    ccre_primitive_clusters_path = reports_path + 'cCRE Clustering/'
    ccre_agglomerative_ds = pd.read_csv(ccre_primitive_clusters_path + 'variable_k/agglomerative_clust_cCRE_8.csv')
    prim_ccre_ds = ccre_ds.set_index('cCRE_ID').join(ccre_agglomerative_ds.set_index('cCRE_ID'))[['cluster']]
    prim_ccre_ds.columns = ['primitive_cluster']
    prim_ccre_ds.primitive_cluster += 5
    return np.array(prim_gene_ds.primitive_cluster.to_list() + prim_ccre_ds.primitive_cluster.to_list())


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

    ccre_ds = pd.read_csv(datapath + 'cCRE_variational_mean_reduced.csv')
    if chr_to_filter != None:
        filtered_ccre_ds = pd.DataFrame()
        for chr_number in chr_to_filter:
            filtered_ccre_ds = pd.concat([filtered_ccre_ds, ccre_ds[ccre_ds['cCRE_ID'].str.startswith('chr'+str(chr_number))]])

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


def get_primitive_gene_clusters():
    gene_primitive_clusters_path = reports_path
    kmeans_ds = pd.read_csv(gene_primitive_clusters_path + 'kmeans_clustered_genes_4.csv')
    primitive_ge_clustered_ds = kmeans_ds.set_index('EnsembleID').drop('Unnamed: 0', axis=1)
    primitive_ge_clustered_ds.columns = ['primitive_cluster']
    gene_ds = link_ds.reset_index().drop_duplicates('EnsembleID').set_index('EnsembleID')
    prim_gene_ds = gene_ds.join(primitive_ge_clustered_ds)['primitive_cluster'].reset_index()

    return np.array(prim_gene_ds.primitive_cluster.to_list())




#####
# ADAGAE OBJECT
########

SPARSITY_LABEL: str = 'Sparsity'
GENE_SPARSITY_LABEL: str = 'Gene_Sparsity'
SLOPE_LABEL: str = 'GeneticSlope'
REPULSIVE_CE_TERM: str = 'Repulsive_CE_loss'
ATTRACTIVE_CE_TERM: str = 'Attractive_CE_loss'
RQ_QUOTIENT_LOSS: str = 'RQ Quotient Loss'
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
LAMBDA_REPULSIVE_LABEL:str = 'Lambda Repulsive'
LAMBDA_ATTRACTIVE_LABEL:str= 'Lambda Attractive'
LAMBDA_RQ_LABEL:str= 'Lambda RQ'

class AdaGAE_NN(torch.nn.Module):

    def __init__(self,
                 data_matrix,
                 device,
                 pre_trained,
                 pre_trained_state_dict,
                 pre_computed_embedding,
                 gcn
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
        if softmax_reconstruction:
            softmax = torch.nn.Softmax(dim=1)
            recons_w = softmax(-distances)
        else:
            recons_w = 1 / (distances + 1)
            recons_w /= recons_w.sum(dim=1).reshape([recons_w.shape[0], 1])
        return recons_w + 10 ** -10


class AdaGAE():

    def __init__(self, X,
                 device=None,
                 pre_trained=False,
                 pre_trained_state_dict='models/combined_adagae_z12_initk150_150epochs',
                 pre_computed_embedding='models/combined_adagae_z12_initk150_150epochs_embedding'):

        super(AdaGAE, self).__init__()

        self.device = device
        if self.device is None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.X = X
        self.pre_trained = pre_trained
        self.pre_trained_state_dict = pre_trained_state_dict
        self.pre_computed_embedding = pre_computed_embedding
        self.global_step = 0
        self.global_ccres_over_genes_ratio = (ccre_count / ge_count)
        self.reset()

        classes = ['gene', 'ccre']
        self.class_label_array = np.array([classes[0]] * ge_count + [classes[1]] * ccre_count)


    def reset(self):
        self.iteration = 0
        self.gae_nn = None
        torch.cuda.empty_cache()
        self.gae_nn = AdaGAE_NN(self.X,
                                self.device,
                                self.pre_trained,
                                self.pre_trained_state_dict,
                                self.pre_computed_embedding,
                                gcn).to(self.device)
        self.current_sparsity = init_sparsity + 1
        self.current_gene_sparsity = math.ceil(self.current_sparsity / self.global_ccres_over_genes_ratio)
        if self.current_gene_sparsity == 0: self.current_gene_sparsity += 1
        self.current_genomic_slope = init_genomic_slope
        self.current_genomic_C = init_genomic_C
        self.current_genetic_balance_factor = init_gbf
        self.current_rq_loss_weight = init_RQ_loss_weight
        self.current_cluster_number = init_cluster_num
        self.init_adj_matrices()
        self.current_repulsive_loss_weight = init_repulsive_loss_weight
        self.current_attractive_ce_loss_weight = init_attractive_loss_weight
        self.current_lambda_attractive = init_lambda_attractive
        self.current_lambda_repulsive = init_lambda_repulsive
        self.current_lambda_rq = init_lambda_rq
        if not self.pre_trained: self.init_embedding()

    def init_adj_matrices(self):
        if not eval:
            tensorboard.add_scalar(SPARSITY_LABEL, self.current_sparsity, self.global_step)
            tensorboard.add_scalar(GENE_SPARSITY_LABEL, self.current_gene_sparsity, self.global_step)
        # adj is A tilded, it is the symmetric modification of the p distribution
        # raw_adj is the p distribution before the symetrization.
        if self.pre_trained:
            self.adj, self.raw_adj = self.cal_weights_via_CAN(self.gae_nn.embedding.t())
        else:
            self.adj, self.raw_adj = self.cal_weights_via_CAN(self.X.t())

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
            _ = self.gae_nn(self.norm_adj.to(device))
            _ = None
            torch.cuda.empty_cache()

    def cal_max_neighbors(self):
        size = self.X.shape[0]
        return 2.0 * size / init_cluster_num

    def update_graph(self):
        self.adj, self.raw_adj = self.cal_weights_via_CAN(self.gae_nn.embedding.t())
        self.adj = self.adj.detach()
        self.raw_adj = self.raw_adj.detach()
        # threshold = 0.5
        # connections = (recons > threshold).type(torch.IntTensor).cuda()
        # weights = weights * connections
        self.norm_adj = get_normalized_adjacency_matrix(self.adj)
        return self.adj, self.norm_adj, self.raw_adj


    def build_loss(self, recons):

        self.adj = self.adj.to(device)
        self.raw_adj = self.raw_adj.to(device)

        size = self.X.shape[0]
        loss = 0
        '''
        # notice that recons is actually the q distribution.
        # and that raw_weigths is the p distribution. (before the symmetrization)
        '''
        # This acts as a REPULSIVE force for the embedding learning:
        repulsive_CE_term = -(self.raw_adj * torch.log(recons + 10 ** -10))

        if differential_repulsive_forces:
            repulsive_CE_term[:ge_count] *= self.current_lambda_repulsive
            repulsive_CE_term[ge_count:] *= (1 - self.current_lambda_repulsive)
        repulsive_CE_term = repulsive_CE_term.sum(dim=1)
        repulsive_CE_term = repulsive_CE_term.mean()

        tensorboard.add_scalar(REPULSIVE_CE_TERM, repulsive_CE_term.item(), self.global_step)


        # If we use the Fuzzy Cross Entropy. (Binary cross entropy)
        # We could create an ATTRACTIVE force
        # https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668
        attractive_CE_term = -(1 - self.raw_adj) * torch.log(1-recons)

        if differential_attractive_forces:
            attractive_CE_term[:ge_count] *= self.current_lambda_attractive
            attractive_CE_term[ge_count:] *= (1 - self.current_lambda_attractive)

        attractive_CE_term = attractive_CE_term.sum(dim=1)
        attractive_CE_term = attractive_CE_term.mean()

        tensorboard.add_scalar(ATTRACTIVE_CE_TERM, attractive_CE_term.item(), self.global_step)

        # This loss acts as an attractive force for the embedding:
        # It strengthens element-wise similarities
        degree = self.adj.sum(dim=1)
        laplacian = torch.diag(degree) - self.adj

        if diff_RQ:
            # We should differentiate this attractive force based on each "modality"
            ge_rq_loss = torch.trace(
                (self.gae_nn.embedding[:ge_count]).t().matmul(laplacian[:ge_count, :ge_count]).matmul(
                    (self.gae_nn.embedding[:ge_count]))) / ge_count
            ccre_rq_loss = torch.trace(
                (self.gae_nn.embedding[ge_count:]).t().matmul(laplacian[ge_count:,ge_count:]).matmul(
                    (self.gae_nn.embedding[ge_count:]))) / ccre_count
            rayleigh_quotient_loss = (ge_rq_loss * self.current_lambda_rq) + (ccre_rq_loss * (1 - self.current_lambda_rq))
        else:
            # This is exactly equation 11 in the AdaGAE paper.
            # Notice that torch.trace return the sum of the elements in the diagonal of the input matrix.
            rayleigh_quotient_loss = torch.trace(
                self.gae_nn.embedding.t().matmul(laplacian).matmul(self.gae_nn.embedding)) / size

        tensorboard.add_scalar(RQ_QUOTIENT_LOSS, rayleigh_quotient_loss.item(), self.global_step)

        loss += self.current_repulsive_loss_weight * repulsive_CE_term
        loss += self.current_attractive_ce_loss_weight * attractive_CE_term
        loss += self.current_rq_loss_weight * rayleigh_quotient_loss

        tensorboard.add_scalar(TOTAL_LOSS_LABEL, loss.item(), self.global_step)


        self.adj.to('cpu')
        self.raw_adj.to('cpu')

        return loss

    def cal_clustering_metric(self, predicted_labels):

        ge_cc_raw, ccre_cc_raw, ge_clust_completeness, ccre_clust_completeness = self.get_raw_score(predicted_labels)
        mean_heterogeneity = self.get_mean_heterogeneity(predicted_labels)
        distance_score = self.get_mean_distance_scores(predicted_labels)
        return ge_cc_raw, ccre_cc_raw, mean_heterogeneity, ge_clust_completeness, ccre_clust_completeness, distance_score

    def get_mean_distance_scores(self, predicted_labels):
        distance_score_matrix = self.current_link_score[:ge_count, ge_count:]
        cluster_labels = np.unique(predicted_labels)
        distance_scores = []

        if -1 in cluster_labels:
            cluster_labels = cluster_labels[1:]

        for k in cluster_labels:
            gene_cluster_mask = (predicted_labels == k)[:ge_count]
            ccre_cluster_mask = (predicted_labels == k)[ge_count:]
            current_distance_score_matrix = distance_score_matrix[gene_cluster_mask, :]
            current_distance_score_matrix = current_distance_score_matrix[:, ccre_cluster_mask]
            # If we have an "only ccres" or "only genes" cluster, we put distance score directly to zero
            distance_score = 0
            if current_distance_score_matrix.shape[0] != 0 and current_distance_score_matrix.shape[1] != 0:
                # Notice that when the clusters are bigger, then it will be more difficult
                # to reach a good distance score. That is why we now give a  normalization factor:
                cluster_dim = current_distance_score_matrix.shape[0] + current_distance_score_matrix.shape[1]
                distance_score = current_distance_score_matrix.mean() * (cluster_dim**0.3)
            distance_scores.append(distance_score)

        if len(distance_scores)==0:
            return 0
        else:
            return sum(distance_scores) / len(distance_scores)

    def get_mean_heterogeneity(self, predicted_labels):
        cluster_heterogeneities = []
        le_classes = np.unique(predicted_labels)
        if -1 in le_classes:
            le_classes = le_classes[1:]

        for cluster in le_classes:
            cluster_points = _safe_indexing(self.class_label_array, predicted_labels == cluster)
            cluster_gene_count = np.count_nonzero(cluster_points == 'gene')
            cluster_ccre_count = np.count_nonzero(cluster_points == 'ccre')
            current_ccres_over_genes_ratio = cluster_ccre_count / 1 + cluster_gene_count
            heterogeneity_drift = abs(current_ccres_over_genes_ratio - self.global_ccres_over_genes_ratio)
            current_heterogeneity = 1 / (1 + heterogeneity_drift)
            cluster_heterogeneities.append(current_heterogeneity)

        if len(cluster_heterogeneities)== 0:
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

        if len(cluster_concisenesses)==0:
            return 0
        else:
            return sum(cluster_concisenesses) / len(cluster_concisenesses)

    def get_raw_score(self, predicted_labels):

        ges = self.X.detach().cpu().numpy()[:ge_count][:, :8]
        gene_labels = predicted_labels[:ge_count]
        scattered_genes = 0
        if -1 in np.unique(gene_labels):
            scattered_gene_indexes = np.where(gene_labels == -1)
            scattered_genes = len(scattered_gene_indexes[0])
            clustered_genes = np.delete(ges, scattered_gene_indexes, 0)
            valid_gene_labels = np.delete(gene_labels, scattered_gene_indexes)
            ge_cc = self.get_mean_cluster_conciseness(clustered_genes, valid_gene_labels)
        else:
            ge_cc = self.get_mean_cluster_conciseness(ges, gene_labels)

        ge_clustering_completeness = 1 - scattered_genes / ge_count

        ccre_as = self.X.detach().cpu().numpy()[ge_count:][:, 8:]
        ccre_labels = predicted_labels[ge_count:]
        scattered_ccres = 0
        if -1 in np.unique(ccre_labels):
            scattered_ccre_indexes = np.where(ccre_labels == -1)
            scattered_ccres = len(scattered_ccre_indexes[0])
            clustered_ccres = np.delete(ccre_as, scattered_ccre_indexes, 0)
            valid_ccre_labels = np.delete(ccre_labels, scattered_ccre_indexes)
            ccre_ch = self.get_mean_cluster_conciseness(clustered_ccres, valid_ccre_labels)
        else:
            ccre_ch = self.get_mean_cluster_conciseness(ccre_as, ccre_labels)

        ccre_clustering_completeness = 1 - scattered_ccres / ccre_count

        return ge_cc, ccre_ch, ge_clustering_completeness, ccre_clustering_completeness

    def step(self, action):
        self.global_step += 1
        self.iteration += 1
        action = action.detach().to('cpu').numpy()
        self.current_rq_loss_weight = action[0]
        tensorboard.add_scalar(RQ_LOSS_WEIGHT, float(self.current_rq_loss_weight), self.global_step)
        self.current_sparsity = int(action[1])
        self.current_gene_sparsity = math.ceil(self.current_sparsity / self.global_ccres_over_genes_ratio)
        tensorboard.add_scalar(SPARSITY_LABEL, self.current_sparsity, self.global_step)
        tensorboard.add_scalar(GENE_SPARSITY_LABEL, self.current_gene_sparsity, self.global_step)
        self.current_genomic_slope = action[2]
        tensorboard.add_scalar(SLOPE_LABEL, self.current_genomic_slope, self.global_step)
        self.current_genetic_balance_factor = action[3]
        tensorboard.add_scalar(GENETIC_BALANCE_FACTOR_LABEL, float(self.current_genetic_balance_factor),self.global_step)
        self.current_genomic_C = action[4]
        tensorboard.add_scalar(GENOMIC_C_LABEL, self.current_genomic_C, self.global_step)
        self.current_repulsive_loss_weight = action[5]
        tensorboard.add_scalar(REPULSIVE_CE_LOSS_WEIGHT_LABEL, self.current_repulsive_loss_weight, self.global_step)
        self.current_attractive_ce_loss_weight = action[6]
        tensorboard.add_scalar(ATTRACTIVE_CE_LOSS_WEIGHT_LABEL, self.current_attractive_ce_loss_weight, self.global_step)
        self.current_lambda_repulsive = action[7]
        tensorboard.add_scalar(LAMBDA_REPULSIVE_LABEL, self.current_lambda_repulsive, self.global_step)
        self.current_lambda_attractive = action[8]
        tensorboard.add_scalar(LAMBDA_ATTRACTIVE_LABEL, self.current_lambda_attractive, self.global_step)
        self.current_lambda_rq = action[9]
        tensorboard.add_scalar(LAMBDA_RQ_LABEL, self.current_lambda_rq, self.global_step)



        self.gae_nn.optimizer.zero_grad()

        # recons is the q distribution.
        recons = self.gae_nn(self.norm_adj)
        loss = self.build_loss(recons)

        torch.cuda.empty_cache()
        loss.backward()
        self.gae_nn.optimizer.step()

        done_flag = False

        gene_cc_score, ccre_cc_score, heterogeneity_score, ge_comp, ccre_comp, distance_score = 0, 0, 0, 0, 0, 0


        if self.iteration % (graphical_report_period_epochs * max_iter) == 0:
            #visual_clustering = False
            #if self.iteration % (2 * max_iter) == 0:
            #    visual_clustering = True
            visual_clustering = True
            gene_cc_score, ccre_cc_score, heterogeneity_score, ge_comp, ccre_comp, distance_score = self.clustering(
                visual_clustering)
            tensorboard.add_scalar(GE_CC_SCORE_TAG, gene_cc_score, self.global_step)
            tensorboard.add_scalar(CCRE_CC_SCORE_TAG, ccre_cc_score, self.global_step)
            tensorboard.add_scalar(HETEROGENEITY_SCORE_TAG, heterogeneity_score, self.global_step)
            tensorboard.add_scalar(GENE_CLUSTERING_COMPLETENESS_TAG, ge_comp, self.global_step)
            tensorboard.add_scalar(CCRE_CLUSTERING_COMPLETENESS_TAG, ccre_comp, self.global_step)
            tensorboard.add_scalar(DISTANCE_SCORE_TAG, distance_score, self.global_step)

        # reward = 1/(loss.item()+1) + ge_ch_score/1 + ccre_ch_score/300
        '''
        scaled_ge_cc_score = gene_cc_score / 100
        scaled_ccre_cc_score = ccre_cc_score / 400
        scaled_heterogeneity = heterogeneity_score / 20

        reward = (0.25 * scaled_ge_cc_score) + \
                 (0.25 * scaled_ccre_cc_score) + \
                 (0.25 * scaled_heterogeneity) + \
                 (0.125 * ge_comp) + \
                 (0.125 * ccre_comp) + \
                 (0.25 * distance_score)

        tensorboard.add_scalar(REWARD_TAG, reward, self.global_step)
        '''
        return 0, loss, done_flag

    @profile(output_file='profiling_adagae')
    def dummy_run(self):

        self.gae_nn.to(self.device)
        for epoch in tqdm(range(max_epoch)):
            self.epoch_losses = []
            if epoch % 10 == 0:
                save(epoch)
            for i in range(max_iter):
                dummy_action = torch.Tensor([self.current_rq_loss_weight,
                                             self.current_sparsity,
                                             self.current_genomic_slope,
                                             self.current_genetic_balance_factor,
                                             self.current_genomic_C,
                                             self.current_repulsive_loss_weight,
                                             self.current_attractive_ce_loss_weight,
                                             self.current_lambda_repulsive,
                                             self.current_lambda_attractive,
                                             self.current_lambda_rq]).to(self.device)

                reward, loss, done_flag = self.step(dummy_action)
                self.epoch_losses.append(loss.item())

            self.current_sparsity += sparsity_increment
            self.current_genetic_balance_factor = self.get_dinamic_param(init_gbf, final_gbf)
            self.current_rq_loss_weight = self.get_dinamic_param(init_RQ_loss_weight, final_RQ_loss_weight)
            self.current_repulsive_loss_weight = self.get_dinamic_param(init_repulsive_loss_weight, final_repulsive_loss_weight)
            self.current_attractive_ce_loss_weight = self.get_dinamic_param(init_attractive_loss_weight, final_attractive_loss_weight)
            self.current_lambda_repulsive = self.get_dinamic_param(init_lambda_repulsive, final_lambda_repulsive)
            self.current_lambda_attractive = self.get_dinamic_param(init_lambda_attractive, final_lambda_attractive)
            self.current_lambda_rq = self.get_dinamic_param(init_lambda_rq, final_lambda_rq)

            self.update_graph()



            mean_loss = sum(self.epoch_losses) / len(self.epoch_losses)
            print('epoch:%3d,' % epoch, 'loss: %6.5f' % mean_loss)


    def get_dinamic_param(self, init_value, final_value):
        T = max_epoch * max_iter
        return init_value + ((final_value - init_value) * (1 / (1 + math.e ** (-1 * (self.global_step - (T/ 2)) / (T/10)))))

    def cal_weights_via_CAN(self, transposed_data_matrix):
        """
        Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
        See section 3.6 of the paper! (specially equation 20)
        """
        size = transposed_data_matrix.shape[1]

        distances = distance(transposed_data_matrix, transposed_data_matrix)
        tensorboard.add_scalar(EMBEDDING_DIAMETER, distances.median()*(1e5), self.global_step)
        distances = torch.max(distances, torch.t(distances))
        sorted_distances, _ = distances.sort(dim=1)

        # distance to the k-th nearest neighbor ONLY GENES:
        top_k_genes = sorted_distances[:ge_count, self.current_gene_sparsity]
        top_k_genes = torch.t(top_k_genes.repeat(size, 1)) + 10 ** -10
        tensorboard.add_scalar(DISTANCE_TO_KNN_TAG, top_k_genes.median() * (1e5), self.global_step)
        # summatory of the nearest k distances ONLY GENES:
        sum_top_k_genes = torch.sum(sorted_distances[:ge_count, 0:self.current_gene_sparsity], dim=1)
        sum_top_k_genes = torch.t(sum_top_k_genes.repeat(size, 1))

        # numerator of equation 20 in the paper ONLY GENES
        T_genes = top_k_genes - distances[:ge_count,]

        # equation 20 in the paper. notice that self.current_sparsity = k. ONLY GENES
        weights_genes = torch.div(T_genes, self.current_gene_sparsity * top_k_genes - sum_top_k_genes)


        # distance to the k-th nearest neighbor ONLY CCRES:
        top_k_ccres = sorted_distances[ge_count:, self.current_sparsity]
        top_k_ccres = torch.t(top_k_ccres.repeat(size, 1)) + 10 ** -10

        # summatory of the nearest k distances ONLY CCRES:
        sum_top_k_ccres = torch.sum(sorted_distances[ge_count:, 0:self.current_sparsity], dim=1)
        sum_top_k_ccres = torch.t(sum_top_k_ccres.repeat(size, 1))

        # numerator of equation 20 in the paper ONLY CCRES
        T_ccres = top_k_ccres - distances[ge_count:, ]

        # equation 20 in the paper. notice that self.current_sparsity = k. ONLY CCRES
        weights_ccres = torch.div(T_ccres, self.current_sparsity * top_k_ccres - sum_top_k_ccres)

        weights_diff = torch.cat((weights_genes,weights_ccres))

        sorted_distances = None
        T_genes = None
        T_ccres = None
        top_k_genes = None
        top_k_ccres = None
        sum_top_k_genes = None
        sum_top_k_ccres = None
        torch.cuda.empty_cache()

        weights = weights_diff.relu().to(self.device)
        '''
        # distance to the k-th nearest neighbor:
        top_k = sorted_distances[:, self.current_sparsity]
        top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

        # summatory of the nearest k distances:
        sum_top_k = torch.sum(sorted_distances[:, 0:self.current_sparsity], dim=1)
        sum_top_k = torch.t(sum_top_k.repeat(size, 1))
        sorted_distances = None
        torch.cuda.empty_cache()

        # numerator of equation 20 in the paper
        T = top_k - distances
        distances = None
        torch.cuda.empty_cache()

        # equation 20 in the paper. notice that self.current_sparsity = k.
        weights = torch.div(T, self.current_sparsity * top_k - sum_top_k)
        T = None
        top_k = None
        sum_top_k = None
        torch.cuda.empty_cache()
        # notice that the following line is also part of equation 20
        weights = weights.relu().to(self.device)
        '''
        # now at this point, after computing the generative model of the
        # k sparse graph being based on node divergences and similarities,
        # we add weight to some points of the connectivity distribution being based
        # on the explicit graph information.

        # Notice that the link distance matrix has already self loop weight information
        self.current_link_score = fast_genomic_distance_to_similarity(links, self.current_genomic_C,
                                                                      self.current_genomic_slope)


        link_scores_tensor = torch.Tensor(self.current_link_score).to(self.device)
        element_max_distance_scores = torch.max(link_scores_tensor, dim=0)[0]
        element_max_similarity_scores = torch.max(weights, dim=0)[0]
        scaled_link_scores = (link_scores_tensor.t() * element_max_similarity_scores / element_max_distance_scores)
        scaled_link_scores = scaled_link_scores.t()




        weights = (weights * (1 - self.current_genetic_balance_factor)) +  (scaled_link_scores * self.current_genetic_balance_factor)


        if not add_self_loops_euclidean:
            # self-loop avoidance (changed to basic GNN model as in hamilton's book)
            weights.fill_diagonal_(0)

        # row-wise normalization.
        weights /= weights.sum(dim=1).reshape([size, 1])

        torch.cuda.empty_cache()
        # UN-symmetric connectivity distribution
        raw_weights = weights
        # Symmetrization of the connectivity distribution
        weights = (weights + weights.t()) / 2
        raw_weights = raw_weights.to(device)
        weights = weights.to(device)
        return weights, raw_weights

    def clustering(self, visual=True, n_neighbors=30, min_dist=0):

        cpu_embedding = self.gae_nn.embedding.detach().cpu().numpy()

        if visual:
            umap_embedding = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist
            ).fit_transform(cpu_embedding)
            self.plot_classes(umap_embedding)
            if clusterize:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20)
                prediction = clusterer.fit_predict(umap_embedding)
                clusters = np.unique(prediction)
                if -1 in clusters:
                    self.current_cluster_number = len(clusters) - 1
                else:
                    self.current_cluster_number = len(clusters)
                if not eval:
                    tensorboard.add_scalar(CLUSTER_NUMBER_LABEL, self.current_cluster_number, self.global_step)
                self.plot_clustering(prediction, umap_embedding)
        else:
            if clusterize:
                km = KMeans(n_clusters=self.current_cluster_number).fit(cpu_embedding)
                prediction = km.predict(cpu_embedding)

        if clusterize:
            return self.cal_clustering_metric(prediction)
        else:
            return 0,0,0,0,0,0

    def plot_clustering(self, prediction, umap_embedding):
        le = LabelEncoder()
        labels = le.fit_transform(prediction)
        for cluster in le.classes_:
            cluster_points = _safe_indexing(umap_embedding, labels == cluster)
            cluster_marker = markers[cluster % len(markers)]
            cluster_color = colors[cluster % len(colors)]
            cluster_marker_size = sizes[cluster % len(sizes)]
            plt.scatter(cluster_points[:, 0],
                        cluster_points[:, 1],
                        marker=cluster_marker,
                        color=cluster_color,
                        label='Cluster' + str(cluster),
                        s=cluster_marker_size)
        plt.legend()
        if not eval:
            self.send_image_to_tensorboard(plt, UMAP_CLUSTER_PLOT_TAG)
        plt.show()

    def plot_classes(self, umap_embedding):

        class_labels = np.array(ge_class_labels + ['ccres'] * ccre_count)
        classes = np.unique(class_labels)

        classplot_alphas = [0.3, 1, 1, 1, 1]
        classplot_markers = ["o", "^", "v", "v", "^"]
        classplot_colors = ['aquamarine', 'r', 'g', 'g', 'r']
        classplot_sizes = [10, 40, 40, 40, 40]
        for idx, elem_class in enumerate(classes):
            cluster_points = _safe_indexing(umap_embedding, class_labels == elem_class)
            cluster_marker = classplot_markers[idx % len(markers)]
            cluster_color = classplot_colors[idx % len(colors)]
            cluster_marker_size = classplot_sizes[idx % len(sizes)]
            plt.scatter(cluster_points[:, 0],
                        cluster_points[:, 1],
                        marker=cluster_marker,
                        color=cluster_color,
                        label=elem_class,
                        alpha=classplot_alphas[idx],
                        s=cluster_marker_size)
        plt.legend()
        if not eval:
            self.send_image_to_tensorboard(plt, UMAP_CLASS_PLOT_TAG)
        plt.show()

    def send_image_to_tensorboard(self, plt, tag):
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).squeeze(0)
        tensorboard.add_image(tag, image, self.global_step)

    def visual_eval(self, n_neighbors=30, min_dist=0):

        embedding = self.gae_nn.embedding.detach().cpu().numpy()
        km = KMeans(n_clusters=self.current_cluster_number).fit(embedding)
        prediction = km.predict(embedding)
        scores = self.cal_clustering_metric(prediction)
        print('EVAL ge_raw_ch_score: %5.4f, '
              'ccre_raw_ch_score: %5.4f, '
              'mean_heterogeneity_score: %5.4f, '
              'gene_clustering_completeness: %5.4f, '
              'ccre_clustering_completeness: %5.4f,'
              'distance_score: %5.4f,' % (scores[0], scores[1], scores[2], scores[3], scores[4], scores[5]))
        class_label = np.array(['genes'] * ge_count + ['ccres'] * ccre_count)
        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist
        ).fit(embedding)
        umap.plot.points(mapper, width=1500, height=1500, labels=prediction)
        umap.plot.points(mapper, width=1500, height=1500, labels=class_label)
        # primitive_clusters = get_primitive_clusters()
        # umap.plot.points(mapper, width=1500, height=1500, labels=primitive_clusters)

        return mapper, prediction


def save(epoch):
    torch.save(gae.gae_nn.state_dict(), datapath + 'models' + modelname + '_model_' + str(epoch) + '_epochs')
    torch.save(gae.gae_nn.embedding, datapath + 'models' + modelname + '_embedding_' + str(epoch) + '_epochs')


###########
## HYPER-PARAMS
###########
eval=False
pre_trained = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
genes_to_pick = 0
learning_rate = 5 * 10 ** -3
init_genomic_C = 3e5
init_genomic_slope = 0.4

init_cluster_num = 40
max_iter = 20
max_epoch = 20
sparsity_increment = 3
init_sparsity = 5


add_self_loops_genomic = False
add_self_loops_euclidean = False
bounded_sparsity = False
gcn = False
clusterize=False
softmax_reconstruction = True



init_gbf = 0.6
final_gbf = 0
init_repulsive_loss_weight = 1
final_repulsive_loss_weight = 1
init_attractive_loss_weight = 0
final_attractive_loss_weight = 0
init_RQ_loss_weight = 0
final_RQ_loss_weight = 0


differential_attractive_forces = True
init_lambda_attractive = 0.5
final_lambda_attractive = 0.5

differential_repulsive_forces = True
init_lambda_repulsive = 0.5
final_lambda_repulsive = 0.5

diff_RQ = False
init_lambda_rq = 0.5
final_lambda_rq = 0.5

plt.rcParams["figure.figsize"] = (12, 12)
graphical_report_period_epochs = 6




################
##############

link_ds, ccre_ds = load_data(datapath, genes_to_pick, chr_to_filter=[16,19])

X, ge_count, ccre_count = get_hybrid_feature_matrix(link_ds, ccre_ds)

links = get_genomic_distance_matrix(link_ds)

ge_class_labels = ['genes_'+str(ge_cluster_label) for ge_cluster_label in get_primitive_gene_clusters()]

print('Analyzing ',ge_count, ' genes and ',ccre_count, ' ccres for a total of ', ge_count + ccre_count, ' elements.')



# POSITIVE_X
# X += torch.abs(torch.min(X))

X /= torch.max(X)
X = torch.Tensor(X).to(device)
input_dim = X.shape[1]
layers = [input_dim, 24, 12]

if __name__ == '__main__':
    modelname = '/non_diff_RQ'

    tensorboard = SummaryWriter(LOG_DIR + modelname)

    gae = AdaGAE(X,
                 device=device,
                 pre_trained=pre_trained,
                 pre_trained_state_dict='models\\gC_3e5_gS0.4\\gC_3e5_gS0.4_model_140_epochs',
                 pre_computed_embedding='models\\gC_3e5_gS0.4\\gC_3e5_gS0.4_embedding_140_epochs')

    gae.dummy_run()
