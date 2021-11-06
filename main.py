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

reports_path = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\RL_developmental_studies\\Reports\\'

LOG_DIR = 'local_runs/'

##COPY TO NOTEBOOK FROM HERE!!!###


import torch
from torch.utils.tensorboard import SummaryWriter
import cProfile
import pstats
from functools import wraps
import pandas as pd
import numpy as np

from sklearn.metrics import calinski_harabasz_score
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
    # We don't create self loops with 1 (nor with any calue)
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
    see https://www.desmos.com/calculator/frrfbs0tas
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
    if add_self_loops:
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


def load_data(datapath, num_of_genes=0):
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


#####
# ADAGAE OBJECT
########

SPARSITY_LABEL: str = 'Sparsity'
SLOPE_LABEL: str = 'GeneticSlope'


class AdaGAE_NN(torch.nn.Module):

    def __init__(self,
                 data_matrix,
                 device,
                 layers=None,
                 pre_trained=False,
                 pre_trained_state_dict='models/combined_adagae_z12_initk150_150epochs',
                 pre_computed_embedding='models/combined_adagae_z12_initk150_150epochs_embedding'
                 ):
        super(AdaGAE_NN, self).__init__()
        if layers is None: layers = [32, 24, 12]
        self.device = device
        self.embedding_dim = layers[-1]
        self.embedding = None
        self.mid_dim = layers[1]
        self.input_dim = layers[0]
        self.data_matrix = data_matrix.to(device)
        self.W1 = get_weight_initial([self.input_dim, self.mid_dim])
        self.W2 = get_weight_initial([self.mid_dim, self.embedding_dim])
        if pre_trained:
            self.load_state_dict(torch.load(datapath + pre_trained_state_dict))
            self.embedding = torch.load(datapath + pre_computed_embedding)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, norm_adj_matrix):
        embedding = norm_adj_matrix.mm(self.data_matrix.matmul(self.W1))
        embedding = torch.relu(embedding)
        self.embedding = norm_adj_matrix.mm(embedding.matmul(self.W2))
        distances = distance(self.embedding.t(), self.embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)
        return recons_w + 10 ** -10


class AdaGAE():

    def __init__(self, X,
                 layers=None,
                 device=None,
                 pre_trained=False):

        super(AdaGAE, self).__init__()

        self.device = device
        if self.device is None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.X = X
        self.layers = layers
        if bounded_sparsity:
            self.max_sparsity = self.cal_max_neighbors()
        else:
            self.max_sparsity = None
        print('Neighbors will increment up to ', self.max_sparsity)
        self.pre_trained = pre_trained
        self.reset()

    def reset(self):
        self.iteration = 0
        self.gae_nn = None
        torch.cuda.empty_cache()
        self.gae_nn = AdaGAE_NN(self.X, self.device, self.layers, self.pre_trained).to(self.device)
        self.current_sparsity = init_sparsity + 1
        self.current_genomic_slope = init_genomic_slope
        self.init_adj_matrices()
        if not self.pre_trained: self.init_embedding()

    def init_adj_matrices(self):
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
        return 2.0 * size / num_clusters

    def update_graph(self, epoch):
        print('updating A hat with sparsity: ', self.current_sparsity, ' and gs: ', self.current_genomic_slope)
        tensorboard.add_scalar(SPARSITY_LABEL, self.current_sparsity, epoch * max_iter)
        tensorboard.add_scalar(SLOPE_LABEL, self.current_genomic_slope, epoch * max_iter)

        self.adj, self.raw_adj = self.cal_weights_via_CAN(self.gae_nn.embedding.t())
        self.adj = self.adj.detach()
        self.raw_adj = self.raw_adj.detach()
        # threshold = 0.5
        # connections = (recons > threshold).type(torch.IntTensor).cuda()
        # weights = weights * connections
        self.norm_adj = get_normalized_adjacency_matrix(self.adj)
        return self.adj, self.norm_adj, self.raw_adj


    def build_loss(self, recons, weights, raw_weights, global_step):

        size = self.X.shape[0]
        loss = 0
        # notice that recons is actually the q distribution.
        # and that raw_weigths is the p distribution. (before the symmetrization)
        # the following line is the definition of kl divergence
        loss += raw_weights * torch.log(raw_weights / recons + 10 ** -10)
        loss = loss.sum(dim=1)
        # In the paper they mention the minimization of the row-wise kl divergence.
        # here we know we have to compute the mean kl divergence for each point.
        loss = loss.mean()
        tensorboard.add_scalar('KL_divergence', loss.item(), global_step)

        degree = weights.sum(dim=1)
        laplacian = torch.diag(degree) - weights
        # This is exactly equation 11 in the paper. notice that torch.trace return the sum of the elements in the diagonal of the input matrix.
        local_distance_preserving_loss = lam * torch.trace(
            self.gae_nn.embedding.t().matmul(laplacian).matmul(self.gae_nn.embedding)) / size
        tensorboard.add_scalar('LocalDistPreservingPenalty', local_distance_preserving_loss.item(), global_step)

        loss += local_distance_preserving_loss
        tensorboard.add_scalar('Total_Loss', loss.item(), global_step)

        return loss

    def cal_clustering_metric(self, feature_matrix, predicted_labels):
        # silhouette = silhouette_score(feature_matrix, predicted_labels)
        # davies_bouldin = davies_bouldin_score(feature_matrix, predicted_labels)
        # return silhouette, davies_bouldin
        ch_score = calinski_harabasz_score(feature_matrix, predicted_labels)
        ge_ch_raw, ccre_ch_raw = self.get_raw_ch_score(predicted_labels)
        return ch_score, ge_ch_raw, ccre_ch_raw

    def get_raw_ch_score(self, predicted_labels):

        ges = self.X.detach().cpu().numpy()[:ge_count]
        ge_ch = calinski_harabasz_score(ges, predicted_labels[:ge_count])

        ccre_as = self.X.detach().cpu().numpy()[ge_count:]
        ccre_ch = calinski_harabasz_score(ccre_as, predicted_labels[ge_count:])

        return ge_ch, ccre_ch

    def step(self, action):
        self.iteration += 1
        action = action.detach().to('cpu').numpy()
        current_lambda = action[0]
        current_sparsity = int(action[1])
        current_genomic_slope = action[2]
        current_genomic_balance_factor = action[3]

        self.gae_nn.optimizer.zero_grad()

        # recons is the q distribution.
        recons = self.gae_nn(self.norm_adj)
        loss = self.build_loss(recons, self.adj.to(device), self.raw_adj.to(device), self.iteration)

        #self.adj = self.adj.cpu()
        #self.raw_adj = self.raw_adj.cpu()
        torch.cuda.empty_cache()
        loss.backward()
        self.gae_nn.optimizer.step()

        reward = 0
        done_flag = False
        return reward, loss, done_flag


    @profile(output_file='profiling_adagae')
    def dummy_run(self):
        tensorboard.add_scalar(SPARSITY_LABEL, self.current_sparsity, 0)

        self.gae_nn.to(self.device)
        for epoch in tqdm(range(max_epoch)):
            self.epoch_losses = []
            for i in range(max_iter):
                dummy_action = torch.Tensor([lam,
                                             self.current_sparsity,
                                             self.current_genomic_slope,
                                             genetic_balance_factor]).to(self.device)

                reward, loss, done_flag = self.step(dummy_action)
                self.epoch_losses.append(loss.item())

            update = False
            if self.current_genomic_slope > min_genomic_slope:
                self.current_genomic_slope -= genomic_slope_decrement
                update = True
            if (not bounded_sparsity) or (self.current_sparsity < self.max_sparsity):
                self.current_sparsity += sparsity_increment
                update = True
            if update: self.update_graph(epoch + 1)

            if (epoch > 1) and (epoch % 10 == 0): self.clustering()

            if bounded_sparsity and (self.current_sparsity >= self.max_sparsity):
                self.current_sparsity = int(self.max_neighbors)
                break

            mean_loss = sum(self.epoch_losses) / len(self.epoch_losses)
            print('epoch:%3d,' % epoch, 'loss: %6.5f' % mean_loss)

    def cal_weights_via_CAN(self, transposed_data_matrix):
        """
        Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
        See section 3.6 of the paper! (specially equation 20)
        """
        size = transposed_data_matrix.shape[1]

        # We have notice a difference between the distributions of same-class distances.
        # (see the report C:\Users\Jesus Cevallos\odrive\DIAG Drive\RL_developmental_studies\Next Steps.docx)
        if regularized_distance:
            distances = distance(transposed_data_matrix, transposed_data_matrix)
            distances[ge_count:, ge_count:] = distances[ge_count:, ge_count:] / CCRE_dist_reg_factor
        else:
            distances = distance(transposed_data_matrix, transposed_data_matrix)

        distances = torch.max(distances, torch.t(distances))
        sorted_distances, _ = distances.sort(dim=1)
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

        # equation 20 in the paper. notirce that num_neighbors = k.
        weights = torch.div(T, self.current_sparsity * top_k - sum_top_k)
        T = None
        top_k = None
        sum_top_k = None
        torch.cuda.empty_cache()
        # notice that the following line is also part of equation 20
        weights = weights.relu().to(self.device)

        # now at this point, after computing the generative model of the
        # k sparse graph being based on node divergences and similarities,
        # we add weight to some points of the connectivity distribution being based
        # on the explicit graph information.

        # Notice that the link distance matrix has already self loop weight information
        current_link_score = fast_genomic_distance_to_similarity(links, genomic_C, current_genomic_slope)

        if balance_genomic_information:
            # We know that, in the (quasi) simple dist_to_score model, range of link scores go from 0 to 1.
            # We scale the link information to the p distribution.
            current_link_score *= (torch.max(weights).item() * genetic_balance_factor)

        current_link_score = torch.Tensor(current_link_score).to(device)

        weights += current_link_score
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

        embedding = self.gae_nn.embedding.detach().cpu().numpy()
        km = KMeans(n_clusters=num_clusters).fit(embedding)
        prediction = km.predict(embedding)
        ch_score, ge_ch_score_raw, ccre_ch_score_raw = self.cal_clustering_metric(embedding, prediction)
        cpu_embedding = embedding

        print(' k-means --- ch_score: %5.4f, ge_raw_ch_score: %5.4f, ccre_raw_ch_score: %5.4f' % (
            ch_score, ge_ch_score_raw, ccre_ch_score_raw))

        if visual:
            umap_embedding = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist
            ).fit_transform(cpu_embedding)

            '''
            By now, we done care about cluster coloured plots.
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
            plt.show()
            '''

            classes = ['genes', 'ccres']
            class_labels = np.array([classes[0]] * ge_count + [classes[1]] * ccre_count)
            alphas = [1, 0.3]
            for idx, elem_class in enumerate(classes):
                cluster_points = _safe_indexing(umap_embedding, class_labels == elem_class)
                cluster_marker = markers[idx % len(markers)]
                cluster_color = colors[idx % len(colors)]
                cluster_marker_size = sizes[idx % len(sizes)]
                plt.scatter(cluster_points[:, 0],
                            cluster_points[:, 1],
                            marker=cluster_marker,
                            color=cluster_color,
                            label=elem_class,
                            alpha=alphas[idx],
                            s=cluster_marker_size)
            plt.legend()
            plt.show()

    def visual_eval(self, n_neighbors=30, min_dist=0):

        embedding = self.gae_nn.embedding.detach().cpu().numpy()
        km = KMeans(n_clusters=num_clusters).fit(embedding)
        prediction = km.predict(embedding)
        ch_score, ge_ch_score_raw, ccre_ch_score_raw = self.cal_clustering_metric(embedding, prediction)
        print('EVAL ch_score: %5.4f, ge_raw_ch_score: %5.4f, ccre_raw_ch_score: %5.4f' % (
            ch_score, ge_ch_score_raw, ccre_ch_score_raw))
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


###########
## HYPER-PARAMS
###########

genomic_C = 1e5
genes_to_pick = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_iter = 50
max_epoch = 100
sparsity_increment = 5
learning_rate = 5 * 10 ** -3
init_sparsity = 150
init_genomic_slope = current_genomic_slope = 0.2
genomic_slope_decrement = 0
num_clusters = 20
lam = 4.0
add_self_loops = False
balance_genomic_information = False
genetic_balance_factor = None
min_genomic_slope = 0.05
bounded_sparsity = False
regularized_distance = False
CCRE_dist_reg_factor = 10.5

if __name__ == '__main__':
    tensorboard = SummaryWriter(LOG_DIR + '/lambda_8_gbf_0.8_dynamicGS')

    link_ds, ccre_ds = load_data(datapath, genes_to_pick)

    X, ge_count, ccre_count = get_hybrid_feature_matrix(link_ds, ccre_ds)

    links = get_genomic_distance_matrix(link_ds)

    X /= torch.max(X)
    X = torch.Tensor(X).to(device)
    input_dim = X.shape[1]
    layers = [input_dim, 24, 12]

    gae = AdaGAE(X,
                 layers=layers,
                 device=device,
                 pre_trained=False)
    gae.dummy_run()

    # tensorboard.close()
