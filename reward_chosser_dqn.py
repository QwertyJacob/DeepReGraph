from tqdm import tqdm as tqdm

datapath = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\GE_Datasets\\'
models_path = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\GE_Datasets\\models\\'
reports_path = 'C:\\Users\\Jesus Cevallos\\odrive\\DIAG Drive\\RL_developmental_studies\\Reports\\'
LOG_DIR = 'dqn_local_runs/'

##
## COPY TO NOTEBOOK FROM HERE!!
##
import os
import torch
import pandas as pd
import numpy as np
from sklearn.utils import _safe_indexing
import random
import math
from torch.utils.tensorboard import SummaryWriter

REWARD_LABEL = 'REWARD'
GENE_DISPERSION_DELTA = 'GENE_DISPERSION_DELTA'
CCRE_DISPERION_DELTA = 'CCRE_DISPERION_DELTA'
HETEROGENEITY_DELTA = 'HETEROGENEITY_DELTA'
DISTANCE_SCORE_DELTA = 'DISTANCE_SCORE_DELTA'
WHOLE_GENE_DISPERSION_DELTA = 'WHOLE_GENE_DISPERSION_DELTA'
WHOLE_CCRE_DISPERION_DELTA = 'WHOLE_CCRE_DISPERION_DELTA'
WHOLE_HETEROGENEITY_DELTA = 'WHOLE_HETEROGENEITY_DELTA'
WHOLE_DISTANCE_SCORE_DELTA = 'WHOLE_DISTANCE_SCORE_DELTA'
WHOLE_GENE_DISPERSION = 'WHOLE_GENE_DISPERSION'
WHOLE_CCRE_DISPERION = 'WHOLE_CCRE_DISPERION'
WHOLE_HETEROGENEITY = 'WHOLE_HETEROGENEITY'
WHOLE_DISTANCE_SCORE = 'WHOLE_DISTANCE_SCORE'
EPSILON_EGREEDY = 'EPSILON_EGREEDY'


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay)
    tensorboard.add_scalar(EPSILON_EGREEDY, epsilon, steps_done)
    return epsilon


def save_model(model, filename):
    torch.save(model.state_dict(), models_path + filename)


def load_model(filename):
    return torch.load(models_path + filename)


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


def get_hybrid_feature_matrix(link_ds, ccre_ds):
    ge_values = link_ds.reset_index().drop_duplicates('EnsembleID')[['Heart_E10_5', 'Heart_E11_5', 'Heart_E12_5',
                                                                     'Heart_E13_5', 'Heart_E14_5', 'Heart_E15_5',
                                                                     'Heart_E16_5', 'Heart_P0']].values
    ge_count = ge_values.shape[0]
    ge_values_new = np.zeros((ge_values.shape[0], max_clusters))
    ge_values_new[:, 0:8] = ge_values

    ccre_activity = ccre_ds.set_index('cCRE_ID').values
    ccre_count = ccre_activity.shape[0]
    ccre_activity_new = np.zeros((ccre_activity.shape[0], max_clusters))
    ccre_activity_new[:, max_clusters - 24:max_clusters] = ccre_activity
    return torch.Tensor(np.concatenate((ge_values_new, ccre_activity_new))).cpu(), ge_count, ccre_count


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


def fast_genomic_distance_to_similarity(link_matrix, c, d):
    '''
    see https://www.desmos.com/calculator/frrfbs0tas
    TODO play aroun'
    '''
    return 1 / (((link_matrix / c) ** (10 * d)) + 1)


def clustering_to_onehot(clustering_array):
    clustering_array = clustering_array.reshape(element_count, 1)
    onehot_clustering = np.zeros((element_count, max_clusters))
    np.put_along_axis(onehot_clustering, clustering_array, 1, 1)
    return onehot_clustering


def one_hot_to_clustering(onehot_c):
    proto_cluster_vector = np.where(onehot_c)[1]
    return proto_cluster_vector - 1


def get_current_intra_cluster_dispersion(single_action, some_clustering):
    '''
    Returns a measure of the dispersion within a cluster.
    Params:
    single_action: The cluster for which to compute the dispersion.

    '''
    indexes_to_take = some_clustering == single_action.item()
    cluster_elements = numpy_X[indexes_to_take]
    cluster_size = cluster_elements.shape[0]
    current_centroid = np.mean(cluster_elements, axis=0)
    dispersions = (numpy_X[indexes_to_take] - current_centroid) ** 2
    return np.sum(dispersions) / cluster_size


def get_initial_state(some_clustering):
    cluster_labels = np.unique(some_clustering)
    centroids = np.zeros((max_clusters, max_clusters))
    gene_mean_dispersions = np.zeros((max_clusters, 1))
    ccre_mean_dispersions = np.zeros((max_clusters, 1))
    cluster_heterogeneities = np.zeros((max_clusters, 1))
    distance_scores = np.zeros((max_clusters, 1))
    gene_completeness = 1
    ccre_completeness = 1
    gene_cluster_sizes = np.zeros((max_clusters, 1))
    ccre_cluster_sizes = np.zeros((max_clusters, 1))

    if -1 in cluster_labels:
        cluster_labels = cluster_labels[1:]
        gene_labels = some_clustering[:ge_count]
        scattered_gene_indexes = np.where(gene_labels == -1)
        scattered_gene_count = len(scattered_gene_indexes[0])
        gene_completeness = 1 - scattered_gene_count / ge_count

        ccre_labels = some_clustering[ge_count:]
        scattered_ccre_indexes = np.where(ccre_labels == -1)
        scattered_ccre_count = len(scattered_ccre_indexes[0])
        ccre_completeness = 1 - scattered_ccre_count / ccre_count

    for k in cluster_labels:
        cluster_k_components = numpy_X[some_clustering == k]
        centroid_k = np.mean(cluster_k_components, axis=0)
        centroids[k] = centroid_k
        dispersion_vectors = (cluster_k_components - centroid_k) ** 2

        gene_dispersions = np.sum(dispersion_vectors[:, 0:8], axis=1)
        gene_diameter = gene_dispersions.max()
        scaled_gene_dispersions = gene_dispersions / gene_diameter
        gene_mean_dispersions[k] = scaled_gene_dispersions.mean()

        ccre_dispersions = np.sum(dispersion_vectors[:, max_clusters - 24:], axis=1)
        ccre_diameter = ccre_dispersions.max()
        scaled_ccre_dispersions = ccre_dispersions / ccre_diameter
        ccre_mean_dispersions[k] = scaled_ccre_dispersions.mean()

        cluster_entity_array = _safe_indexing(class_label_array, some_clustering == k)
        cluster_dimension = len(cluster_entity_array)
        cluster_gene_count = np.count_nonzero(cluster_entity_array == 'gene')
        cluster_ccre_count = np.count_nonzero(cluster_entity_array == 'ccre')
        absolute_omogeneity = abs(cluster_gene_count - (cluster_ccre_count / 2))
        # omogeneity distribution tends to be more long tailed as the cluster dimension grows.
        # so we normalize omogeneity between clusters:
        relative_omogeneity = absolute_omogeneity / (cluster_dimension ** 0.6)
        current_heterogeneity = 1 / (1 + relative_omogeneity)
        cluster_heterogeneities[k] = current_heterogeneity

        gene_cluster_mask = (some_clustering == k)[:ge_count]
        ccre_cluster_mask = (some_clustering == k)[ge_count:]

        current_distance_score_matrix = distance_score_matrix[gene_cluster_mask, :]
        current_distance_score_matrix = current_distance_score_matrix[:, ccre_cluster_mask]
        # Notice we dont have to normalize this score with the dimension of the cluster,
        # because the mean function is already doing it.
        distance_score = current_distance_score_matrix.mean()
        distance_scores[k] = distance_score

        gene_cluster_sizes[k] = cluster_gene_count / max_conventional_cluster_shape
        ccre_cluster_sizes[k] = cluster_ccre_count / max_conventional_cluster_shape

    return cluster_labels, \
           centroids, \
           gene_mean_dispersions, \
           ccre_mean_dispersions, \
           cluster_heterogeneities, \
           distance_scores, \
           gene_cluster_sizes, \
           ccre_cluster_sizes, \
           gene_completeness, \
           ccre_completeness


def get_state(some_clustering, some_action):
    cluster_labels = np.unique(some_clustering)
    centroids = np.zeros((max_clusters, max_clusters))
    gene_mean_dispersions = np.zeros((max_clusters, 1))
    ccre_mean_dispersions = np.zeros((max_clusters, 1))
    cluster_heterogeneities = np.zeros((max_clusters, 1))
    distance_scores = np.zeros((max_clusters, 1))
    gene_completeness = 1
    ccre_completeness = 1

    if -1 in cluster_labels:
        cluster_labels = cluster_labels[1:]
        gene_labels = some_clustering[:ge_count]
        scattered_gene_indexes = np.where(gene_labels == -1)
        scattered_gene_count = len(scattered_gene_indexes[0])
        gene_completeness = 1 - scattered_gene_count / ge_count

        ccre_labels = some_clustering[ge_count:]
        scattered_ccre_indexes = np.where(ccre_labels == -1)
        scattered_ccre_count = len(scattered_ccre_indexes[0])
        ccre_completeness = 1 - scattered_ccre_count / ccre_count

    for k in cluster_labels:
        cluster_k_components = numpy_X[some_clustering == k]
        centroid_k = np.mean(cluster_k_components, axis=0)
        centroids[k] = centroid_k
        dispersion_vectors = (cluster_k_components - centroid_k) ** 2

        gene_dispersions = np.sum(dispersion_vectors[:, 0:8], axis=1)
        gene_diameter = gene_dispersions.max()
        scaled_gene_dispersions = gene_dispersions / gene_diameter
        gene_mean_dispersions[k] = scaled_gene_dispersions.mean()

        ccre_dispersions = np.sum(dispersion_vectors[:, max_clusters - 24:], axis=1)
        ccre_diameter = ccre_dispersions.max()
        scaled_ccre_dispersions = ccre_dispersions / ccre_diameter
        ccre_mean_dispersions[k] = scaled_ccre_dispersions.mean()

        cluster_entity_array = _safe_indexing(class_label_array, some_clustering == k)
        cluster_dimension = len(cluster_entity_array)
        cluster_gene_count = np.count_nonzero(cluster_entity_array == 'gene')
        cluster_ccre_count = np.count_nonzero(cluster_entity_array == 'ccre')
        absolute_omogeneity = abs(cluster_gene_count - (cluster_ccre_count / 2))
        # omogeneity distribution tends to be more long tailed as the cluster dimension grows.
        # so we normalize omogeneity between clusters:
        relative_omogeneity = absolute_omogeneity / (cluster_dimension ** 0.6)
        current_heterogeneity = 1 / (1 + relative_omogeneity)
        cluster_heterogeneities[k] = current_heterogeneity

        gene_cluster_mask = (some_clustering == k)[:ge_count]
        ccre_cluster_mask = (some_clustering == k)[ge_count:]

        current_distance_score_matrix = distance_score_matrix[gene_cluster_mask, :]
        current_distance_score_matrix = current_distance_score_matrix[:, ccre_cluster_mask]
        # Notice we dont have to normalize this score with the dimension of the cluster,
        # because the mean function is already doing it.
        distance_score = current_distance_score_matrix.mean()
        distance_scores[k] = distance_score


    return centroids, \
           gene_mean_dispersions, \
           ccre_mean_dispersions, \
           cluster_heterogeneities, \
           distance_scores, \
           gene_completeness, \
           ccre_completeness

def update_completeness():
    global gene_completeness
    global ccre_completeness

    if -1 in cluster_labels:
        real_cluster_labels = cluster_labels[1:]
        gene_labels = some_clustering[:ge_count]
        scattered_gene_count = np.count_nonzero(gene_labels == -1)
        gene_completeness = 1 - scattered_gene_count / ge_count

        ccre_labels = some_clustering[ge_count:]
        scattered_ccre_count = np.count_nonzero(ccre_labels == -1)
        ccre_completeness = 1 - scattered_ccre_count / ccre_count


def update_state_variables(decoded_action, step):
    global cluster_labels
    global centroids
    global gene_mean_dispersions
    global ccre_mean_dispersions
    global cluster_heterogeneities
    global gene_cluster_sizes
    global ccre_cluster_sizes
    global distance_scores

    if decoded_action not in cluster_labels:
        cluster_labels = np.sort(np.concatenate((cluster_labels, np.array([decoded_action]))))

    if -1 in cluster_labels:
        real_cluster_labels = cluster_labels[1:]

    if decoded_action != -1:
        mask = some_clustering == decoded_action
        cluster_k_components = numpy_X[mask]
        centroid_k = np.mean(cluster_k_components, axis=0)
        centroids[decoded_action] = centroid_k

        dispersion_vectors = (cluster_k_components - centroid_k) ** 2

        gene_dispersions = np.sum(dispersion_vectors[:, 0:8], axis=1)
        gene_diameter = gene_dispersions.max()
        scaled_gene_dispersions = gene_dispersions / gene_diameter
        gene_mean_dispersions[decoded_action] = scaled_gene_dispersions.mean()
        ccre_dispersions = np.sum(dispersion_vectors[:, max_clusters - 24:], axis=1)
        ccre_diameter = ccre_dispersions.max()
        scaled_ccre_dispersions = ccre_dispersions / ccre_diameter
        ccre_mean_dispersions[decoded_action] = scaled_ccre_dispersions.mean()

        if (step % HETEROGENEITY_SCORE_UPDATE == 0):
            cluster_entity_array = _safe_indexing(class_label_array, mask)
            _, elems_per_class = np.unique(cluster_entity_array, return_counts=True)
            cluster_dimension = cluster_k_components.shape[0]
            cluster_gene_count = np.count_nonzero(cluster_entity_array == 'gene')
            cluster_ccre_count = np.count_nonzero(cluster_entity_array == 'ccre')
            absolute_omogeneity = abs(cluster_gene_count - (cluster_dimension / 2))
            # omogeneity distribution tends to be more long tailed as the cluster dimension grows.
            # so we normalize omogeneity between clusters:
            relative_omogeneity = absolute_omogeneity / (cluster_dimension ** 0.6)
            current_heterogeneity = 1 / (1 + relative_omogeneity)
            cluster_heterogeneities[decoded_action] = current_heterogeneity

            gene_cluster_sizes[decoded_action] = 5 * cluster_gene_count / max_conventional_cluster_shape
            ccre_cluster_sizes[decoded_action] = cluster_ccre_count / max_conventional_cluster_shape

        # We update the distance score not very often, as it is a time-consuming step:
        if (step % DISTANCE_SCORE_UPDATE_INTERVAL == 0):
            gene_cluster_mask = mask[:ge_count]
            ccre_cluster_mask = mask[ge_count:]

            current_distance_score_matrix = distance_score_matrix[gene_cluster_mask, :]
            current_distance_score_matrix = current_distance_score_matrix[:, ccre_cluster_mask]

            # Notice we dont have to normalize this score with the dimension of the cluster,
            # because the mean function is already doing it.
            distance_score = current_distance_score_matrix.mean()
            distance_scores[decoded_action] = distance_score


class Q_Clust(torch.nn.Module):

    def __init__(self, device):
        super(Q_Clust, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4), stride=(1, 2))
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 4), stride=(1, 2))
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 4), stride=(1, 1))
        self.linear_advantage_1 = torch.nn.Linear(in_features=323, out_features=128)
        self.linear_advantage_2 = torch.nn.Linear(in_features=128, out_features=max_clusters)
        self.linear_state_value_1 = torch.nn.Linear(in_features=323, out_features=128)
        self.linear_state_value_2 = torch.nn.Linear(in_features=128, out_features=1)
        self.activation = torch.relu
        self.to(device)

    def forward(self, state_tensor, state_linear_tensor):
        transform = self.conv1(state_tensor)
        transform = self.activation(transform)
        transform = self.conv2(transform)
        transform = self.activation(transform)
        transform = self.conv3(transform)

        mid_term_vector = torch.flatten(transform)
        mid_term_vector = mid_term_vector.reshape(-1, 280)
        state_linear_tensor = state_linear_tensor.reshape(-1, 43)
        mid_term_vector = torch.cat((mid_term_vector, state_linear_tensor), dim=1)

        action_advantages = self.linear_advantage_1(mid_term_vector)
        action_advantages = self.activation(action_advantages)
        action_advantages = self.linear_advantage_2(action_advantages)

        state_value = self.linear_state_value_1(mid_term_vector)
        state_value = self.activation(state_value)
        state_value = self.linear_state_value_2(state_value)

        action_state_values = state_value + action_advantages - action_advantages.mean()
        output = torch.softmax(action_state_values, dim=1)
        return output


class ExperienceReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state_tensor, linear_state_tensor, action, new_state_tensor,
             new_linear_state_tensor, reward, done_signal):

        transition = (state_tensor, linear_state_tensor, action, new_state_tensor,
                      new_linear_state_tensor, reward, done_signal)

        if self.position >= len(self.memory):
            self.memory.append(transition)

        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


def select_action(curr_state_tensor, linear_state_tensor, epsilon=None):
    if eval:
        action_from_nn = q_clust(curr_state_tensor, linear_state_tensor)
        action = torch.max(action_from_nn, 1)[1]
        action = action.item()

    else:
        random_for_egreedy = torch.rand(1)[0]
        if random_for_egreedy > epsilon:

            with torch.no_grad():
                action_from_nn = q_clust(curr_state_tensor, linear_state_tensor)
                action = torch.max(action_from_nn, 1)[1]
                action = action.item()
        else:
            action = random.randint(0, max_clusters - 1)

    return action


def optimize(steppie):
    if (len(memory) < batch_size):
        return

    states_sampled, linear_states, actions, new_states_sampled, \
    new_linear_states, rewards, done_signals = memory.sample(batch_size)

    net_states = torch.cat([state for state in states_sampled]).to(device)

    new_states = torch.cat([new_state for new_state in new_states_sampled]).to(device)

    linear_states = torch.cat([lin_s for lin_s in linear_states]).to(device)

    new_lin_states = torch.cat([new_lin_s for new_lin_s in new_linear_states]).to(device)

    rewards = torch.Tensor(np.asarray(rewards)).to(device)

    actions = torch.LongTensor(np.asarray(actions)).unsqueeze(1).to(device)

    done_signals = torch.Tensor(done_signals).unsqueeze(1).to(device)

    if double_dqn:

        new_state_indexes = q_clust(new_states, new_lin_states).detach()

        max_new_state_indexes = torch.max(new_state_indexes, 1)[1]

        new_state_values = target_q_clust(new_states, new_lin_states).detach()

        max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)

    else:

        new_state_values = target_q_clust(new_states, new_lin_states).detach()

        max_new_state_values = torch.max(new_state_values, 1)[0]

    target_values = rewards.unsqueeze(1) + (1 - done_signals) * (gamma * max_new_state_values)

    predicted_values = q_clust(net_states, linear_states).gather(1, actions).squeeze(1)

    loss = loss_func(predicted_values, target_values)

    optimizer.zero_grad()

    loss.backward()

    if clip_error:

        for param in q_clust.parameters():
            param.grad.data.clamp_(-1, 1)

    optimizer.step()

    if steppie % update_target_frequency == 0:
        target_q_clust.load_state_dict(q_clust.state_dict())

    if steppie % save_model_frequency == 0:
        save_model(q_clust, model_name)


########
#########

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ELEM_CLASSES = ['gene', 'ccre']

add_self_loops = False
genomic_C = 1e4
genes_to_pick = 0
genomic_slope = 1
genetic_balance_factor = 0
max_clusters = 40

link_ds, ccre_ds = load_data(datapath, genes_to_pick)
X, ge_count, ccre_count = get_hybrid_feature_matrix(link_ds, ccre_ds)

X /= torch.max(X)
element_count = ge_count + ccre_count
X = torch.Tensor(X).to(device)
numpy_X = X.detach().cpu().numpy()

links = get_genomic_distance_matrix(link_ds)
link_scores = fast_genomic_distance_to_similarity(links, genomic_C, genomic_slope)
class_label_array = np.array([ELEM_CLASSES[0]] * ge_count + [ELEM_CLASSES[1]] * ccre_count)
distance_score_matrix = link_scores[:ge_count, ge_count:]

max_conventional_cluster_shape = element_count / max_clusters

with open(datapath + 'some_prediction.npy', 'rb') as f:
    init_clustering = np.load(f)
    init_clustering -= 1

####
####


resume_previous_training = False
eval = False
model_name = 'DQNClusterer'
q_clust = Q_Clust(device)
target_q_clust = Q_Clust(device)
loss_func = torch.nn.MSELoss()

# hyper-params:
training_episodes = 20
double_dqn = True
clip_error = True

learning_rate = 3e-4
batch_size = 64
gamma = 0.99
update_target_frequency = 10000
save_model_frequency = 1000000
memory = ExperienceReplay(1000000)
egreedy_final = 1e-2
egreedy = .99
egreedy_decay = 2.5e4

HETEROGENEITY_SCORE_UPDATE = 1
DISTANCE_SCORE_UPDATE_INTERVAL = HETEROGENEITY_SCORE_UPDATE * 2
SPARSE_REWARD_INTERVAL = DISTANCE_SCORE_UPDATE_INTERVAL * 2
VERY_SPARSE_REWARD_INTERVAL = SPARSE_REWARD_INTERVAL * 2

optimizer = torch.optim.Adam(params=q_clust.parameters(), lr=learning_rate)

if resume_previous_training:
    if os.path.exists(models_path + '/' + model_name):
        print("Loading previously saved model ... ")

        q_clust.load_state_dict(load_model(model_name))
    else:
        print("path not found! ... ", models_path + '/' + model_name)

if eval:
    print('putting model in eval mode')
    q_clust.eval()

tensorboard = SummaryWriter(LOG_DIR + '/third')

for episode in range(training_episodes):

    some_clustering = init_clustering.copy()

    # state variables:
    initial_state = get_initial_state(init_clustering)
    cluster_labels = initial_state[0]
    centroids = initial_state[1]
    gene_mean_dispersions = initial_state[2]
    ccre_mean_dispersions = initial_state[3]
    cluster_heterogeneities = initial_state[4]
    distance_scores = initial_state[5]
    gene_cluster_sizes = initial_state[6]
    ccre_cluster_sizes = initial_state[7]
    gene_completeness = initial_state[8]
    ccre_completeness = initial_state[9]


    for step in tqdm(range(element_count)):

        # Taking into account the current situation, we compute the next possible reward for each action
        # and such possible greedy reward values compose is the new state vector

        previous_gene_dispersions = np.nan_to_num(gene_mean_dispersions).copy()
        previous_ccre_dispersions = np.nan_to_num(ccre_mean_dispersions).copy()

        if step % HETEROGENEITY_SCORE_UPDATE == 0:
            previous_heterogeneities = np.nan_to_num(cluster_heterogeneities).copy()
        if step % DISTANCE_SCORE_UPDATE_INTERVAL == 0:
            previous_distance_scores = np.nan_to_num(distance_scores).copy()
        if step % SPARSE_REWARD_INTERVAL == 0:
            previous_whole_gene_dispersion = np.mean(np.nan_to_num(gene_mean_dispersions)).copy()
            previous_whole_ccre_dispersion = np.mean(np.nan_to_num(ccre_mean_dispersions)).copy()
            previous_whole_heterogeneity = np.mean(np.nan_to_num(cluster_heterogeneities)).copy()
            previous_whole_distance_score = np.mean(np.nan_to_num(distance_scores)).copy()

        for possible_action in range(0, max_clusters):
            proto_clustering = some_clustering.copy()
            proto_clustering[step] = possible_action
            get_state(proto_clustering, possible_action)

        state_matrix = np.concatenate((centroids,
                                       gene_mean_dispersions,
                                       ccre_mean_dispersions,
                                       cluster_heterogeneities,
                                       distance_scores,
                                       gene_cluster_sizes,
                                       ccre_cluster_sizes
                                       ), axis=1)

        linear_state_suffix = torch.Tensor([step / element_count])
        elem_to_clusterize = X[step]
        linear_state = torch.cat((X[step], linear_state_suffix))
        state = torch.Tensor(state_matrix).unsqueeze(0).unsqueeze(0)

        prev_linear_state = linear_state.cpu().clone()
        prev_state = state.cpu().clone()

        epsilon = calculate_epsilon(step + (episode * element_count))
        decoded_action = select_action(state.to(device), linear_state.to(device), epsilon)



        #We now assign the effective reward beign based in the action:


        # Updating state:
        decoded_action -= 1
        some_clustering[step] = decoded_action
        update_state_variables(decoded_action, step)
        if (some_clustering[step] == -1 and decoded_action != -1) or (
                decoded_action == -1 and some_clustering[step] != -1):
            update_completeness()

        # Computing reward:
        reward = 0.0

        if decoded_action != -1:

            gene_dispersion_delta = 1e4 * (
                        previous_gene_dispersion - np.nan_to_num(gene_mean_dispersions)[decoded_action])
            ccre_dispersion_delta = 1e4 * (
                        previous_ccre_dispersion - np.nan_to_num(ccre_mean_dispersions)[decoded_action])

            tensorboard.add_scalar(GENE_DISPERSION_DELTA, gene_dispersion_delta, step + (episode * element_count))
            tensorboard.add_scalar(CCRE_DISPERION_DELTA, ccre_dispersion_delta, step + (episode * element_count))

            reward = gene_dispersion_delta + ccre_dispersion_delta

            if step % HETEROGENEITY_SCORE_UPDATE == 0:
                heterogeneity_delta = 1e2 * (
                            np.nan_to_num(cluster_heterogeneities)[decoded_action] - previous_heterogeneity)
                tensorboard.add_scalar(HETEROGENEITY_DELTA, heterogeneity_delta, step + (episode * element_count))
                reward += heterogeneity_delta

            if step % DISTANCE_SCORE_UPDATE_INTERVAL == 0:
                distance_score_delta = 1e5 * (np.nan_to_num(distance_scores)[decoded_action] - previous_distance_score)
                tensorboard.add_scalar(DISTANCE_SCORE_DELTA, distance_score_delta, step + (episode * element_count))
                reward += distance_score_delta

            if step % SPARSE_REWARD_INTERVAL == 0:
                whole_gene_disp_delta = (
                            previous_whole_gene_dispersion - (np.mean(np.nan_to_num(gene_mean_dispersions))))
                whole_ccre_disp_delta = (
                            previous_whole_ccre_dispersion - (np.mean(np.nan_to_num(ccre_mean_dispersions))))
                whole_heterogeneity_delta = (
                            (np.mean(np.nan_to_num(cluster_heterogeneities))) - previous_whole_heterogeneity)
                whole_distance_score_delta = ((np.mean(np.nan_to_num(distance_scores))) - previous_whole_distance_score)

                tensorboard.add_scalar(WHOLE_GENE_DISPERSION_DELTA, whole_gene_disp_delta,
                                       step + (episode * element_count))
                tensorboard.add_scalar(WHOLE_CCRE_DISPERION_DELTA, whole_ccre_disp_delta,
                                       step + (episode * element_count))
                tensorboard.add_scalar(WHOLE_HETEROGENEITY_DELTA, whole_heterogeneity_delta,
                                       step + (episode * element_count))
                tensorboard.add_scalar(WHOLE_DISTANCE_SCORE_DELTA, whole_distance_score_delta,
                                       step + (episode * element_count))

                reward += whole_gene_disp_delta + \
                          whole_ccre_disp_delta + \
                          whole_heterogeneity_delta + \
                          whole_distance_score_delta


            if step % VERY_SPARSE_REWARD_INTERVAL == 0:

                whole_gene_dispersions = 1 / 10 * (np.mean(np.nan_to_num(gene_mean_dispersions)))
                whole_ccre_dipersions = 1 / 10 * (np.mean(np.nan_to_num(ccre_mean_dispersions)))
                whole_heterogeneities = 5 * (np.mean(np.nan_to_num(cluster_heterogeneities)))
                whole_distance_scores = 500 * (np.mean(np.nan_to_num(distance_scores)))

                tensorboard.add_scalar(WHOLE_GENE_DISPERSION, whole_gene_dispersions,
                                       step + (episode * element_count))
                tensorboard.add_scalar(WHOLE_CCRE_DISPERION, whole_ccre_dipersions,
                                       step + (episode * element_count))
                tensorboard.add_scalar(WHOLE_HETEROGENEITY, whole_heterogeneities,
                                       step + (episode * element_count))
                tensorboard.add_scalar(WHOLE_DISTANCE_SCORE, whole_distance_scores,
                                       step + (episode * element_count))

                reward += whole_gene_dispersions + \
                          whole_ccre_dipersions + \
                          whole_heterogeneities + \
                          whole_distance_scores

            reward = reward[0]

        tensorboard.add_scalar(REWARD_LABEL, reward, step + (episode * element_count))

        # Computing done_signal
        if (step != element_count - 1):
            done_signal = False
        else:
            done_signal = True

        # Storing transition:
        decoded_action += 1
        memory.push(prev_state, prev_linear_state, decoded_action, state.cpu(), linear_state.cpu(), reward, done_signal)

        optimize(step)
