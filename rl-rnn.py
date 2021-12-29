
'''
The following code was obtained from https://github.com/mnqu/DRL/
were an implementation of the paper "Curriculum Learning for Heterogeneous Star Network Embedding via Deep Reinforcement Learning"
in https://dl.acm.org/doi/10.1145/3159652.3159711 is proposed.
Comments were added by Jesus Cevallos and refer to such a paper.
'''


import sys
import os
import random
import struct
import pylinelib as linelib
import mctslib
import rnnlib



def search_from_pool(mcnode):
	'''

	Args:
		mcnode: a Node from the tree

	Returns: if mcnode is in the pool array of tree nodes, it will return the index of mcnode in such an array,
	otherwise it will return -1

	'''
	pointer = mcnode._id
	pst = -1

	for k in range(pool_size):
		if pointer == pointers[k]:
			pst = k
			break
	return pst


def add_to_pool(mcnode):
	'''
	Adds a new Tree node to the current node pool array, unless it already is in the current pool array.
	Args:
		mcnode:

	Returns: the index of the node in the pool array if the adding process was succesfull (already added or available space founded)
	and -1 otherwise.

	'''

	pointer = mcnode._id

	pst = search_from_pool(mcnode)

	if pst != -1:
		return pst

	ok = -1

	for pst in range(pool_size):

		if pointers[pst] == -1:
			pointers[pst] = pointer
			linelib.save_to(node_pools[pst])
			linelib.save_to(cont_pools[pst])
			ok = pst
			break

	return ok


def load_from_pool(mcnode):
	'''

	Args:
		mcnode:

	Returns: the index of the node in the pool array if such a node is in the node pool. and -1 otherwise.

	'''
	pointer = mcnode._id
	pst = search_from_pool(mcnode)
	if pst != -1:
		linelib.load_from(node_pools[pst])
		linelib.load_from(cont_pools[pst])
	return pst


def delete_from_pool(pst):
	'''
	Deletes the node whose index is pst from the pool array.
	Args:
		pst:

	Returns:

	'''
	if pst >= 0 and pst < pool_size:
		pointers[pst] = -1


def run(curiter, act):
	linelib.run_trainer_line(trainers[act], samples, negative, alpha, threads)
	print('\rIter:', curiter, 'Type:', act, 'Training DONE!')


def calculate_priors(state):
	'''

	Args:
		state:

	Returns: The Q values estimated for each action according to the learning module

	'''
	priors = [0 for k in range(type_size)]

	for action in range(type_size):
		score = rnn.predict(state, action)
		priors[action] = score

	print(priors)

	return priors


def calculate_best(state):
	'''

	Args:
		state:

	Returns: The action to which the higher Q value corresponds being at state "state" according to the learning module

	'''
	bestid, bestvl = -1, -100000000.0

	for action in range(type_size):

		score = rnn.predict(state, action)
		if score > bestvl:
			bestid = action
			bestvl = score

	return bestid


#Data loading.

cont_file = '../data_dblp/node0.txt'
node_file = '../data_dblp/node1.txt'
net_file = '../data_dblp/hinet.txt'
train_file = '../data_dblp/train.lb'
test_file = '../data_dblp/test.lb'
output_file = 'vec.emb'


# Hyper params.

#vector_size is the node embedding dimension
vector_size = 100

# number of negative samples per positive sample for a run of the LINE algorithm
negative = 5

# Number of link samples taken at each run of the LINE algorithm
samples = 1000000


threads = 20

# alpha is the learning rate for the LINE embedding algorithm
alpha = 0.015

# type_size is the number of available actions.
type_size = 3

depth = 0

tree_size = 7

pool_size = 12

# lamb balances exploration and exploitation
lamb = 0.5

binary = 1

# the penalty given for each action different from "end training"
penalty = 0.0005

hist_length = 5



node = linelib.add_node(node_file, vector_size)
cont = linelib.add_node(cont_file, vector_size)
hin = linelib.add_hin(net_file, cont, node, 1)
trainers = [linelib.add_trainer_line(hin, k) for k in range(type_size)]
classifier = linelib.add_node_classifier(node, train_file, test_file)

node_pools = [linelib.add_emb_backup(node) for k in range(pool_size)]
cont_pools = [linelib.add_emb_backup(cont) for k in range(pool_size)]



#pointers is an array containing the ids of the available Tree nodes
pointers = [-1 for k in range(pool_size)]

# Tree creation
mctree = mctslib.Tree()
mctree.set_lambda(lamb)
mctree.set_action_size(type_size)

# Learning module creation
rnn_dims = 10
rnn = rnnlib.RNNRegression('lstm', type_size, rnn_dims, rnn_dims)


print('Training process:')

# set priors for the root node and add some random variance
mctree.get_root().add_prior([random.random() / 100000 for k in range(type_size)])

add_to_pool(mctree.get_root())

selected_act_seq = [type_size]

stage = 0

while True:

	# init training samples of rnn
	current_sample_pool = []

	# add nodes
	for T in range(tree_size):

		if mctree.get_size() >= tree_size:
			break

		print('-- Stage:', stage, 'Simu:', T)

		# prioritized selection of the leaf node
		# (Tree traversal following the hybrid offline/online version of UCT algorithm based o eq (4) in the paper)
		mcnode = mctree.select()


		# load embeddings
		prev_mcnode = mcnode

		# hist_act will contain the actions that have not been performed yet.
		hist_act = []

		# load embeddings from the latest node in the pool
		while prev_mcnode != None:

			if load_from_pool(prev_mcnode) != -1:
				# The node has been found in the pool of nodes, and its embedding has been loaded,
				# we break the loop and go ahoead.
				break

			# The node has not been found in the pool, so we registrate the action related to such
			# a node to perform then later,
			hist_act.insert(0, prev_mcnode.get_last_action())

			# we go 1 step backwards on the traversal and repeat the embedding load cycle.
			prev_mcnode = prev_mcnode.get_parent()


		# At least the root node will be found on the node pool, so prev_mcnode is not going to be None
		# when exiting the previous loop. Now we now that the embedding of
		# prev_mcnode, which corresponds to the embedding learnt by taking a series of actions, has been loaded.

		for k in range(len(prev_mcnode._act)):
			print('Iter:', stage+k, 'Type:', prev_mcnode._act[k], 'Loaded from pool!')


		# execute the actions that have not been executed so far:
		for k in range(len(hist_act)):

			act = hist_act[k]
			run(stage + len(prev_mcnode._act) + k, act)


		# find the best action from the leaf node and execute it
		action = mcnode.get_max()
		run(stage + len(mcnode._act), action)


		# expand the tree adding the new node
		next_mcnode = mctree.expand(mcnode, action)

		# Evaluate the new embedding on the downstream task to get the current reward
		curvl = linelib.run_classifier_train(classifier, 1, 0.1)
		# set such an inmediate reward as the base of the current Q value estimation for the node.
		next_mcnode.set_base(curvl)

		# calculate priors for the new node
		# notice that the series of actions that constiture the actual state of the node will begin
		# with the action in selected_act_seq
		priors = calculate_priors(selected_act_seq + next_mcnode._act)
		next_mcnode.set_prior(priors)


		# save embeddings for this new node
		add_to_pool(next_mcnode)


		# simulation
		print('----------')

		if depth != 0:

			act_seq = mcnode._act + [action]
			lastvl = curvl
			curdepth = 0

			while True:
				simu_action = calculate_best(selected_act_seq + act_seq)

				run(stage + len(mcnode._act) + curdepth + 1, simu_action)

				act_seq.append(simu_action)

				curvl = linelib.run_classifier_train(classifier, 1, 0.1)

				if curvl - lastvl < penalty:
					break

				lastvl = curvl
				curdepth += 1

				if curdepth == depth:

					break

			curvl = lastvl - penalty * curdepth


		# backup
		current_samples = mctree.update_with_penalty(mcnode, action, curvl, penalty)

		# collect data and train RNN
		current_sample_pool += current_samples



	print('-- Stage:', stage, 'Final')

	# once we have updated the Q values of every traversed node,
	# we can execute the best action from the ROOT node.
	action = mctree.get_root().get_best()

	selected_act_seq.append(action)

	if len(selected_act_seq) > hist_length:
		del selected_act_seq[0]

	if mctree.get_root().get_action_score(action) < 0:
		break

	if search_from_pool(mctree.get_root().get_child(action)) == -1:
		load_from_pool(mctree.get_root())
		run(i, action)
	else:
		print('Iter:', stage, 'Type:', action, 'Load from pool!')
		load_from_pool(mctree.get_root().get_child(action))

	curvl = linelib.run_classifier_train(classifier, 1, 0.1)
	print('!!!!!!!!!!!!!!!!!!!!')
	print('Iter:', stage, 'Final Decision:', action, curvl)
	print('!!!!!!!!!!!!!!!!!!!!')

	# delete other branches
	mctree.derive(action)
	for pst in range(pool_size):
		if mctree.search_id(pointers[pst]) == 0:
			delete_from_pool(pst)
	add_to_pool(mctree.get_root())

	# update prior table
	#print current_sample_pool
	for k in range(20):
		for current_sample in current_sample_pool:
			su, au = current_sample[0][0], current_sample[0][1]
			sv, av = current_sample[1][0], current_sample[1][1]
			reward = current_sample[2]

			if sv == None and av == None:
				target = reward
				rnn.train(selected_act_seq + su, au, target)
			else:
				estimate = rnn.predict(selected_act_seq + sv, av)
				target = reward + estimate
				rnn.train(selected_act_seq + su, au, target)

	# update stage
	stage += 1

linelib.run_classifier_train(classifier, 100, 0.01)
print('Test Accuracy:', linelib.run_classifier_test(classifier))
linelib.write_node_vecs(node, output_file, binary)