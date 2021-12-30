
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

	Returns: if mcnode embeddins are backed up in the pool of embeddings,
	it will return the index of mcnode in such an array,
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
	Adds a new Tree node's embeddings to the current embeddings pool array, unless it already is in it.
	Args:
		mcnode: the nodewhose embeddign we want to add to the embeddings pool

	Returns: the index of the node in the embeddgins pool array if the adding process was succesfull
	(the node was already in the pool  or available space was actually found)
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
	If mcnode's embeddings are backed up in the embeddings pool, then
	load the embeddings from the embeddings pools.
	Args:
		mcnode:

	Returns: the index of the node in the embeddings pool array if such a node is in the node pool. and -1 otherwise.

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

# Each stage consist of:
# 1. a number of MCT searches,
# 2. best action taking, (implies updating the current tree root).
# 3. updating the learning module.
stage = 0

#The actual ending condition of this zero-level loop is chossing an action  (step 2 of each phase)
# whose Q value resulted < 0
while True:

	# init the training samples memory of the learning module as an empty array
	current_sample_pool = []

	# Monte Carlo Tree searches. (we execute "tree_size" tree searches).
	for T in range(tree_size):

		# if, for any MCTS traversal, we reach the maximum tree_size,
		# then we stop the tree searches, and go ahead with the acting and learning phases.
		# of this stage.
		if mctree.get_size() >= tree_size:
			break

		print('-- Stage:', stage, 'Simu:', T)

		# prioritized selection of the leaf node
		# (Tree traversal following the hybrid offline/online
		# version of UCT algorithm based o eq (4) in the paper)
		mcnode = mctree.select()

		#
		# load embeddings
		#
		prev_mcnode = mcnode

		# hist_act will contain a list of actions (in reverse order)
		# that need to be simulated to produce
		# the embedding corresponding to the current leaf node.
		hist_act = []

		# load the most sophisticated embedding from the current tree search
		# that is backed up in the embeddings pool.
		while prev_mcnode != None:

			if load_from_pool(prev_mcnode) != -1:
				# The node's embedding has been found in the embeddins pool,
				# and its embedding has been loaded,
				# so we break the loop and go ahoead. (the current embedding contains the result of
				# every embedding step done previously).
				break

			# The node's embedding has not been found in the embeddgins pool,
			# so we keep in memory the action related to such
			# a node to perform it later, notice we insert always such an action at position zero of the array
			# because we are doing a backward tree traversal, and actions will be executed in the same order
			# as they appear in the array to reach the desired result.
			hist_act.insert(0, prev_mcnode.get_last_action())

			# we go 1 step backwards on the traversal and repeat the embedding load cycle.
			prev_mcnode = prev_mcnode.get_parent()


		# At least the root node's embedding will be found on the embeddings pool,
		# so prev_mcnode is not going to be None when exiting the previous while loop.
		# Now we now that the embedding of
		# prev_mcnode, which corresponds to the most sophistcated embedding backed up in the embeddings pool
		# has been loaded. The following loop outputs all the actions that have produced such a loaded embedding.
		for k in range(len(prev_mcnode._act)):
			print('Iter:', stage+k, 'Type:', prev_mcnode._act[k], 'Loaded from pool!')


		# execute the actions that have not been executed so far:
		# This should imply that the embedding produced by each "run" function call gets loaded.
		for k in range(len(hist_act)):
			act = hist_act[k]
			run(stage + len(prev_mcnode._act) + k, act)

		# When we reach this point, we have loaded the embedding specified in the action sequence array of the
		# current leaf node found. Now we do a further step to expand the tree:
		# find the best action from the leaf node usign ec. (4) and execute it
		# notice that the new embedding should get automatically loaded.
		action = mcnode.get_max()
		run(stage + len(mcnode._act), action)


		# expand the tree adding the new node
		next_mcnode = mctree.expand(mcnode, action)

		# Evaluate the new embedding on the downstream task to get its score.
		# note that, this "run_classifier_train" function call should take the embedding produced by
		# the latest call to the "run" function call, to perform and evaluate the downstream task.
		curvl = linelib.run_classifier_train(classifier, 1, 0.1)
		# set such a score as the "base" value of the node.
		next_mcnode.set_base(curvl)

		# calculate priors for the new node using the learning module,
		# The learning module combines the sequence of actions to form the state vector,
		# such a sequence is not infinite but it takes all the actions performed in the current tree search,
		# which cannot be more than tree_size plus a truncated (hist_length) sequence of the actions taken to
		# generate the current root's embedding.
		# (such a root is updated each time we perform a number of MCT searches and choose the action to take to
		# continue the embedding game (phase 2  of each "stage")
		priors = calculate_priors(selected_act_seq + next_mcnode._act)
		next_mcnode.set_prior(priors)

		# BACKUP the current produced embedding in the embeddings pool.
		# notice this method can fail when there is no space left in the pool.
		# (that is the reason why we made a particular embedding loading cycle before).
		add_to_pool(next_mcnode)


		# "simulation" of "depth" further actions from the NEW node on.
		# notice that this simulation may terminate the current episode.
		# these are done not with the MCT prioritized action sampling but
		# following the current learning module's policy. (on policy)
		# We do that to give more weight to the score of the NEW node if
		# our learning module has learnt to estimate Q values accurately.
		# (might were useful if we give the same weight to the learned Q's and the prior Q in the
		# tree traversal equation (4)).
		# Notice we wont back up any embeddings anymore in this step
		# Notice we wont extend the MC Tree anymore.
		print('----------')

		if depth != 0:

			# create an action sequence sequence array equal to
			# the current NEW node's action seq.
			act_seq = mcnode._act + [action]
			# lastvl will contain the score of the NEW node's embedding
			lastvl = curvl


			curdepth = 0
			while True:

				# get the next action to do according to the learning module
				simu_action = calculate_best(selected_act_seq + act_seq)

				# run such an action (should imply embedding loading)
				run(stage + len(mcnode._act) + curdepth + 1, simu_action)

				# add this action to the state action sequence
				act_seq.append(simu_action)

				#get the score of this new embedding
				curvl = linelib.run_classifier_train(classifier, 1, 0.1)

				if curvl - lastvl < penalty:
					# no further gain has been obtained from the last embedding
					# so we terminate our simulated episode
					break

				# if significant gain has been obtained from the "simulated" action
				# then update the score of the NEW node.
				lastvl = curvl

				# we do this a maximum of "depth" times
				curdepth += 1
				if curdepth == depth:
					break

			# Update the score or base value for the NEW node.
			# Notice that we subtract the penalty each time a action different from the
			# "terminate learning" action that has been taken
			curvl = lastvl - penalty * curdepth


		# Backward pass on the tree updating Q values for nodes following equation (5)
		# current samples will contain the learning samples for the learning module to be trained.
		current_samples = mctree.update_with_penalty(mcnode, action, curvl, penalty)

		# collect data to train the learning module
		current_sample_pool += current_samples
	# End of Monte Carlo Tree searches.


	print('-- Stage:', stage, 'Final')

	# once we have updated the Q values of every traversed node,
	# we can execute the best action from the ROOT node.
	action = mctree.get_root().get_best()

	# As we are going to get rid of the current tree root, we backup the action that produced the new root
	# in this array, which is going to influence the embedding of the state vectors for the nn estimator.
	selected_act_seq.append(action)

	# The action embeddings are produced with a finite lenght sequence of the last taken actions.
	# this selected_act_seq array maintains a fixed length sequence of the actions taken before running the
	# current MCT search. (the actions made for producing the current root node).
	if len(selected_act_seq) > hist_length:
		del selected_act_seq[0]

	# condition for ending the whole training cycle: we have choosen an action whose
	# Q value resulted < 0
	if mctree.get_root().get_action_score(action) < 0:
		break


	if search_from_pool(mctree.get_root().get_child(action)) == -1:
		# if the embeddings of the current chosen root's child are not backed
		# up in the embeddings pool, then load from pool the embeddings of the current root node.
		load_from_pool(mctree.get_root())
		# run the current chosen action using the embedding algorithm.
		# this should imply loading the produced embedding
		run(stage, action)
	else:
		# if instead the embeddings of the action chosen are already in the pool, then we load them.
		print('Iter:', stage, 'Type:', action, 'Loading from pool!')
		load_from_pool(mctree.get_root().get_child(action))


	#Reporting results of the action take by the root node.
	curvl = linelib.run_classifier_train(classifier, 1, 0.1)
	print('!!!!!!!!!!!!!!!!!!!!')
	print('Iter:', stage, 'Final Decision:', action, curvl)
	print('!!!!!!!!!!!!!!!!!!!!')

	# delete other branches (update the root of the tree: the new root is going to be the node chosen
	# by the MCTS algotirhm. Get rid of the previous root)
	mctree.derive(action)


	# delete cached nodes from the pool that are not in the MC Tree anymore
	# (the old root that we have just removed from the tree is also removed from the
	# embeddings pool array
	for pst in range(pool_size):
		if mctree.search_id(pointers[pst]) == 0:
			delete_from_pool(pst)

	# And the new root is going to be added to the embeddings pool.
	add_to_pool(mctree.get_root())


	# UPDATE THE LEARNING MODULE
	# (the parameter of the NN)
	# notice that we run one gradient descent for every sample in the sample pool
	# 20 times.
	for k in range(20):

		for current_sample in current_sample_pool:

			su, au = current_sample[0][0], current_sample[0][1]
			sv, av = current_sample[1][0], current_sample[1][1]
			reward = current_sample[2]

			if sv == None and av == None:
				# We have these kind of tuples when we reach a leaf node and we perform an action
				# reward is the reward of such action, but we have no more infromation about the next state,
				# so we train our NN giving the reward as if it were the target Q value.
				# This training sample is exact when we reach a leaf node and take an action that leads to the end of our episode.
				# (an action whose score gain is less than the action penalty).
				# NOTICE that such an exact, episode-ending training sample could turn CRUCIAL for the learning module convergence.
				target = reward
				rnn.train(selected_act_seq + su, au, target)
			else:
				# Temporal difference learning:
				# Q_l(s_u, a_u) = Q_l(s_u, a_u) + alpha * (R_u + Q_l(s_v,a_v) - Q_l(s_u, a_u))
				# L() = (R_u + Q_l(s_v,a_v) - Q_l(s_u, a_u)
				# w_q <- w_q -learning_rate * dL/dw
				# w_q <- w_q -learning_rate * (R_u + Q_l(s_v,a_v) - Q_l(s_u, a_u) * -dQ/dw
				# w_q <- w_q  + learning_rate * (R_u + Q_l(s_v,a_v) - Q_l(s_u, a_u) * dQ/dw
				# (equation (6)
				# TODO: try to experiment with duelling dqn
				estimate = rnn.predict(selected_act_seq + sv, av)
				target = reward + estimate
				rnn.train(selected_act_seq + su, au, target)

	# update stage
	stage += 1


linelib.run_classifier_train(classifier, 100, 0.01)

print('Test Accuracy:', linelib.run_classifier_test(classifier))

linelib.write_node_vecs(node, output_file, binary)