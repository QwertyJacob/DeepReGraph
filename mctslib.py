
'''
The following code was obtained from https://github.com/mnqu/DRL/
were an implementation of the paper "Curriculum Learning for Heterogeneous Star Network Embedding via Deep Reinforcement Learning"
in https://dl.acm.org/doi/10.1145/3159652.3159711 is proposed.
Comments were added by Jesus Cevallos and refer to such a paper.
'''
import math

class Node:
    global_node_id = 0

    def __init__(self, action_size, lamb):
        self._action_size = action_size
        self._lambda = lamb
        self._lact = None
        self._act = []
        self._base = 0
        self._N = [0 for k in range(action_size)]
        self._P = [0 for k in range(action_size)]
        self._Q = [0 for k in range(action_size)]
        self._parent = None
        self._child = [None for k in range(action_size)]

        self._id = Node.global_node_id
        Node.global_node_id += 1

    def get_action_size(self):
        return self._action_size

    def get_last_action(self):
        return self._lact

    def get_best(self):
        bid, bvl = -1, -100000000.0
        for k in range(self._action_size):
            # curvl = (self._P[k] + self._Q[k]) / (self._N[k] + 1.0)
            curvl = 0.0
            if self._N[k] != 0:
                curvl = self._Q[k] / self._N[k]

            if curvl > bvl:
                bid = k
                bvl = curvl
        return bid

    def get_action_score(self, action):
        # return (self._P[action] + self._Q[action]) / (self._N[action] + 1.0)
        if self._N[action] == 0:
            return 0.0
        return self._Q[action] / self._N[action]

    def get_max(self):
        '''
        Returns: the id of the child node whose action maximizes the criterion in equation (4) in the paper
        (combines offline and online input as specified in the reference [8]
        '''
        bid, bvl, SN = -1, -100000000.0, self._action_size
        for k in range(self._action_size):
            SN += self._N[k]
        for k in range(self._action_size):
            curvl = (self._P[k] + self._Q[k]) / (self._N[k] + 1.0)
            curvl += self._lambda * math.sqrt(math.log(SN) / (self._N[k] + 1.0))

            if curvl > bvl:
                bid = k
                bvl = curvl
        return bid

    def add_prior(self, prior):
        '''
        Updates the planning module information
        Args:
            prior:
            the Q values to add to each action
        Returns:

        '''
        for k in range(self._action_size):
            self._P[k] += prior[k]

    def set_prior(self, prior):
        '''
        Sets the initial information in the planning module
        Args:
            prior:
            the Q values to define for each action
        Returns:

        '''
        for k in range(self._action_size):
            self._P[k] = prior[k]

    def add_parent(self, parent):
        self._parent = parent

    def add_child(self, child, action):
        self._child[action] = child

    def get_parent(self):
        return self._parent

    def get_child(self, action):
        return self._child[action]

    def set_base(self, base):
        self._base = base

    def get_base(self):
        return self._base

    def get_action_seq(self):
        '''

        Returns: The action sequence that constitutes the current MDP state.

        '''
        return self._act

    def print_info(self):
        print('--------------------')
        if self._parent == None:
            nid = 'None'
        else:
            nid = self._parent._id
        print('Parent:', nid)
        print('This:', self._id)

        string = ''
        for k in range(self._action_size):
            if self._child[k] == None:
                nid = 'None'
            else:
                nid = str(self._child[k]._id)
            string += nid + ' '
        print('Child:', string)

        print('Last Act:', self._lact)

        string = ''
        for a in self._act:
            string += str(a) + ' '
        print('Act Seq:', string)

        string = ''
        for k in range(self._action_size):
            string += str(self._N[k]) + ' '
        print('N:', string)

        string = ''
        for k in range(self._action_size):
            string += str(self._Q[k]) + ' '
        print('Q:', string)

        string = ''
        for k in range(self._action_size):
            string += str(self._P[k]) + ' '
        print('P:', string)

        SN = self._action_size
        for k in range(self._action_size):
            SN += self._N[k]
        S = []
        for k in range(self._action_size):
            curvl = (self._P[k] + self._Q[k]) / (self._N[k] + 1.0)
            curvl += self._lambda * math.sqrt(math.log(SN) / (self._N[k] + 1.0))
            S.append(curvl)
        string = ''
        for k in range(self._action_size):
            string += str(S[k]) + ' '
        print('SC:', string)
        print('--------------------')




def traverse(root):
    '''
    This method seems to return a list of all the already explored states in the MDP.
    Args:
        root:

    Returns:

    '''
    if root == -1:
        return []
    node_list = []
    que = [root]
    while que != []:
        node = que[0]
        del que[0]
        node_list.append(node)
        for k in range(node._action_size):
            if node._child[k] != None:
                que.append(node._child[k])
    return node_list



class Tree:


    def __init__(self):

        self._action_size = 0
        self._lambda = 1
        self._root = None
        self._nodes = []


    def set_lambda(self, lamb):
        '''
        Lambda is the exploitation hyperparamenter (directly proportional to the resultant exploitation percentage).
        Args:
            lamb:

        Returns:

        '''
        self._lambda = lamb


    def get_lambda(self):
        return self._lambda


    def set_action_size(self, action_size):
        '''
        Sets the current tree action size and then creates
        a Node with the corresponding action size and puts it as the
        root of the current tree.
        Args:
            action_size:

        Returns:

        '''
        self._nodes = []

        self._action_size = action_size
        self._root = Node(action_size, self._lambda)
        self._nodes = [self._root]


    def get_action_size(self):
        return self._action_size


    def get_root(self):
        return self._root


    def get_size(self):
        '''
        Returns: The length of the _nodes list.

        '''
        return len(self._nodes)


    def clear(self):
        '''
        Creates a Node with the corresponding action size and puts it as the
        root of the current tree.
        Returns:

        '''
        self._nodes = []
        self._root = Node(self._action_size, self._lambda)
        self._nodes.append(self._root)


    def select(self):
        '''
        Returns: The first leaf node it founds following a Tree search where the
        actions chosen are those corresponding to eq (4) in the paper.
        '''
        curnode = self._root
        bestid = -1
        while True:
            bestid = curnode.get_max()
            if bestid == -1:
                return curnode
            if curnode._child[bestid] == None:
                return curnode
            else:
                curnode = curnode._child[bestid]


    def expand(self, node, action):
        '''
        Creates and adds a new node to the current Tree.
        Notice that the sequence of actions that constitute the new node are equal to the
        sequence of actions of the parent node plus the action that lead to the creation of the current node.
        Args:
            node: The parent of the node that we are adding
            action: The action that lead to the generation of the new node in the tree.

        Returns: the new created node.

        '''

        child = Node(self._action_size, self._lambda)
        child._lact = action
        child._act = [a for a in node._act]
        child._act.append(action)

        child.add_parent(node)
        node.add_child(child, action)
        self._nodes.append(child)

        return child


    def update(self, node, action, value):
        '''
        Performs a backward tree traversal (from the leaf node to the radix node)
        updating the value of the learning-module Q values and the visit counts for
        each node action pair explored.

        Args:
            node: The leaf node for which we have obtained a new value estimate.
            action: The action that generated the leaf node
            value: The Q value estimate obtained from the learning module for the leaf node.

        Returns:

        '''

        curnode = node
        curaction = action

        while curnode != None:

            curnode._Q[curaction] += value - curnode._base
            curnode._N[curaction] += 1

            curaction = curnode._lact
            curnode = curnode._parent


    def update_with_penalty(self, node, action, value, penalty):
        '''
            Performs a backward tree traversal (from the leaf node to the radix node)
            updating the value of the learning-module Q values and the visit counts for
            each node action pair explored.

            Notice that for each node a penalty is given to balance training efficiency and effectiveness as stated at the
            end of page 3 in the paper.
            Args:
                node: The leaf node for which we have obtained a new value estimate.
                action: The action that we have taken from the leaf node to generate a new node.
                value: The Q value estimate obtained from the learning module for the leaf node.
                penalty: the penalty that balances the effectiveness and efficacy of the algorithm.

            Returns: a list of samples where each sample is a tuple containing 3 inner tuples:
            1. (s_t,a_t)
            2. (s_t+1, a_t+1)
            3. not a tuple but an scalar: R_t + Q_l(s_t+1, a_t+1)  - penalty

            The samples correspond to the tree traversal path realized.
            These samples, IN THIS ORDER, might turn useful to update the experience memory of the learning module and drive the
            optimization of the parameters of the RNN that approximates the Q_l function, using eq. (6) in the paper.
            '''

        current_node = node
        current_action = action
        next_action = None
        sample_list = []

        while current_node != None:

            next_node = current_node._child[current_action]

            if current_node is node:
                gain = value - current_node._base - penalty
                current_node._Q[current_action] += gain
                current_node._N[current_action] += 1


                next_pair = [None, None]
                sample_list.append([current_pair, next_pair, gain])
            else:
                short_gain = value - current_node._base - penalty
                long_gain = next_node._Q[next_action] / next_node._N[next_action]
                gain = short_gain + long_gain
                current_node._Q[current_action] += gain
                current_node._N[current_action] += 1

                current_pair = [current_node._act, current_action]
                next_pair = [next_node._act, next_action]
                sample_list.append([current_pair, next_pair, short_gain])

            value = current_node._base
            next_action = current_action
            current_action = current_node._lact
            current_node = current_node._parent
        return sample_list


    def derive(self, action):
        '''
        Removes a specific action from the action space of the tree,
        This implies removing the root node of the tree and placing the corresponding action node form tha root as the new root.
        For every child node, the paths concerning the specified action are deleted from the tree.
        Args:
            action: -1 if the current root node has not explored the specified action.
            1 if the current root node has explored the specified action and all the removing process has been done

        Returns:

        '''
        if self._root._child[action] == None:
            return -1

        self._root = self._root._child[action]
        self._nodes = traverse(self._root)

        for k in range(len(self._nodes)):
            if self._nodes[k] is self._root:
                self._nodes[k]._lact = -1
                self._nodes[k]._act = []
                self._nodes[k]._parent = None
            else:
                del self._nodes[k]._act[0]
        return 1


    def search(self, node):
        '''

        Args:
            node:

        Returns: 1 if the specific node in input is a node of the current Tree and 0 otherwise.

        '''
        for k in range(len(self._nodes)):
            if self._nodes[k] is node:
                return 1
        return 0


    def search_id(self, node_id):
        '''

        Args:
            node_id: 1 if the specific node id in input corresponds to a node of the current Tree and 0 otherwise.

        Returns:

        '''
        for k in range(len(self._nodes)):
            if self._nodes[k]._id is node_id:
                return 1
        return 0

    def print_info(self):
        for k in range(len(self._nodes)):
            self._nodes[k].print_info()
        print('\n')