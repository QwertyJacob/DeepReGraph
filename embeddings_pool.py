import torch

class AdaGAEPool():

    def __init__(self, max_cap):
        self.max_cap = max_cap

        self.encoder_state_pool = [None for k in range(self.max_cap)]
        self.embeddings_pool = [None for k in range(self.max_cap)]

        # pointers is an array containing the ids of the available Tree nodes
        self.pointers = [-1 for k in range(self.max_cap)]


    def save_to_pool(self, position, gae_object):
        gae_state = gae_object.gae_nn.cpu().state_dict()
        gae_embedding = gae_object.gae_nn.embedding.detach().cpu().numpy()

        self.encoder_state_pool.insert(position, gae_state)
        self.embeddings_pool.insert(position, gae_embedding)

        return position


    def load_from_pool(self, position, gae_object):
        gae_nn_device = gae_object.gae_nn.device
        gae_object.gae_nn.load_state_dict(self.encoder_state_pool[position])
        gae_object.gae_nn.to(gae_nn_device)
        gae_object.gae_nn.embedding = torch.Tensor(self.embeddings_pool[position]).to(gae_nn_device)


    def search_from_pool(self, mcnode):
        '''

        Args:
            mcnode: a Node from the tree

        Returns: if mcnode embeddins are backed up in the pool of embeddings,
        it will return the index of mcnode in such an array,
        otherwise it will return -1

        '''
        pointer = mcnode._id
        pst = -1

        for k in range(self.max_cap):
            if pointer == self.pointers[k]:
                pst = k
                break
        return pst


    def add_to_pool(self, mcnode, gae_object):
        '''
        Adds a new Tree node's embeddings to the current embeddings pool array, unless it already is in it.
        Args:
            mcnode: the nodewhose embeddign we want to add to the embeddings pool

        Returns: the index of the node in the embeddgins pool array if the adding process was succesfull
        (the node was already in the pool  or available space was actually found)
        and -1 otherwise.

        '''

        pointer = mcnode._id

        pst = self.search_from_pool(mcnode)

        if pst != -1:
            return pst

        ok = -1

        for pst in range(self.max_cap):

            if self.pointers[pst] == -1:
                self.pointers[pst] = pointer
                self.save_to_pool(pst, gae_object)
                ok = pst
                break

        return ok


    def try_load_from_pool(self, mcnode, gae_object):
        '''
        If mcnode's embeddings are backed up in the embeddings pool, then
        load the embeddings from the embeddings pools.
        Args:
            mcnode:

        Returns: the index of the node in the embeddings pool array if such a node is in the node pool. and -1 otherwise.

        '''
        pst = self.search_from_pool(mcnode)

        if pst != -1:
            assert self.embeddings_pool[pst] is not None
            assert self.encoder_state_pool[pst] is not None

            self.load_from_pool(pst, gae_object)

        return pst


    def delete_from_pool(self, pst):
        '''
        Deletes the node whose index is pst from the pool array.
        Args:
            pst:

        Returns:

        '''
        if pst >= 0 and pst < self.max_cap:

            self.embeddings_pool[pst] = None
            self.encoder_state_pool[pst] = None
            self.pointers[pst] = -1
