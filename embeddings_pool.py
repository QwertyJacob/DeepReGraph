import torch

class AdaGAEPool():

    def __init__(self, max_cap):
        self.max_cap = max_cap

        self.encoder_state_pool = [None for k in range(self.max_cap)]
        self.embeddings_pool = [None for k in range(self.max_cap)]

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
