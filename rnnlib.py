
'''
The following code was obtained from https://github.com/mnqu/DRL/
were an implementation of the paper "Curriculum Learning for Heterogeneous Star Network Embedding via Deep Reinforcement Learning"
in https://dl.acm.org/doi/10.1145/3159652.3159711 is proposed.
Comments were added by Jesus Cevallos and refer to such a paper.
'''

import sys
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class RNNRegression(nn.Module):

    def __init__(self, rnn_type, vocab_size, emb_size, hidden_size):
        '''
        Sets the module learning parameters and initializes them, initializes also a RMSProp Torch optimizer on them.
        Args:
            rnn_type: can be rnn or lstm, it uses the corresponding torch.nn modules for the state embedding recursive proto-layer
            vocab_size: is the number of available actions given a state.
            emb_size: The dimension of the vector that encodes both states and actions.
            (notice that states are successions of actions, but with the recursive proto-layer, such successions are represented by
            a vector with the same dimension of each single action embedding vector).
            hidden_size: hidden size for the hidden fully connected layers that take in input state and action and return the
            Q value for such a state-action pair
        '''
        super(RNNRegression, self).__init__()

        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.en_state = nn.Embedding(vocab_size + 1, emb_size)
        self.en_action = nn.Embedding(vocab_size, emb_size)

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_size, emb_size, nonlinearity='relu')
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_size, emb_size)

        self.syn = nn.Linear(2 * emb_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        self.en_state.weight.data.uniform_(-0.1, 0.1)
        self.en_action.weight.data.uniform_(-0.1, 0.1)
        self.syn.weight.data.uniform_(-0.1, 0.1)
        self.out.weight.data.uniform_(-0.1, 0.1)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001)


    def init_hidden(self):
        '''
        Every recursive module has a hidden state. This method initializes such a hidden state with a torch.zero vector.
        Notice that lstm module has two hidden states, the long and the short memory related hidden states, so if the current
        rnn type is lstm, we initialize two hidden state vectors, notice that, in this case,
        the dimension of the hidde state coincides with the dimension of the input layer.
        Returns: the torch.zero vector corresponding to the hidden state vector for the recursive proto-layer.

        '''
        if self.rnn_type == 'rnn':
            return Variable(torch.zeros(1, 1, self.emb_size))

        if self.rnn_type == 'lstm':
            return (Variable(torch.zeros(1, 1, self.emb_size)), Variable(torch.zeros(1, 1, self.emb_size)))


    def forward(self, state, action):
        '''

        Args:
            state:
            action:

        Returns:

        '''
        hidden = self.init_hidden()

        state = Variable(torch.LongTensor(state))
        action = Variable(torch.LongTensor([action]))
        emb_state = self.en_state(state)
        emb_state = emb_state.view(-1, 1, self.emb_size)
        out, hidden = self.rnn(emb_state, hidden)
        if self.rnn_type == 'rnn':
            vec_state = hidden.view(-1)
        if self.rnn_type == 'lstm':
            vec_state = hidden[0].view(1, -1)
        vec_action = self.en_action(action)
        vec = torch.cat([vec_state, vec_action], dim=1)
        vec = F.relu(self.syn(vec))
        pred = self.out(vec)

        return pred, hidden


    def init_parameter(self):

        self.en_state.weight.data.uniform_(-0.1, 0.1)
        self.en_action.weight.data.uniform_(-0.1, 0.1)
        self.syn.weight.data.uniform_(-0.1, 0.1)
        self.out.weight.data.uniform_(-0.1, 0.1)


    def train(self, state, action, target):

        target = Variable(torch.Tensor([target]))

        self.optimizer.zero_grad()

        _output, _hidden = self.forward(state, action)
        loss = (_output - target) * (_output - target)
        loss.backward()
        self.optimizer.step()

        return loss, _output, _hidden


    def predict(self, state, action):

        output = self.forward(state, action)[0]

        return output.data.numpy()[0, 0]
