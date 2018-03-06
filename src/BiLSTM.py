"""

    Bidirectional LSTM module

    author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@ucla.edu>

    This module is for BiLSTM. We use BiLSTM to extract the information
    from the text (title, question)
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gensim


w2v_model = gensim.models.Word2Vec.load_word2vec_format()

class BiLSTMProcessor(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMProcessor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidire


        pass


