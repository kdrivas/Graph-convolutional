import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size, n_layers=2, dropout=0.1, lang=None, USE_CUDA=False):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.USE_CUDA = USE_CUDA
        self.lang = lang
        
        self.embedding = nn.Embedding(input_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        self.init_weights()
            
    def forward(self, input_seqs, hidden = None):
        embedded = self.embedding(input_seqs)

        self.gru.flatten_parameters() 
        outputs, hidden = self.gru(embedded, hidden)
        
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden
    
    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size))
        if self.USE_CUDA: hidden = hidden.cuda()
        return hidden
    
    def init_weights(self):
        if self.lang:
            self.embedding.weight.data.copy_(self.lang.vocab.vectors)           
            self.embedding.weight.required_grad = False
            
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)