import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter

######################### ATTENTION ###########################

class Global_attn(nn.Module):
    def __init__(self, method, hidden_size, USE_CUDA=False):
        super(Global_attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.USE_CUDA = USE_CUDA
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))
            
    def forward(self, hidden, encoder_outputs):
        '''
        hidden: (BS, hidden_size)
        encoder_outputs(seq_len, BS, encoder_hidden_size)
        '''
        # encoder_outputs: (seq_len, batch_size, encoder_hidden_size)
        seq_len = len(encoder_outputs)
        batch_size = encoder_outputs.shape[1]
        
        # Calculate attention energies for each encoder output
        # attn_energies: (seq_len, batch_size)
        # hidden: (batch_size, hidden_size)
        attn_energies = Variable(torch.zeros(seq_len, batch_size))
        if self.USE_CUDA: attn_energies = attn_energies.cuda()
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        
        # Normalize energies [0-1] and resize to (batch_size, x=1, seq_len)
        return F.softmax(attn_energies, 0).transpose(0, 1).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        # hidden: (batch_size, hidden_size)
        # encoder_output: (batch_size, encoder_hidden_size)
        
        # hidden sizes must match, batch_size = 1 only
        if self.method == 'dot': 
            # batch element-wise dot product
            energy = torch.bmm(hidden.unsqueeze(1), 
                        encoder_output.unsqueeze(2)).squeeze().squeeze()
            # energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            # batch element-wise dot product
            energy = torch.bmm(hidden.unsqueeze(1), 
                        encoder_output.unsqueeze(2)).squeeze().squeeze()
            # energy = hidden.dot(energy)
            return energy
        
        # TODO: test / modify method to support batch size > 1
        elif self.method == 'concat': 
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy  
            
######################### DECODER  LUONG ###########################

class Decoder_luong(nn.Module):
    def __init__(self, attn_method, hidden_size, output_size, emb_size, n_layers=1, dropout=0.1, lang=None, USE_CUDA=False):
        
        super(Decoder_luong, self).__init__()
        
        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.USE_CUDA = USE_CUDA
        self.lang = lang
        
        # (size of dictionary of embeddings, size of embedding vector)
        self.embedding = nn.Embedding(output_size, emb_size)
        # (input features: hidden_size + emb_size, hidden state features, number of layers)
        self.gru = nn.GRU(emb_size + hidden_size, hidden_size, n_layers, dropout=dropout)
        self.attn = Global_attn(attn_method, hidden_size, USE_CUDA)
        self.out = nn.Linear(hidden_size * 2, output_size)        
        
        self.init_weights()
        
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        '''
        word_input: (seq_len, BS)
        last_context: (BS, encoder_hidden_size)
        last_hidden: (n_layers, BS, hidden_size)
        last_cell: (n_layers, BS, hidden_size)
        encoder_outputs: (seq_len, BS, encoder_hidden)
        < output: (BS, output_size)
        < attn_weights: (BS, 1, seq_len)
        '''
        # This is run one step at a time
        
        # Get the embedding of the current input word (last output word)
        # word_input: (seq_len=1, batch_size), values in [0..output_size)
        word_embedded = self.embedding(word_input) #.view(1, 1, -1)
        # word_embedded: (seq_len=1, batch_size, embedding_size)
        
        # Combine embedded input word and last context, run through RNN
        # last_context: (batch_size, encoder_hidden_size)
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        # rnn_input: (seq_len=1, batch_size, embedding_size + encoder_hidden_size)
        # last_hidden: (num_layers, batch_size, hidden_size)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        # rnn_output: (seq_len=1, batch_size, hidden_size)
        # hidden: same
        
        # Calculate attention and apply to encoder outputs
        # encoder_outputs: (seq_len, batch_size, encoder_hidden_size)
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        
        # Check softmax:
        # print('attn_weights sum: ', torch.sum(attn_weights.squeeze(), 1))
        
        # attn_weights: (batch_size, x=1, seq_len)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context: (batch_size, x=1, encoder_hidden_size)
        
        # Final output layer using hidden state and context vector
        rnn_output = rnn_output.squeeze(0)
        # rnn_output: (batch_size, hidden_size)
        context = context.squeeze(1)
        # context: (batch_size, encoder_hidden_size)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), 1)
        # output: (batch_size, output_size)
        # Check softmax (not log_softmax):
        # print('output sum: ', torch.sum(output.squeeze(), 1))
        
        # Also return attention weights for visualization
        return output, context, hidden, attn_weights
    
    def init_weights(self):
        if self.lang:
            self.embedding.weight.data.copy_(self.lang.vocab.vectors)
            self.embedding.weight.requires_grad = False
            
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.out.bias.data.fill_(0)
        self.out.weight.data.uniform_(-0.1, 0.1)