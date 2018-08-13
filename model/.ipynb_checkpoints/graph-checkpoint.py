import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter

class Sintactic_GCN(nn.Module):
    def __init__(self, num_inputs, num_units,
                 num_labels,
                 dropout = 0.2,
                 in_arcs = True,
                 out_arcs = True,
                 batch_first = False,
                 USE_CUDA=False):       
        super(Sintactic_GCN, self).__init__()      

        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        
        self.retain = 1. - dropout
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_labels = num_labels
        self.batch_first = batch_first
        
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_rate = dropout
        
        if in_arcs:
            self.V_in = Parameter(torch.FloatTensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.V_in)
            
            self.b_in = Parameter(torch.FloatTensor(num_labels, self.num_units))
            nn.init.constant_(self.b_in, 0)
            
            self.V_in_gate = Parameter(torch.FloatTensor(self.num_inputs, 1))
            nn.init.uniform_(self.V_in_gate)
            
            self.b_in_gate = Parameter(torch.FloatTensor(num_labels, 1))
            nn.init.constant_(self.b_in_gate, 1)

        if out_arcs:
            self.V_out = Parameter(torch.FloatTensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.V_out)
            
            self.b_out = Parameter(torch.FloatTensor(num_labels, self.num_units))
            nn.init.constant_(self.b_in, 0)
            
            self.V_out_gate = Parameter(torch.FloatTensor(self.num_inputs, 1))
            nn.init.uniform_(self.V_out_gate)
            
            self.b_out_gate = Parameter(torch.FloatTensor(num_labels, 1))
            nn.init.constant_(self.b_out_gate, 1)
        
        self.W_self_loop = Parameter(torch.FloatTensor(self.num_inputs, self.num_units))
        nn.init.xavier_normal_(self.W_self_loop)        
        
        self.W_self_loop_gate = Parameter(torch.FloatTensor(self.num_inputs, 1))
        nn.init.uniform_(self.W_self_loop_gate)

    def forward(self, encoder_outputs,
                 arc_tensor_in, arc_tensor_out,
                 label_tensor_in, label_tensor_out,
                 mask_in, mask_out,  # batch* t, degree
                 mask_loop):

        if(not self.batch_first):
            encoder_outputs = encoder_outputs.permute(1, 0, 2).contiguous()
        
        batch_size, seq_len, _ = encoder_outputs.shape
        input_ = encoder_outputs.view((batch_size * seq_len , self.num_inputs))  # [b* t, h]        
        
        max_degree = 1
        if self.in_arcs:
            input_in = torch.mm(input_, self.V_in)  # [b* t, h] * [h,h] = [b*t, h]
            first_in = input_in.index_select(0, arc_tensor_in[0] * seq_len + arc_tensor_in[1])
            
            second_in = self.b_in.index_select(0, label_tensor_in.squeeze(0))  # [b* t* 1, h]
            in_ = (first_in + second_in).view((batch_size, seq_len, 1, self.num_units))

            # compute gate weights
            input_in_gate = torch.mm(input_, self.V_in_gate)  # [b* t, h] * [h,h] = [b*t, h]
            first_in_gate = input_in_gate.index_select(0, arc_tensor_in[0] * seq_len + arc_tensor_in[1])
            
            second_in_gate = self.b_in_gate.index_select(0, label_tensor_in.squeeze(0))
            in_gate = (input_in_gate + second_in_gate).view((batch_size, seq_len, 1))

            max_degree += 1
            
        if self.out_arcs:           
            input_out = torch.mm(input_, self.V_out)  # [b* t, h] * [h,h] = [b* t, h]        
            first_out = input_out.index_select(0, arc_tensor_out[0] * seq_len + arc_tensor_out[1])
        
            second_out = self.b_out.index_select(0, label_tensor_out.squeeze(0))     
            
            degr = int(first_out.shape[0] / batch_size / seq_len)
            max_degree += degr

            out_ = (first_out + second_out).view((batch_size, seq_len, degr, self.num_units))

            # compute gate weights
            input_out_gate = torch.mm(input_, self.V_out_gate)  # [b* t, h] * [h,h] = [b* t, h]
            first_out_gate = input_out_gate.index_select(0, arc_tensor_out[0] * seq_len + arc_tensor_out[1])
            
            second_out_gate = self.b_out_gate.index_select(0, label_tensor_out.squeeze(0))
            
            out_gate = (first_out_gate + second_out_gate).view((batch_size, seq_len, degr))
       
        same_input = torch.mm(encoder_outputs.view(-1,encoder_outputs.size(2)), self.W_self_loop).\
                        view(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        same_input = same_input.view(encoder_outputs.size(0), encoder_outputs.size(1), 1, self.W_self_loop.size(1))
        
        same_input_gate = torch.mm(encoder_outputs.view(-1, encoder_outputs.size(2)), self.W_self_loop_gate)\
                                .view(encoder_outputs.size(0), encoder_outputs.size(1), -1)

        if self.in_arcs and self.out_arcs:
            potentials = torch.cat((in_, out_, same_input), dim=2)  # [b, t,  mxdeg, h]         
            potentials_gate = torch.cat((in_gate, out_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_in, mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
        elif self.out_arcs:
            potentials = torch.cat((out_, same_input), dim=2)  # [b, t,  2*mxdeg+1, h]
            potentials_gate = torch.cat((out_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
        elif self.in_arcs:
            potentials = torch.cat((in_, same_input), dim=2)  # [b, t,  2*mxdeg+1, h]
            potentials_gate = torch.cat((in_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_in, mask_loop), dim=1)  # [b* t, mxdeg]

        potentials_ = potentials.permute(3, 0, 1, 2).contiguous()  # [h, b, t, mxdeg]
        potentials_resh = potentials_.view((self.num_units,
                                               batch_size * seq_len,
                                               max_degree))  # [h, b * t, mxdeg]

        potentials_r = potentials_gate.view((batch_size * seq_len,
                                                  max_degree))  # [h, b * t, mxdeg]
        # calculate the gate
        probs_det_ = self.sigmoid(potentials_r) * mask_soft  # [b * t, mxdeg]
        potentials_masked = potentials_resh * mask_soft * probs_det_  # [h, b * t, mxdeg]

        
        #if self.retain == 1 or deterministic:
        #    pass
        #else:
        #    drop_mask = self._srng.binomial(potentials_resh.shape[1:], p=self.retain, dtype=input.dtype)
        #    potentials_masked /= self.retain
        #    potentials_masked *= drop_mask

        potentials_masked_ = potentials_masked.sum(dim=2)  # [h, b * t]
        potentials_masked_ = self.relu(potentials_masked_)

        result_ = potentials_masked_.permute(1, 0).contiguous()   # [b * t, h]
        result_ = F.dropout(result_, p=self.dropout_rate, training=self.training)
        
        if not self.batch_first:
            result_ = result_.view((seq_len, batch_size, self.num_units))  # [ b, t, h]
        else:
            result_ = result_.view((batch_size, seq_len, self.num_units))

        
        return result_    
