import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
import re 
from tqdm import tqdm
from data import generate_batches

class Beam():
    def __init__(self, decoder_input, decoder_context, decoder_hidden,
                    decoded_words=[], decoder_attentions=[], sequence_log_probs=[], decoded_index=[]):
        self.decoded_words = decoded_words
        self.decoded_index = decoded_index
        self.decoder_attentions = decoder_attentions
        self.sequence_log_probs = sequence_log_probs
        self.decoder_input = decoder_input
        self.decoder_context = decoder_context
        self.decoder_hidden = decoder_hidden

class Evaluator():
    def __init__(self, encoder, decoder, gcn1, gcn2, input_lang, output_lang, max_length, USE_CUDA):
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length
        self.USE_CUDA = USE_CUDA
        self.gcn1 = gcn1
        self.gcn2 = gcn2

    def evaluate(self, input_batch, k_beams, testing_luong=True):
        
        [input_var, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop] = input_batch        
        input_length = input_var.shape[0]
        
        encoder_hidden = self.encoder.init_hidden(1)
        encoder_outputs, encoder_hidden = self.encoder(input_var, encoder_hidden)

        if self.gcn1:
            encoder_outputs = self.gcn1(encoder_outputs,
                                 adj_arc_in, adj_arc_out,
                                 adj_lab_in, adj_lab_out,
                                 mask_in, mask_out,  
                                 mask_loop)
        if self.gcn2:     
            encoder_outputs = self.gcn2(encoder_outputs,
                                 adj_arc_in, adj_arc_out,
                                 adj_lab_in, adj_lab_out,
                                 mask_in, mask_out,  
                                 mask_loop)
            
        if testing_luong:
            decoder_input = Variable(torch.LongTensor([[self.output_lang.vocab.stoi['<sos>']]]))
        else:
            decoder_input = Variable(torch.LongTensor([self.output_lang.vocab.stoi['<sos>']]))
            
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        
        if self.USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            
        decoded_words = []
        decoder_attentions = torch.zeros(self.max_length, self.max_length)

        beams = [Beam(decoder_input, decoder_context, decoder_hidden)]
        top_beams = []
        
        # Use decoder output as inputs
        for di in range(input_length):      
            new_beams = []
            for beam in beams:
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
                    beam.decoder_input, beam.decoder_context, beam.decoder_hidden, encoder_outputs)     

                # Beam search, take the top k with highest probability
                topv, topi = decoder_output.data.topk(k_beams)

                for ni, vi in zip(topi[0], topv[0]):
                    new_beam = Beam(None, decoder_context, decoder_hidden, 
                                    beam.decoded_words[:], beam.decoder_attentions[:], beam.sequence_log_probs[:])
                    new_beam.decoder_attentions.append(decoder_attention.squeeze().cpu().data)
                    new_beam.sequence_log_probs.append(vi)

                    if ni == self.output_lang.vocab.stoi['<eos>'] or ni == self.output_lang.vocab.stoi['<pad>']: 
                        new_beam.decoded_words.append('<eos>')
                        top_beams.append(new_beam)

                    else:
                        new_beam.decoded_words.append(self.output_lang.vocab.itos[ni])                        
                    
                        if testing_luong:
                            decoder_input = Variable(torch.LongTensor([[ni]]))
                        else:
                            decoder_input = Variable(torch.LongTensor([ni]))
                        if self.USE_CUDA: decoder_input = decoder_input.cuda()

                        new_beam.decoder_input = decoder_input                        
                        new_beams.append(new_beam)                    
            
            new_beams = {beam: np.mean(beam.sequence_log_probs) for beam in new_beams}
            beams = sorted(new_beams, key=new_beams.get, reverse=True)[:k_beams]

            if len(beams) == 0:
                break
                
        if len(top_beams) != 0:
            top_beams = {beam: np.mean(beam.sequence_log_probs) for beam in top_beams}
        else:
            top_beams = {beam: np.mean(beam.sequence_log_probs) for beam in new_beams}

        top_beams = sorted(top_beams, key=top_beams.get, reverse=True)[:k_beams]        

        decoded_words = top_beams[0].decoded_words

        return decoded_words, top_beams
    
    def evaluate_sentence(self, input_batch, k_beams=3):        
        output_words, beams = self.evaluate(input_batch, k_beams)
        output_sentence = ' '.join(output_words)
        
        print('>', sentence)
        print('<', output_sentence)
        print('')
    
    def ref_to_string(self, reference):
        aux = ''
        for i in range(len(reference)):
            aux += self.output_lang.vocab.itos[reference[i]] + ' '
        return aux.strip()
    
    def get_candidates_and_references(self, pairs, arr_dep, k_beams=3):
        input_batches, _ = generate_batches(self.input_lang, self.output_lang, 1, pairs, return_dep_tree=True, arr_dep=arr_dep, max_degree=10, USE_CUDA=self.USE_CUDA)

        candidates = [self.evaluate(input_batch, k_beams)[0] for input_batch in tqdm(input_batches)]
        candidates = [' '.join(candidate[:-1]) for candidate in candidates]
        references = pairs[:,1]
        references = [self.ref_to_string(reference) for reference in references]
        return candidates, references