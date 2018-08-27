import unicodedata
import string
import re
import random
import time
import math
import os
import sys
import pandas as pd
import numpy as np

import nltk

import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.stanford import StanfordParser
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

import enchant

import torchtext 
from torchtext import data
from torchtext import datasets

# label of dependencies https://nlp.stanford.edu/pubs/USD_LREC14_paper_camera_ready.pdf

DEP_LABELS = ['ROOT', 'ACL','ACVLCL', 'ADVMOD', 'AMOD', 'APPOS', 'AUX', 'CASE', 'CC', 'CCOMP',
               'CLF', 'COMPOUND', 'CONJ', 'COP', 'CSUBJ', 'DEP', 'DET',
               'DISCOURSE', 'DISLOCATED', 'EXPL', 'FIXED', 'FLAT', 'GOESWITH',
               'IOBJ', 'LIST', 'MARK', 'NMOD', 'NSUBJ', 'NUMMOD',
               'OBJ', 'OBL', 'ORPHAN', 'PARATAXIS', 'PUNXT', 'REPARANDUM', 'VOCATIVE',
               'XCOMP']

_DEP_LABELS_DICT = {label:ix for ix, label in enumerate(DEP_LABELS)}

def find_type(type_dep):
    if type_dep=='NSUBJ' or type_dep=='OBJ' or type_dep=='IOBJ' or type_dep=='CSUBJ' or type_dep=='CCOMP' or type_dep == 'XCOMP':
        return 0
    elif type_dep=='OBL' or type_dep=='VOCATIVE' or type_dep=='DISLOCATED' or type_dep=='ADVCL' or type_dep=='ADVMOD' or type_dep=='DISCOURSE' or type_dep=='AUX' or type_dep=='COP' or type_dep=='MARK':
        return 1
    elif type_dep=='NMOD' or type_dep=='APPOS' or type_dep=='NUMMOD' or type_dep=='ACL' or type_dep=='AMOD' or type_dep=='DET' or type_dep=='CLF' or type_dep=='CASE':
        return 2
    else:
        return 3

def get_adj(deps, batch_size, seq_len, max_degree):

    adj_arc_in = np.zeros((batch_size * seq_len, 2), dtype='int32')
    adj_lab_in = np.zeros((batch_size * seq_len, 1), dtype='int32')
    adj_arc_out = np.zeros((batch_size * seq_len * max_degree, 2), dtype='int32')
    adj_lab_out = np.zeros((batch_size * seq_len * max_degree, 1), dtype='int32')


    mask_in = np.zeros((batch_size * seq_len), dtype='float32')
    mask_out = np.zeros((batch_size * seq_len * max_degree), dtype='float32')

    mask_loop = np.ones((batch_size * seq_len, 1), dtype='float32')

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(deps):
        for a, arc in enumerate(de):
            if arc[0] != 'ROOT' and arc[0].upper() in DEP_LABELS:         
                arc_1 = int(arc[2])-1
                arc_2 = int(arc[1])-1
                
                if a in tmp_in:
                    tmp_in[a] += 1
                else:
                    tmp_in[a] = 0

                if arc_2 in tmp_out:
                    tmp_out[arc_2] += 1
                else:
                    tmp_out[arc_2] = 0

                idx_in = (d * seq_len) + a + tmp_in[a]
                idx_out = (d * seq_len * max_degree) + arc_2 * max_degree + tmp_out[arc_2]

                adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                adj_lab_in[idx_in] = np.array([find_type([arc[0].upper()])])  # incoming arcs

                mask_in[idx_in] = 1.

                if tmp_out[arc_2] < max_degree:
                    adj_arc_out[idx_out] = np.array([d, arc_1])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array([find_type([arc[0].upper()])])  # outgoing arcs
                    mask_out[idx_out] = 1.

        tmp_in = {}
        tmp_out = {}

    adj_arc_in = Variable(torch.LongTensor(np.transpose(adj_arc_in)))
    adj_arc_out = Variable(torch.LongTensor(np.transpose(adj_arc_out)))

    adj_lab_in = Variable(torch.LongTensor(np.transpose(adj_lab_in)))
    adj_lab_out = Variable(torch.LongTensor(np.transpose(adj_lab_out)))

    mask_in = Variable(torch.FloatTensor(mask_in.reshape((batch_size * seq_len, 1))))
    mask_out = Variable(torch.FloatTensor(mask_out.reshape((batch_size * seq_len, max_degree))))
    mask_loop = Variable(torch.FloatTensor(mask_loop))

    return adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop

def pad_seq(lang, seq, max_length):
    seq += [lang.vocab.stoi['<pad>'] for i in range(max_length - len(seq))]
    return seq

def generate_batches(input_lang, output_lang, batch_size, pairs, return_dep_tree=False, arr_dep=None, max_degree=None, USE_CUDA=False):
    input_batches = []
    target_batches = []
    
    for pos in range(0, len(pairs), batch_size):
        # Avoiding out of array
        if pos == 10431:
            continue
        cant = min(batch_size, len(pairs) - pos)
        
        input_seqs = pairs[pos:cant+pos, 0]#.tolist()
        target_seqs = pairs[pos:cant+pos, 1]#.tolist()
        if return_dep_tree:
            arr_aux = arr_dep[pos:cant+pos]#.tolist()

        seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)

        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(input_lang, s, max(input_lengths)) for s in input_seqs]
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [pad_seq(output_lang, s, max(target_lengths)) for s in target_seqs]

        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

        if USE_CUDA:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        if return_dep_tree:
            # max len is setting mannually
            adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop = get_adj(arr_aux, cant, max(input_lengths), max_degree)  

            if USE_CUDA:
                adj_arc_in = adj_arc_in.cuda()
                adj_arc_out = adj_arc_out.cuda()
                adj_lab_in = adj_lab_in.cuda()
                adj_lab_out = adj_lab_out.cuda()

                mask_in = mask_in.cuda()
                mask_out = mask_out.cuda()
                mask_loop = mask_loop.cuda()
        else:
            adj_arc_in = None
            adj_arc_out = None
            adj_lab_in = None
            adj_lab_out = None

            mask_in = None
            mask_out = None
            mask_loop = None
                
        input_batches.append([input_var, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop])
        target_batches.append(target_var)

    return input_batches, target_batches

def indexes_from_sentence(lang, sentence):
    return [lang.vocab.stoi[word] for word in sentence.split(' ')] + [lang.vocab.stoi['<eos>']]

def data_to_index(pairs, input_vec, output_vec):
    new_pairs = []
    
    for pair in pairs:
        new_pairs.append([indexes_from_sentence(input_vec, pair[0]), indexes_from_sentence(output_vec, pair[1])])
        
    return np.array(new_pairs)

def construct_vector(pair, name_lang, construct_vector=True, vector_name='fasttext.en.300d'):
    lang = pd.DataFrame(pair, columns=[name_lang])

    lang.to_csv('corpus/' + name_lang + '.csv', index=False)

    lang = data.Field(sequential=True, lower=True, init_token='<sos>', eos_token='<eos>')

    mt_lang = data.TabularDataset(
        path='corpus/' + name_lang + '.csv', format='csv',
        fields=[(name_lang, lang)])

    lang.build_vocab(mt_lang)

    if construct_vector:
        lang.vocab.load_vectors(vector_name)
    
    return lang
            
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(pair):
    pair = unicode_to_ascii(pair.lower().strip())
    pair = re.sub(r'([.,;!?])', r' \1', pair) # separate .!? from words
    
    
    return ' '.join(pair.split())

def normalize_pairs(pairs):
    for pair in pairs:
        pair[0] = normalize_string(pair[0])
        pair[1] = normalize_string(pair[1])

def filter_pairs_lang(pairs, min_length, max_length):
    filtered_pairs = []
    for pair in pairs:
        # Removing '' and "" in pairs, this is for easy processing 
        if len(pair[0].split()) >= min_length and len(pair[0].split()) <= max_length \
            and len(pair[1].split()) >= min_length and len(pair[1].split()) <= max_length \
            and "'" not in pair[0] and '"' not in pair[0]:
                filtered_pairs.append(pair)
    return filtered_pairs

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    filename = f'corpus/{lang1}-{lang2}.txt'
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    return pairs

def prepare_data(lang1_name, lang2_name, reverse=False, min_length=3, max_length=50):
    pairs = read_langs(lang1_name, lang2_name, reverse=reverse)
    print("Read %d sentence pairs" % len(pairs))
    
    pairs = filter_pairs_lang(pairs, min_length, max_length)
    print("Filtered to %d pairs" % len(pairs))
    
    print("Creating vocab...")
    pairs = np.array(pairs)
    vector_1 = construct_vector(pairs[:, 0], lang1_name)
    vector_2 = construct_vector(pairs[:, 1], lang2_name)

    print('Indexed %d words in input language, %d words in output' % (len(vector_1.vocab.itos), len(vector_2.vocab.itos)))
    return vector_1, vector_2, pairs